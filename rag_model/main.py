import logging
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class RetrievedContext(BaseModel):
    text: str
    source: str | None = None
    source_type: str | None = None
    url: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[RetrievedContext]


class LoraCompatibleGenerator:
    """
    Context-grounded generator.

    Defaults to a small public model for easy local execution. For the LLaMA + LoRA
    project path, set HF_MODEL_NAME and LORA_ADAPTER_PATH in the environment.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.hf_model_name,
            token=settings.huggingface_token,
        )

        self.model_type = "seq2seq"

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.hf_model_name,
                token=settings.huggingface_token,
            )
            self.model_type = "seq2seq"
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.hf_model_name,
                token=settings.huggingface_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model_type = "causal"

        if settings.lora_adapter_path and PeftModel is not None:
            logger.info("Loading LoRA adapter: %s", settings.lora_adapter_path)
            self.model = PeftModel.from_pretrained(self.model, settings.lora_adapter_path)

        self.model.eval()

    @staticmethod
    def build_prompt(question: str, contexts: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{idx + 1}] source={ctx.get('source')} url={ctx.get('url')}\n{ctx.get('text')}"
            for idx, ctx in enumerate(contexts)
        )

        return f"""
You are a ROS2 technical support assistant.

Use only the retrieved context to answer. If the answer is not supported by
the context, say what information is missing.

Question:
{question}

Retrieved context:
{context_block}

Answer:
""".strip()

    def generate(self, question: str, contexts: list[dict]) -> str:
        prompt = self.build_prompt(question, contexts)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False,
                num_beams=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(settings.embedding_model)

        self.qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )

        self.generator = LoraCompatibleGenerator()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        query_vector = self.embedder.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        results = self.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k or settings.top_k,
        )

        contexts = []

        for result in results:
            payload = result.payload or {}
            contexts.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source"),
                    "source_type": payload.get("source_type"),
                    "url": payload.get("url"),
                    "score": result.score,
                    "metadata": {
                        "chunk_index": payload.get("chunk_index"),
                        "document_hash": payload.get("document_hash"),
                    },
                }
            )

        return contexts

    def answer(self, question: str, top_k: int | None = None) -> dict:
        contexts = self.retrieve(question, top_k)
        answer = self.generator.generate(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "retrieved_context": contexts,
        }


app = FastAPI(
    title="RAG Web Application for ROS2 Support",
    version="1.0.0",
    description="FastAPI backend for ROS2 retrieval-augmented generation.",
)

pipeline: RAGPipeline | None = None


@app.on_event("startup")
def startup_event() -> None:
    global pipeline
    pipeline = RAGPipeline()


@app.get("/")
def root() -> dict:
    return {
        "service": "RAG Web Application for ROS2 Support",
        "status": "running",
        "docs": "/docs",
        "query_endpoint": "/query",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "generator_model": settings.hf_model_name,
        "lora_adapter_enabled": bool(settings.lora_adapter_path),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return pipeline.answer(question, request.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)