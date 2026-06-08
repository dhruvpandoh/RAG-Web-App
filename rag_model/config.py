import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "ros2_rag")
    raw_collection: str = os.getenv("RAW_COLLECTION", "raw_documents")
    chunk_collection: str = os.getenv("CHUNK_COLLECTION", "document_chunks")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "ros2_knowledge_base")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Default is runnable. For LLaMA/LoRA runs, set HF_MODEL_NAME and LORA_ADAPTER_PATH.
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
    lora_adapter_path: str | None = os.getenv("LORA_ADAPTER_PATH") or None
    huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN") or None

    clearml_project: str = os.getenv("CLEARML_PROJECT", "ROS2 RAG Web Application")
    clearml_task_prefix: str = os.getenv("CLEARML_TASK_PREFIX", "ros2-rag")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_pages_per_source: int = int(os.getenv("MAX_PAGES_PER_SOURCE", "25"))


settings = Settings()