import hashlib
import logging
from datetime import datetime

from pymongo import MongoClient, UpdateOne
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from clearml import Task
except Exception:
    Task = None

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def semantic_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Paragraph-aware semantic chunking with overlap.

    This keeps related ROS2 concepts together better than naive sentence splitting,
    while preserving enough overlap for retrieval context continuity.
    """
    normalized = " ".join(text.split())

    if not normalized:
        return []

    chunks = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        candidate = normalized[start:end]

        if end < len(normalized):
            sentence_boundary = max(candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "))
            if sentence_boundary > int(chunk_size * 0.45):
                end = start + sentence_boundary + 1
                candidate = normalized[start:end]

        candidate = candidate.strip()

        if len(candidate) >= 120:
            chunks.append(candidate)

        if end >= len(normalized):
            break

        start = max(0, end - overlap)

    return chunks


def create_qdrant_collection(qdrant: QdrantClient, vector_size: int) -> None:
    existing_collections = [c.name for c in qdrant.get_collections().collections]

    if settings.qdrant_collection not in existing_collections:
        qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", settings.qdrant_collection)
    else:
        logger.info("Using existing Qdrant collection: %s", settings.qdrant_collection)


def run_featurization() -> dict:
    task = None
    if Task is not None:
        try:
            task = Task.init(
                project_name=settings.clearml_project,
                task_name=f"{settings.clearml_task_prefix}-featurization",
                reuse_last_task_id=False,
            )
        except Exception as exc:
            logger.warning("ClearML tracking not initialized: %s", exc)

    mongo_client = MongoClient(settings.mongo_uri)
    db = mongo_client[settings.mongo_db]
    raw_collection = db[settings.raw_collection]
    chunk_collection = db[settings.chunk_collection]

    embedding_model = SentenceTransformer(settings.embedding_model)
    vector_size = embedding_model.get_sentence_embedding_dimension()

    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )

    create_qdrant_collection(qdrant, vector_size)

    raw_documents = list(raw_collection.find())
    logger.info("Loaded %d raw documents", len(raw_documents))

    points_buffer = []
    mongo_operations = []
    total_chunks = 0

    for document in tqdm(raw_documents, desc="Chunking + embedding documents"):
        content = document.get("content", "")
        chunks = semantic_chunk_text(
            content,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(chunks):
            unique_key = f"{document.get('content_hash', document.get('_id'))}:{chunk_index}:{chunk_text}"
            vector_id = chunk_hash(unique_key)

            embedding = embedding_model.encode(
                chunk_text,
                normalize_embeddings=True,
            ).tolist()

            payload = {
                "text": chunk_text,
                "source": document.get("source"),
                "source_type": document.get("source_type"),
                "url": document.get("url"),
                "chunk_index": chunk_index,
                "created_at": datetime.utcnow().isoformat(),
                "document_hash": document.get("content_hash"),
            }

            points_buffer.append(
                PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload=payload,
                )
            )

            mongo_operations.append(
                UpdateOne(
                    {"chunk_id": vector_id},
                    {"$set": {"chunk_id": vector_id, **payload}},
                    upsert=True,
                )
            )

            total_chunks += 1

            if len(points_buffer) >= 128:
                qdrant.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points_buffer,
                )
                points_buffer = []

    if points_buffer:
        qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points_buffer,
        )

    if mongo_operations:
        chunk_collection.bulk_write(mongo_operations)

    metrics = {
        "raw_documents": len(raw_documents),
        "chunks_indexed": total_chunks,
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
    }

    logger.info("Featurization metrics: %s", metrics)

    if task is not None:
        try:
            task.get_logger().report_scalar(
                title="Featurization",
                series="chunks_indexed",
                value=total_chunks,
                iteration=0,
            )
            task.close()
        except Exception:
            pass

    return metrics


if __name__ == "__main__":
    print(run_featurization())