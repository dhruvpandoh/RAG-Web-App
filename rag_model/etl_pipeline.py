import hashlib
import logging
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from clearml import Task
except Exception:
    Task = None

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROS2_DOCUMENTATION_SOURCES = {
    "ros2_docs": "https://docs.ros.org/en/foxy/",
    "nav2_docs": "https://docs.nav2.org/",
    "moveit2_docs": "https://moveit.picknik.ai/main/index.html",
    "gazebo_docs": "https://gazebosim.org/docs/latest/getstarted/",
}

GITHUB_REPOSITORIES = [
    "https://github.com/ros2/ros2",
    "https://github.com/ros-navigation/navigation2",
    "https://github.com/moveit/moveit2",
    "https://github.com/gazebosim/gz-sim",
]

# Add real ROS2 tutorial video IDs when running the YouTube ingestion path.
YOUTUBE_VIDEO_IDS = []


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def is_internal_url(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc == base_domain


def scrape_html_page(url: str) -> tuple[str, list[str]]:
    headers = {"User-Agent": "ROS2-RAG-Web-App/1.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = normalize_text(soup.get_text(separator=" "))
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    return text, links


def crawl_documentation(start_url: str, source_name: str, max_pages: int) -> list[dict]:
    base_domain = urlparse(start_url).netloc
    queue = [start_url]
    visited = set()
    documents = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:
            text, links = scrape_html_page(url)

            if len(text) > 300:
                documents.append(
                    {
                        "source": source_name,
                        "source_type": "documentation",
                        "url": url,
                        "content": text,
                        "content_hash": stable_hash(text),
                        "metadata": {
                            "domain": base_domain,
                            "scraped_at": datetime.utcnow().isoformat(),
                        },
                    }
                )

            for link in links:
                clean_link = link.split("#")[0]
                if (
                    clean_link not in visited
                    and is_internal_url(clean_link, base_domain)
                    and len(queue) < max_pages * 4
                ):
                    queue.append(clean_link)

            logger.info("Scraped documentation page: %s", url)
            time.sleep(0.25)

        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", url, exc)

    return documents


def ingest_github_repositories(repo_urls: list[str]) -> list[dict]:
    """
    Lightweight GitHub ingestion path.

    This stores repository-level metadata and README text where available.
    It can be extended to ingest docs folders, issues, discussions, and source files.
    """
    documents = []

    for repo_url in repo_urls:
        try:
            owner_repo = repo_url.replace("https://github.com/", "").strip("/")
            api_url = f"https://api.github.com/repos/{owner_repo}/readme"

            response = requests.get(
                api_url,
                headers={"Accept": "application/vnd.github.raw"},
                timeout=20,
            )

            if response.status_code == 200:
                content = normalize_text(response.text)
            else:
                content = f"ROS2-related GitHub repository used as ingestion source: {repo_url}"

            documents.append(
                {
                    "source": "github",
                    "source_type": "github_repository",
                    "url": repo_url,
                    "content": content,
                    "content_hash": stable_hash(content + repo_url),
                    "metadata": {
                        "repo": owner_repo,
                        "ingested_at": datetime.utcnow().isoformat(),
                    },
                }
            )

            logger.info("Ingested GitHub source: %s", repo_url)

        except Exception as exc:
            logger.warning("Failed GitHub ingestion for %s: %s", repo_url, exc)

    return documents


def ingest_youtube_transcripts(video_ids: list[str]) -> list[dict]:
    documents = []

    for video_id in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            content = normalize_text(" ".join(item["text"] for item in transcript))

            documents.append(
                {
                    "source": "youtube",
                    "source_type": "youtube_transcript",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "content": content,
                    "content_hash": stable_hash(content),
                    "metadata": {
                        "video_id": video_id,
                        "ingested_at": datetime.utcnow().isoformat(),
                    },
                }
            )

            logger.info("Ingested YouTube transcript: %s", video_id)

        except Exception as exc:
            logger.warning("Failed YouTube transcript ingestion for %s: %s", video_id, exc)

    return documents


def upsert_documents(documents: list[dict]) -> int:
    client = MongoClient(settings.mongo_uri)
    collection = client[settings.mongo_db][settings.raw_collection]

    operations = [
        UpdateOne(
            {"content_hash": document["content_hash"]},
            {"$set": document},
            upsert=True,
        )
        for document in documents
    ]

    if not operations:
        return 0

    result = collection.bulk_write(operations)
    return result.upserted_count + result.modified_count


def run_etl() -> dict:
    task = None
    if Task is not None:
        try:
            task = Task.init(
                project_name=settings.clearml_project,
                task_name=f"{settings.clearml_task_prefix}-etl",
                reuse_last_task_id=False,
            )
        except Exception as exc:
            logger.warning("ClearML tracking not initialized: %s", exc)

    all_documents = []

    for source_name, source_url in ROS2_DOCUMENTATION_SOURCES.items():
        all_documents.extend(
            crawl_documentation(
                start_url=source_url,
                source_name=source_name,
                max_pages=settings.max_pages_per_source,
            )
        )

    all_documents.extend(ingest_github_repositories(GITHUB_REPOSITORIES))
    all_documents.extend(ingest_youtube_transcripts(YOUTUBE_VIDEO_IDS))

    bytes_processed = sum(len(doc["content"].encode("utf-8")) for doc in all_documents)
    written_count = upsert_documents(all_documents)

    metrics = {
        "documents_seen": len(all_documents),
        "documents_written": written_count,
        "mb_processed": round(bytes_processed / (1024 * 1024), 2),
        "sources": list(ROS2_DOCUMENTATION_SOURCES.keys()) + ["github", "youtube"],
    }

    logger.info("ETL metrics: %s", metrics)

    if task is not None:
        try:
            task.get_logger().report_scalar(
                title="ETL",
                series="documents_written",
                value=written_count,
                iteration=0,
            )
            task.get_logger().report_scalar(
                title="ETL",
                series="mb_processed",
                value=metrics["mb_processed"],
                iteration=0,
            )
            task.close()
        except Exception:
            pass

    return metrics


if __name__ == "__main__":
    print(run_etl())