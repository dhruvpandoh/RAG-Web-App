# RAG Web Application for ROS2 Support

A Retrieval-Augmented Generation system built to help developers answer technical questions across ROS2-related subdomains such as robotics middleware, navigation, motion planning, and simulation.

The application combines multi-source data ingestion, semantic chunking, dense-vector retrieval, and a fine-tuned language model to produce context-grounded answers for ROS2 developer workflows.

## Overview

ROS2 developers often need to search across documentation, GitHub repositories, tutorials, and community resources to debug issues or understand how different packages work. This project addresses that problem by building a RAG-based question-answering system for ROS2 support.

The system ingests data from ROS2 documentation, GitHub, and YouTube, processes and deduplicates the content, converts it into embeddings, stores the vectors in Qdrant, and retrieves relevant context at query time. A LoRA fine-tuned LLaMA-based generator then uses the retrieved context to produce precise, domain-specific answers.

## Supported ROS2 Subdomains

The system is designed around common ROS2 developer areas, including:

- **ros2**: Core robotics middleware and communication patterns
- **nav2**: Navigation stack, path planning, localization, and behavior trees
- **moveit2**: Motion planning, robot manipulation, and planning pipelines
- **gazebo**: Simulation environments, robot models, and testing workflows

## Key Features

### Multi-Source ETL Pipeline

The project includes an ETL pipeline that extracts raw technical content from:

- ROS2 documentation
- GitHub repositories
- YouTube technical/tutorial content

The pipeline processes over **1GB of data per day**, applies cleaning and deduplication, and prepares the content for downstream retrieval and generation.

### Semantic Chunking and Featurization

The featurization pipeline converts cleaned documents into semantic chunks and generates dense-vector embeddings using SentenceTransformers. These embeddings are stored in Qdrant to support fast similarity search over a 10k+ document knowledge base.

### Dense-Vector Retrieval with Qdrant

Qdrant is used as the vector database for efficient semantic retrieval. Given a user query, the system retrieves the most relevant chunks from the indexed ROS2 knowledge base and passes them as grounding context to the generator.

### LoRA Fine-Tuned LLaMA Generator

The generation layer uses a LoRA fine-tuned LLaMA-based model to answer developer questions using retrieved context. This improves domain relevance and helps produce answers that are grounded in the source material instead of relying only on the base model’s general knowledge.

### Async FastAPI Backend

The backend is built with FastAPI and exposes API endpoints for submitting questions, retrieving relevant context, and generating answers. The async service design helps support responsive query handling.

### Gradio Interface

A Gradio UI is included for interactive testing. Users can enter ROS2-related questions and view generated answers through a simple web interface.

### ClearML Experiment Tracking

ClearML is used to track experiments, model runs, pipeline behavior, and system iterations during development.

## Tech Stack

- **Python**: Core application logic and ML pipelines
- **FastAPI**: Async backend API
- **Gradio**: Interactive web interface
- **MongoDB Atlas**: Raw and processed document storage
- **Qdrant Cloud**: Dense-vector retrieval
- **SentenceTransformers**: Embedding generation
- **Hugging Face Transformers**: Model loading and fine-tuning
- **LoRA**: Parameter-efficient fine-tuning
- **LLaMA**: Generator model
- **ClearML**: Experiment tracking and orchestration
- **Docker**: Containerized deployment support

## System Architecture

```text
Data Sources
   ├── ROS2 Documentation
   ├── GitHub Repositories
   └── YouTube Content
            │
            ▼
ETL Pipeline
   ├── Extract raw technical content
   ├── Clean and deduplicate documents
   └── Store processed data in MongoDB Atlas
            │
            ▼
Featurization Pipeline
   ├── Apply semantic chunking
   ├── Generate embeddings using SentenceTransformers
   └── Store dense vectors in Qdrant
            │
            ▼
RAG Query Pipeline
   ├── Receive user query through FastAPI or Gradio
   ├── Retrieve relevant chunks from Qdrant
   ├── Pass retrieved context to LoRA fine-tuned LLaMA
   └── Generate a context-grounded ROS2 answer
            │
            ▼
Application Layer
   ├── FastAPI backend for API access
   ├── Gradio UI for interactive testing
   └── ClearML for experiment tracking
```

## Project Workflow

1. **Data Ingestion**  
   The ETL pipeline collects ROS2-related content from documentation, GitHub, and YouTube sources.

2. **Cleaning and Deduplication**  
   Raw data is cleaned, normalized, deduplicated, and prepared for chunking.

3. **Semantic Chunking**  
   Documents are split into meaningful chunks so that retrieved context remains focused and useful.

4. **Embedding Generation**  
   SentenceTransformers are used to convert each chunk into dense-vector embeddings.

5. **Vector Storage**  
   Embeddings are stored in Qdrant for similarity-based retrieval.

6. **Query Processing**  
   A user submits a ROS2-related question through the FastAPI endpoint or Gradio UI.

7. **Context Retrieval**  
   The system retrieves the most relevant chunks from Qdrant.

8. **Answer Generation**  
   The retrieved context is passed to the LoRA fine-tuned LLaMA generator to produce a grounded answer.

9. **Experiment Tracking**  
   ClearML tracks pipeline runs, model behavior, and experiments.

## Repository Setup

### 1. Prerequisites

Make sure you have the following installed:

    python 3.8+
    pip
    git
    docker

You will also need accounts or credentials for:

- MongoDB Atlas
- Qdrant Cloud
- Hugging Face
- ClearML

## 2. Clone the Repository

    git clone https://github.com/dhruvpandoh/RAG-Web-App.git
    cd RAG-Web-App

## 3. Install Dependencies

    pip install -r rag_model/requirements.txt

## 4. Configure Environment Variables

The application uses environment variables loaded through `rag_model/config.py`.

Create a `.env` file in the project root or export the variables directly:

    MONGO_URI=mongodb://localhost:27017
    MONGO_DB=ros2_rag
    RAW_COLLECTION=raw_documents
    CHUNK_COLLECTION=document_chunks

    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=<your_qdrant_api_key_if_using_cloud>
    QDRANT_COLLECTION=ros2_knowledge_base

    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

    HF_MODEL_NAME=google/flan-t5-base
    LORA_ADAPTER_PATH=<optional_lora_adapter_path>
    HUGGINGFACE_TOKEN=<your_huggingface_token_if_needed>

    CLEARML_PROJECT=ROS2 RAG Web Application
    CLEARML_TASK_PREFIX=ros2-rag

    TOP_K=5
    CHUNK_SIZE=900
    CHUNK_OVERLAP=150
    MAX_PAGES_PER_SOURCE=25

For LLaMA + LoRA experiments, set `HF_MODEL_NAME`, `LORA_ADAPTER_PATH`, and `HUGGINGFACE_TOKEN` as needed.

## 5. Run the ETL Pipeline

This extracts ROS2-related content, processes it, and stores it in MongoDB.

    python rag_model/etl_pipeline.py

## 6. Run the Featurization Pipeline

This performs semantic chunking, generates embeddings, and stores vectors in Qdrant.

    python rag_model/featurization_pipeline.py

## 7. Start the FastAPI Backend

    python rag_model/main.py

Then open:

    http://localhost:8000/docs

## 8. Start the Gradio Interface

    python rag_model/gradio_ui.py

Then open:

    http://localhost:7860

## Example Usage

Example questions the system can answer:

    How do I create a ROS2 publisher and subscriber in Python?

    What is the difference between nav2 costmaps and planners?

    How do I configure MoveIt2 for a custom robot arm?

    How can I simulate a robot in Gazebo with ROS2?

## API Example

Example request:

    curl -X POST "http://localhost:8000/query" \
      -H "Content-Type: application/json" \
      -d '{
        "question": "How do I create a ROS2 publisher in Python?"
      }'

Example response format:

    {
      "question": "How do I create a ROS2 publisher in Python?",
      "answer": "Generated answer based on retrieved ROS2 context.",
      "retrieved_context": [
        {
          "source": "ROS2 documentation",
          "content": "Relevant retrieved chunk..."
        }
      ]
    }

## Docker Setup

### Build the Docker Image

From the project root:

    docker build -t rag-web-app -f rag_model/dockerfile rag_model

### Run with Docker Compose

From the project root:

    docker compose -f rag_model/docker-compose.yml up --build

Alternatively, from inside the `rag_model` directory:

    cd rag_model
    docker compose up --build

This starts the FastAPI backend, Gradio interface, MongoDB, and Qdrant services defined in `rag_model/docker-compose.yml`.

## Evaluation and Tracking

ClearML is used to track experiments and pipeline runs, including:

- ETL execution
- Featurization runs
- Embedding generation
- Retrieval experiments
- Fine-tuning experiments
- Model performance comparisons

## Future Improvements

- Expand the knowledge base with more ROS2 community resources
- Improve retrieval ranking with hybrid search
- Add citation links for every generated answer
- Add evaluation metrics for answer faithfulness and retrieval quality
- Improve the Gradio UI with predefined ROS2 debugging workflows
- Deploy the backend and UI as a production-ready cloud service
- Add user feedback loops to improve retrieval and generation quality over time

## Project Highlights

- Built a Python RAG web application for ROS2 support
- Indexed a 10k+ document knowledge base using Qdrant dense-vector retrieval
- Designed a multi-source ETL pipeline across ROS2 docs, GitHub, and YouTube
- Processed over 1GB/day with deduplication and semantic chunking
- Integrated a LoRA fine-tuned LLaMA generator for context-grounded answers
- Deployed an async FastAPI backend with a Gradio interface
- Used ClearML for experiment tracking and orchestration
