# RAG System for ROS2 Subdomains

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system to assist developers in ROS2 subdomains, such as:

- **ros2**: Robotics middleware
- **nav2**: Navigation
- **moveit2**: Motion planning
- **gazebo**: Simulation

The system combines retrieval-based and generation-based approaches to answer domain-specific questions using MongoDB Atlas, Qdrant, and T5-based fine-tuned models.

## Features

- **ETL Pipeline**: Extracts raw data from GitHub (ROS2 documentation) and YouTube, processes it, and loads it into MongoDB Atlas.
- **Featurization Pipeline**: Converts raw data into vector embeddings using SentenceTransformers and stores them in Qdrant for fast retrieval.
- **Fine-tuned Model**: Uses a T5-small model fine-tuned on instruction datasets for domain-specific question-answering.
- **Interactive API**: A FastAPI backend to process queries and return relevant answers.
- **Gradio UI**: An interactive web interface for testing the RAG system.

## Technologies Used

- **Python**: Core programming language
- **FastAPI**: Backend framework for APIs
- **MongoDB Atlas**: Cloud-based NoSQL database
- **Qdrant Cloud**: Vector search engine
- **SentenceTransformers**: For embedding generation
- **Huggingface Transformers**: For T5 fine-tuning
- **ClearML**: Experiment tracking and orchestration
- **Gradio**: Web interface for query testing

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account (Sign up [here](https://www.mongodb.com/cloud/atlas))
- Qdrant Cloud account (Sign up [here](https://qdrant.tech/))
- Docker (optional, for containerization)

### 2. Clone the Repository

```bash
!git clone https://github.com/ARDA7787/rag_model.git
cd rag_model
```

3. Install Dependencies

```bash
pip install -r app/requirements.txt
```

4. Configure Environment Variables
Create a .env file in the project root and add the following:

```bash
MONGO_URI=mongodb+srv://and8995:Aniks777@cluster0.4voet.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
QDRANT_URL="https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY=rzwHZa71bmoNZzJ2YlEvuwWoH8-2WifxSLywPqZc-o8zkaKilb3z1w
```
5. Run the ETL Pipeline

This will scrape the documentation and store it in MongoDB.

```bash
python app/etl_pipeline.py
```

6. Run the Featurization Pipeline

This will process the scraped data and store embeddings in Qdrant.

```bash
python app/featurization_pipeline.py
```

7. Start the FastAPI Server

This will start the API that serves the RAG functionality.

```bash
python app/main.py
```

8. Start the Gradio Interface

This will start the web interface for user interaction.

```bash
python app/gradio_ui.py
```

9. Access the Application

- **FastAPI**: Open your browser and go to `http://localhost:8000/docs` to see the API documentation and test the endpoints.
- **Gradio**: Open your browser and go to `http://localhost:7860` to access the Gradio interface.

## Future Improvements

Fine-tuning: Enhance the T5 model with additional domain-specific instruction datasets.
Interactive UI: Add more interactivity and predefined questions for ease of use.
Scalability: Expand the system to handle larger datasets and more complex queries.



## Docker Setup

### 1. Build the Docker Image

To build the Docker image for the application, run the following command in the project root directory:

```bash
docker build -t rag_system .
```

### 2. Run the Docker Containers

You can use `docker-compose` to run the application and its dependencies. Make sure you have Docker and Docker Compose installed, then run:

```bash
docker-compose up
```

This command will start the FastAPI server and the ClearML service as defined in the `docker-compose.yml` file.

### 3. Access the Application

- **FastAPI**: Open your browser and go to `http://localhost:8000/docs` to see the API documentation and test the endpoints.
- **Gradio**: Open your browser and go to `http://localhost:7860` to access the Gradio interface.

