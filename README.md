# RAG Chatbot PoC — LangChain · Qdrant · E5 · Mistral · FastAPI · Airflow

> **Beginner-friendly, step-by-step Retrieval-Augmented Generation (RAG) project.**  
> Scrape → Chunk (semantic + 20% overlap) → Embed (E5) → Store (Qdrant) → Retrieve → Augment → Answer (Mistral 7B via Ollama).

---

## 🧭 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quickstart (Basic RAG, no Airflow)](#quickstart-basic-rag-no-airflow)
- [Configuration](#configuration)
- [Add Airflow for One-Click Re-ingest](#add-airflow-for-one-click-re-ingest)
  - [Build Airflow Image with Deps](#build-airflow-image-with-deps)
  - [Compose mounts & build & env](#compose-mounts--build--env)
  - [Start Airflow](#start-airflow)
- [License](#license)

---

## Overview

This repo contains a small, **production-flavored PoC** for a RAG chatbot:

- **LangChain** to orchestrate the retrieval + generation pipeline  
- **Qdrant** as the vector DB (cosine similarity)  
- **E5-base-v2** embeddings (“query:” / “passage:” prefixes)  
- **Mistral 7B** LLM served locally via **Ollama**  
- **FastAPI** HTTP endpoint (`/chat`)  
- **Airflow** DAG to re-ingest/experiment quickly

You can start with a **simple manual pipeline**, then layer in **Airflow** to automate re-indexing as you tweak chunk sizes, overlap, or source URLs.

---

## Architecture

![architecture.png](docs%2Farchitecture.png)


#### Key choices
- **Semantic chunking + overlap** keeps sentences intact and reduces context loss.  
- **MMR retriever (k=4, fetch_k=20, λ=0.5)** reduces redundancy in returned chunks.  
- **E5** uses `query:` and `passage:` prefixes; embeddings are **normalized** for cosine sim.




## Project Layout

```text
rag-poc/
  .env
  requirements.txt              # pinned versions for the project
  dags/
    rag_ingest_dag.py            # Airflow DAG (manual trigger)
  pipeline/
    __init__.py
    scrape.py                    # fetch+clean HTML
    build_index.py               # chunk → embed → upsert to Qdrant
  app/
    __init__.py
    embeddings_e5.py             # E5 wrapper (query/passages + normalize)
    rag_chain.py                 # retriever + prompt assembler + LLM
    main.py                      # FastAPI endpoints
```


## Prerequisites

- **Python 3.10+** (project tested with **3.12**)
- **Docker Desktop** (Windows / macOS / Linux)
- **Qdrant** (Docker container)
- **Ollama** with **Mistral 7B** quantized model


## Install / prepare
### Create and activate a venv
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Python deps (host, for running manual pipeline + API)
pip install --upgrade pip
pip install -r requirements.txt
```


### Start Qdrant (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage --name qdrant -d qdrant/qdrant:latest
```


### Install Ollama and pull a small Mistral quant that fits typical memory:

```bash
ollama pull mistral:7b-instruct-q4_K_M
```
If you run low on RAM, use a smaller quant and reduce OLLAMA_NUM_CTX (e.g., 1024).


## Quickstart (Basic RAG, no Airflow)

### Create a `.env` at the repo root:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_poc
DOC_URLS=https://qdrant.tech/documentation/overview/,https://docs.mistral.ai/getting-started/models/
OLLAMA_MODEL=mistral:7b-instruct-q4_K_M
CHUNK_WORDS=300
CHUNK_OVERLAP=0.2
```

### Build the index once:

```bash
python -m pipeline.build_index
# Expect logs:
# [INGEST] Created <N> chunks
# [INGEST] Upsert complete.
```

### Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

### Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Qdrant and how does it store vectors?"}'
```

> Response includes **answer** and **sources** (snippets + source URLs).

## Configuration

- **Chunking:** set `CHUNK_WORDS` (e.g., `300`) and `CHUNK_OVERLAP` (e.g., `0.2`).
- **Sources:** comma-separated `DOC_URLS` in `.env`.
- **Collection:** `QDRANT_COLLECTION` name (per-experiment if desired).
- **Qdrant URL inside containers (Airflow):** use `http://qdrant:6333` (same network) or `http://host.docker.internal:6333`.


## Add Airflow for One-Click Re-ingest

### Build Airflow Image with Deps

Use the official base image and install pinned deps **as the `airflow` user**.

**Dockerfile (example):**
```dockerfile
FROM apache/airflow:2.9.2-python3.12
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

**requirements.txt (pins known to work)**

```text
langchain
langchain-qdrant
qdrant-client
langchain-ollama
langchain-experimental
sentence-transformers
beautifulsoup4
requests
python-dotenv
```

### Compose mounts & build & env
In your `docker-compose.yaml` (Airflow), ensure these mounts per service (**webserver/scheduler/worker**):

```yaml
build:
      context: D:\projects\poc\airflow
      dockerfile: Dockerfile
volumes:
  - "D:/projects/poc/rag-poc/dags:/opt/airflow/dags"
  - "D:/projects/poc/rag-poc:/opt/airflow/repo"
  - "D:/projects/poc/rag-poc/.env:/opt/airflow/.env:ro"
  - "D:/projects/poc/airflow/logs:/opt/airflow/logs"
```

**Environment Variables**

```yaml
# If qdrant container is on the same Docker network:
QDRANT_URL: "http://qdrant:6333"

# Or via host bridge (Docker Desktop / Windows / macOS):
QDRANT_URL: "http://host.docker.internal:6333"

# other env vars:
QDRANT_URL: "http://host.docker.internal:6333"
QDRANT_COLLECTION: "rag_poc"
DOC_URLS: "https://qdrant.tech/documentation/overview/,https://docs.mistral.ai/getting-started/models/"
CHUNK_WORDS: "300"
CHUNK_OVERLAP: "0.2"
```

### Start Airflow
```bash
docker compose up -d --build
```

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.

**MIT License**



