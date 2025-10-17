# MunHelper

Context-based Wikipedia lookup. Searches selected Wikipedia pages and answers queries using a Llama-Index vector store and Ollama for generation.

## Features

- OpenAPI-compatible API using FastAPI
- Semantic search over selected Wikipedia pages
- Ingestion script to build a Llama-Index vector store
- Integration with Ollama for generation

## Quickstart (local)

Follow these steps to get the project running locally.

- Create and activate a Python virtual environment (macOS / zsh example):

```bash
python -m venv .venv
source .venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- (Optional) Edit `.env` to configure `OLLAMA_BASE_URL`, `CHAT_MODEL`, and `INDEX_PATH`.

- Ingest the Wikipedia pages and build the index:

```bash
python scripts/ingest_wikipedia.py
```

- Start the API:

```bash
python app.py
```

Open the API docs at: `http://localhost:5000/docs`

## Usage

1. Populate the index by running the ingestion script (step 4 above). This creates `data/wikipedia_pages/` and persists a Llama-Index under `data/index/` (or the path set in `INDEX_PATH`).

2. Available API endpoints:

- `GET /search/wiki?query=...` — run a semantic search and return retriever results.
- `GET /chat/wiki?query=...` — run retrieval and stream an Ollama-generated answer.
- `GET /manage/reindex-wikipedia` — rebuild the index from `data/wikipedia_pages.txt` (blocking proof-of-concept).

To change which pages are ingested, edit `data/wikipedia_pages.txt` then call `/manage/reindex-wikipedia`.

## Installation options

### Bare metal

- Clone the repository:

```bash
git clone https://github.com/Mahasvan/Munhelper
cd Munhelper
```

- Follow the Quickstart steps above (venv, install, ingest, run).

If you use Ollama locally, you can pull a model with:

```bash
ollama pull llama3.2:1b
```

### Docker

If you prefer Docker, use the repository's Docker setup (if present) or build images using docker-compose. Docker configuration is not modified by this migration; consult existing Docker files in the repo for details.

## Notes and caveats

- Ingestion uses the `wikipedia` Python package; some titles may fail and will be skipped.
- The Ollama streaming parser in `api/service/ollama.py` tries to handle common newline-delimited JSON formats. If your Ollama version streams a different schema, update the parser accordingly.
- The `/manage/reindex-wikipedia` endpoint is a blocking proof-of-concept; for production use you should run ingestion asynchronously or via a job queue.

If you want me to: (a) clean up more docs, (b) add tests for ingestion and retrieval, or (c) wire embeddings to Ollama (if your Ollama exposes embeddings), tell me which and I'll implement next.
