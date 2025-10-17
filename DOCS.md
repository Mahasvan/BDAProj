Project changes and how to run
=================================

What I changed
- Replaced the ECOSOC / ChromaDB pipeline with a Wikipedia-backed RAG prototype using Llama-Index for the vector store and Ollama for generation.
- Added an ingestion script to download a list of Wikipedia pages (from `data/wikipedia_pages.txt`) and build a persisted Llama-Index at `data/index`.
- Created a retriever wrapper (`api/service/llama_retriever.py`) that loads the persisted index and exposes a simple `query()` method.
- Replaced the chat and search routes with wiki-focused endpoints:
  - `GET /search/wiki?query=...` — returns retriever output
  - `GET /chat/wiki?query=...` — runs retrieval and streams an Ollama-generated answer
- Added a management route: `GET /manage/reindex-wikipedia` to rebuild the index from `data/wikipedia_pages.txt` (blocking proof-of-concept).

Quickstart (local)
------------------
1. Create a Python 3.10+ virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `.env` if you need to change defaults (there is a sample `.env` in the repo). The defaults assume Ollama listens at `http://localhost:11434` and index is persisted to `data/index`.

4. Build the index (downloads pages listed in `data/wikipedia_pages.txt`):

```bash
python scripts/ingest_wikipedia.py
```

5. Start the API:

```bash
python app.py
```

6. Open `http://localhost:5000/docs` to explore the API. Useful endpoints:
- `GET /manage/reindex-wikipedia` — rebuild index from `data/wikipedia_pages.txt`.
- `GET /search/wiki?query=...` — run semantic retrieval against the index.
- `GET /chat/wiki?query=...` — run retrieval + stream Ollama answer.

Notes and caveats
- This is a minimal proof-of-concept. The ingestion downloads Wikipedia pages using the `wikipedia` package; some titles may fail to resolve and will be skipped.
- The Ollama HTTP API path used is `/api/generate` and streaming is proxied as raw newline-delimited JSON events. If your Ollama setup differs, update `api/service/ollama.py`.
- I removed the Chroma/ECOSOC-specific files to simplify the codebase — if you need them back, revert the git history.

If you want, I can now run a small static check (import-only) or add unit tests for the ingestion and retrieval pipeline.
