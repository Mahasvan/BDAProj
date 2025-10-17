"""Ollama embeddings adapter for llama-index.

This implements the simple embed_documents / embed_query interface that
llama-index expects. It calls Ollama's HTTP embeddings endpoint and
returns vectors in the expected shape.

Note: Ollama's embeddings API surface may vary by version. This adapter
tries to parse common OpenAI-like responses ({'data':[{'embedding':[...]},...]})
and a fallback key 'embeddings'. Adjust parsing to match your Ollama server.
"""
import os
from typing import List

import requests


class OllamaEmbeddings:
    def __init__(self, base_url: str | None = None, model: str = "all-minilm"):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model

    def _call_embeddings(self, inputs: List[str]):
        url = f"{self.base_url}/api/embeddings"
        # Ollama expects a 'prompt' field for embedding requests (example: all-minilm)
        # Try a bulk request first (prompt as list). If the server doesn't accept
        # bulk prompts, fall back to one request per input.
        def _parse_response(data):
            # Common OpenAI-like shape
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                embeddings = []
                for item in data["data"]:
                    emb = item.get("embedding") or item.get("embeddings")
                    embeddings.append(emb)
                return embeddings

            # Single vector under 'embedding'
            if isinstance(data, dict) and "embedding" in data:
                return [data["embedding"]]

            # Fallback: direct list under 'embeddings'
            if isinstance(data, dict) and "embeddings" in data:
                return data["embeddings"]

            # If response itself is a list of vectors
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data

            return None

        # First attempt: send inputs as a bulk 'prompt'
        try:
            bulk_payload = {"model": self.model, "prompt": inputs}
            resp = requests.post(url, json=bulk_payload)
            if resp.status_code == 200:
                data = resp.json()
                parsed = _parse_response(data)
                if parsed is not None:
                    return parsed
            # if we reach here, fall back to per-item
        except Exception:
            # proceed to per-item fallback
            pass

        # Per-item fallback: call embeddings endpoint once per string
        embeddings = []
        for text in inputs:
            payload = {"model": self.model, "prompt": text}
            resp = requests.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama embeddings call failed: {resp.status_code} {resp.text}")
            data = resp.json()
            parsed = _parse_response(data)
            if parsed is None:
                raise ValueError(f"Unexpected embeddings response: {data}")
            # parsed will be a single-item list for single-prompt responses
            embeddings.append(parsed[0])

        return embeddings

    def embed_documents(self, texts: List[str]):
        """Embed a list of documents and return list of vectors."""
        return self._call_embeddings(texts)

    def embed_query(self, text: str):
        """Embed a single query string and return a single vector."""
        res = self._call_embeddings([text])
        return res[0]
