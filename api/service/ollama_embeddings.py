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
        payload = {"model": self.model, "input": inputs}
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embeddings call failed: {resp.status_code} {resp.text}")
        data = resp.json()

        # Common OpenAI-like shape
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            embeddings = []
            for item in data["data"]:
                emb = item.get("embedding") or item.get("embeddings")
                embeddings.append(emb)
            return embeddings

        # Fallback: direct list under 'embeddings'
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]

        # If response itself is a list of vectors
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data

        raise ValueError(f"Unexpected embeddings response: {data}")

    def embed_documents(self, texts: List[str]):
        """Embed a list of documents and return list of vectors."""
        return self._call_embeddings(texts)

    def embed_query(self, text: str):
        """Embed a single query string and return a single vector."""
        res = self._call_embeddings([text])
        return res[0]
