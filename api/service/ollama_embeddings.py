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

# Import BaseEmbedding from llama-index (path compatible with 0.14.x)
try:
    from llama_index.embeddings import BaseEmbedding
except Exception:
    # fallback import path for older versions
    from llama_index.core.embeddings import BaseEmbedding


class OllamaEmbeddings(BaseEmbedding):
    """Ollama embeddings adapter implementing llama-index BaseEmbedding API.

    This adapter calls an Ollama embeddings HTTP endpoint and returns vectors
    in the shape expected by llama-index.
    """

    # declare as fields so pydantic/BaseModel based BaseEmbedding accepts them
    base_url: str | None = None
    model: str = "all-minilm"

    def __init__(self, base_url: str | None = None, model: str = "all-minilm"):
        # use object.__setattr__ to bypass pydantic's __setattr__ restrictions
        url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        object.__setattr__(self, "base_url", url.rstrip("/"))
        object.__setattr__(self, "model", model)

    def _call_embeddings(self, inputs: List[str]):
        url = f"{self.base_url}/api/embeddings"

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
        except Exception:
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
            embeddings.append(parsed[0])

        return embeddings

    def embed_documents(self, texts: List[str]):
        """Embed a list of documents and return list of vectors."""
        return self._call_embeddings(texts)

    def embed_query(self, text: str):
        """Embed a single query string and return a single vector."""
        res = self._call_embeddings([text])
        return res[0]

    # --- Methods required by llama-index BaseEmbedding ---
    def _get_query_embedding(self, query: str):
        """Synchronous single-query embedding used by llama-index."""
        return self.embed_query(query)

    async def _aget_query_embedding(self, query: str):
        """Asynchronous single-query embedding. Runs sync call in a thread."""
        import asyncio

        res = await asyncio.to_thread(self.embed_query, query)
        return res

    def _get_text_embedding(self, text: str):
        """Synchronous document text embedding used by llama-index."""
        return self.embed_query(text)
