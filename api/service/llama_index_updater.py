"""Ingest a list of Wikipedia pages and build a Llama-Index vector store.
This module uses `llama_index` for indexing and assumes Ollama will be used
as the LLM for generation. Embeddings will be created using the OpenAI
compatible interface provided by llama_index; we'll wire Ollama to the
indexer's response later.
"""
from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import download_loader

from api.service.config import get_index_path, get_embedding_model, get_ollama_base_url


ollama_embedding = OllamaEmbedding(
    model_name=get_embedding_model(),
    base_url=get_ollama_base_url(),
    # Can optionally pass additional kwargs to ollama
    # ollama_additional_kwargs={"mirostat": 0},
)

Settings.embed_model = ollama_embedding

DATA_DIR = Path("data")
WIKI_DIR = DATA_DIR / "wikipedia_pages"
INDEX_DIR = Path(get_index_path())

def build_index_from_titles(titles: List[str], index_path: str = None):
    """Download Wikipedia pages as text and build a Llama-Index GPTVectorStoreIndex.

    This function intentionally keeps model/embedding wiring minimal; the
    exact embedding model will be configured when connecting Ollama.
    """
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=titles)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path or str(INDEX_DIR))

if __name__ == "__main__":
    build_index_from_titles(["Climate Change"], index_path=str(INDEX_DIR))
