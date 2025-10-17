"""Ingest a list of Wikipedia pages and build a Llama-Index vector store.
This module uses `llama_index` for indexing and assumes Ollama will be used
as the LLM for generation. Embeddings will be created using the OpenAI
compatible interface provided by llama_index; we'll wire Ollama to the
indexer's response later.
"""
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext
from llama_index.core.node_parser.text.token import TokenTextSplitter
from api.service.ollama_embeddings import OllamaEmbeddings

from api.service.config import get_index_path

DATA_DIR = Path("data")
WIKI_DIR = DATA_DIR / "wikipedia_pages"
INDEX_DIR = Path(get_index_path())


def load_wikipedia_page_titles(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]
    return titles


def build_index_from_titles(titles: List[str], index_path: str = None):
    """Download Wikipedia pages as text and build a Llama-Index GPTVectorStoreIndex.

    This function intentionally keeps model/embedding wiring minimal; the
    exact embedding model will be configured when connecting Ollama.
    """
    import wikipedia
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for title in titles:
        filename = WIKI_DIR / f"{title.replace(' ', '_')}.txt"
        if not filename.exists():
            print(f"Downloading: {title}")
            try:
                page = wikipedia.page(title)
                text = page.content
            except Exception as e:
                print(f"Failed to download {title}: {e}")
                continue
            # persist using UTF-8 to avoid encoding errors on Windows
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

    # create an index from the directory of text files
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # SimpleDirectoryReader will read the text files
    reader = SimpleDirectoryReader(str(WIKI_DIR))
    documents = reader.load_data()

    # Basic text splitter
    splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunked_docs = []
    for d in documents:
        # print(d.text)
        for chunk in splitter.split_text(d.text):
            new_doc = d.copy()
            new_doc.set_content(chunk)
            # print(new_doc.text == chunk)
            chunked_docs.append(new_doc)

    # Build a simple index; use Ollama embeddings (all-minilm) via ServiceContext
    embeddings = OllamaEmbeddings(model="all-minilm")
    service_context = ServiceContext.from_defaults(embed_model=embeddings)
    index = GPTVectorStoreIndex.from_documents(chunked_docs, service_context=service_context)

    target_path = index_path or str(INDEX_DIR)
    index.storage_context.persist(path=target_path)
    print(f"Index saved to {target_path}")

    return index


if __name__ == "__main__":
    titles = load_wikipedia_page_titles("data/wikipedia_pages.txt")
    build_index_from_titles(titles, index_path=str(INDEX_DIR))
