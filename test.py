"""Run this script to ingest Wikipedia pages listed in `data/wikipedia_pages.txt`
and build a persisted Llama-Index in `data/index`.
"""
from pathlib import Path
from api.service.llama_retriever import LlamaRetriever
from api.service.config import get_embedding_model, get_ollama_base_url
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding



ollama_embedding = OllamaEmbedding(
    model_name=get_embedding_model(),
    base_url=get_ollama_base_url(),
    # Can optionally pass additional kwargs to ollama
    # ollama_additional_kwargs={"mirostat": 0},
)

Settings.embed_model = ollama_embedding


def main():
    # titles = load_wikipedia_page_titles("data/wikipedia_pages.txt")
    retriever = LlamaRetriever(index_path="data/index")
    res = retriever.query("tell me about climate change", top_k=3)
    print("\n")
    print(len(res))
    for r in res:
        print(r["text"])
        break
        

if __name__ == "__main__":
    main()
