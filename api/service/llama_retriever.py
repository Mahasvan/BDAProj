from pathlib import Path
from typing import List, Dict

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from api.service.config import get_index_path, get_embedding_model, get_ollama_base_url, get_chat_model
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model=get_chat_model(),
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=2048,
)


ollama_embedding = OllamaEmbedding(
    model_name=get_embedding_model(),
    base_url=get_ollama_base_url(),
    # Can optionally pass additional kwargs to ollama
    # ollama_additional_kwargs={"mirostat": 0},
)

Settings.embed_model = ollama_embedding
Settings.llm = llm

class LlamaRetriever:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LlamaRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self, index_path: str = None):
        path = index_path or get_index_path()
        if not Path(path).exists():
            raise FileNotFoundError(f"Index path not found: {path}")
        storage_context = StorageContext.from_defaults(persist_dir=path)
        self.index = load_index_from_storage(storage_context)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return a list of dicts: { 'text': ..., 'score': ..., 'source': {...} }"""
        query_engine = self.index.as_query_engine(k=top_k, similarity_top_k=top_k, llm=llm)
        response = query_engine.query(query_text)

        # Extract nodes/sources if available
        results = []
        try:
            for node in response.source_nodes:
                results.append({
                    "text": node.node.get_text(),
                    "extra_info": getattr(node.node, "extra_info", None),
                    "source_info": getattr(node, "source_info", None)
                })
        except Exception:
            # Fallback: return the whole response as text
            results.append({"text": str(response), "extra_info": None, "source_info": None})

        return results


if __name__ == "__main__":
    r = LlamaRetriever()
    print(r.query("What is the UN?", top_k=3))
