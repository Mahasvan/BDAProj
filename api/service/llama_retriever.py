from pathlib import Path
from typing import List, Dict

from llama_index import StorageContext, load_index_from_storage

from api.service.config import get_index_path


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
        query_engine = self.index.as_query_engine(k=top_k, similarity_top_k=top_k)
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
