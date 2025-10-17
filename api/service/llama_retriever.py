from pathlib import Path
from typing import List

from llama_index import StorageContext, load_index_from_storage

INDEX_DIR = Path("data/index")


class LlamaRetriever:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LlamaRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self, index_path: str = None):
        path = index_path or str(INDEX_DIR)
        if not Path(path).exists():
            raise FileNotFoundError(f"Index path not found: {path}")
        storage_context = StorageContext.from_defaults(persist_dir=path)
        self.index = load_index_from_storage(storage_context)

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        response = self.index.as_query_engine(k=top_k).query(query_text)
        return response


if __name__ == "__main__":
    r = LlamaRetriever()
    print(r.query("What is the UN?"))
