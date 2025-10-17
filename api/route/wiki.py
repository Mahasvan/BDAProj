from fastapi import APIRouter, HTTPException

from api.service.llama_retriever import LlamaRetriever
from api.service import shell

router = APIRouter()
prefix = "/wiki"


def setup(app):
    try:
        retriever = LlamaRetriever()
        shell.print_green_message("Llama index retriever loaded.")
    except Exception as e:
        shell.print_red_message(f"Failed to load Llama index: {e}")
        raise

    @router.get("/search")
    def search(query: str, k: int = 5):
        try:
            resp = retriever.query(query, top_k=k)
            return {"results": resp}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(router, prefix=prefix)
