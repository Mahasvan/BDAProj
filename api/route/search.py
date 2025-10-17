from fastapi import APIRouter, HTTPException

from api.service import shell
from api.service.llama_retriever import LlamaRetriever

router = APIRouter()
prefix = "/search"


@router.get("/wiki")
def wiki_search(query: str, k: int = 5):
    try:
        retriever = LlamaRetriever()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        resp = retriever.query(query, top_k=k)
        return {"result": str(resp)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def setup(app):
    app.include_router(router, prefix=prefix)
