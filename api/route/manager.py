import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.service import shell
from api.service.llama_index_updater import load_wikipedia_page_titles, build_index_from_titles

router = APIRouter()
prefix = "/manage"


@router.get("/reindex-wikipedia")
def reindex_wikipedia():
    try:
        shell.print_yellow_message("Starting Wikipedia reindex...")
        titles = load_wikipedia_page_titles("data/wikipedia_pages.txt")
        build_index_from_titles(titles, index_path="data/index")
    except Exception as e:
        shell.print_red_message(f"Reindex failed: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

    shell.print_green_message("Reindex complete.")
    return JSONResponse(content={"success": True})


def setup(app):
    app.include_router(router, prefix=prefix)
