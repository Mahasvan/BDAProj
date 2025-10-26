import os

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.service import shell
from api.service.llama_retriever import LlamaRetriever
from api.service.ollama import OllamaGenerator

router = APIRouter()
prefix = "/chat"

ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
chat_model = os.environ.get("CHAT_MODEL", "llama3")


def check_connection():
    try:
        requests.get(ollama_base_url)
    except requests.exceptions.ConnectionError:
        return False
    return True


@router.get("/wiki", response_class=StreamingResponse)
async def wiki_chat(query: str):
    # load retriever and generator on-demand
    retriever = LlamaRetriever()
    generator = OllamaGenerator(base_url=ollama_base_url, model=chat_model)
    # query = query + "\n\n Do not ask if I need anything else answered."
    # get context from retriever
    response = retriever.query(query, top_k=5)
    # build a compact context string from top results
    parts = []
    for r in response:
        text = r.get("text")
        extra = r.get("extra_info") if r.get("extra_info") else ""
        parts.append(f"---\n{text}\n{extra}\n")
    context = "\n".join(parts)

    return StreamingResponse(generator.stream_response(context=context, question=query), media_type="text/plain")


def setup(app):
    if check_connection():
        shell.print_yellow_message(f"Connected to Ollama at {ollama_base_url}")
    else:
        shell.print_red_message(f"Failed to connect to Ollama at {ollama_base_url}")
        raise ConnectionError("Failed to connect to Ollama")

    app.include_router(router, prefix=prefix)
