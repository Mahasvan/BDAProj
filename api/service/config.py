import os


def get_env(key: str, default=None):
    return os.environ.get(key, default)


def get_index_path():
    return get_env("INDEX_PATH", "data/index")


def get_ollama_base_url():
    return get_env("OLLAMA_BASE_URL", "http://localhost:11434")


def get_chat_model():
    return get_env("CHAT_MODEL", "gemma3:1b")

def get_embedding_model():
    return get_env("EMBEDDING_MODEL", "all-minilm")