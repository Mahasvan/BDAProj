import json
import requests
from typing import Generator

from . import shell


class OllamaGenerator:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        shell.print_green_message(f"Ollama generator configured for model {model} at {self.base_url}")

    def _build_prompt(self, context: str, question: str) -> str:
        prompt = (
            "Context:\n" + context + "\n\n" +
            "Question:\n" + question + "\n\n" +
            "Answer concisely and cite relevant context extracts."
        )
        return prompt

    def stream_response(self, context: str, question: str) -> Generator[bytes, None, None]:
        prompt = self._build_prompt(context, question)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": True
        }

        with requests.post(url, json=payload, stream=True) as r:
            if r.status_code != 200:
                shell.print_red_message(f"Ollama call failed: {r.status_code} {r.text}")
                yield json.dumps({"error": r.text}).encode()
                return

            for chunk in r.iter_lines(decode_unicode=False):
                if not chunk:
                    continue
                try:
                    # Ollama streams newline-delimited JSON events; forward raw bytes
                    yield chunk + b"\n"
                except Exception as e:
                    shell.print_red_message(f"Error streaming chunk: {e}")
                    continue

