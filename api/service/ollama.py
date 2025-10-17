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

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    # each line is typically a JSON object
                    obj = None
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # not JSON, forward raw line
                        yield line.encode() + b"\n"
                        continue

                    # try common fields
                    text_piece = None
                    if isinstance(obj, dict):
                        # Ollama may use 'content' or nested 'choices' -> 'delta' or 'text'
                        if "content" in obj:
                            text_piece = obj.get("content")
                        elif "text" in obj:
                            text_piece = obj.get("text")
                        elif "choices" in obj:
                            # choices can be a list of {"delta": {"content": "..."}}
                            try:
                                choices = obj.get("choices")
                                if isinstance(choices, list) and len(choices) > 0:
                                    c = choices[0]
                                    if isinstance(c, dict):
                                        if "delta" in c and isinstance(c["delta"], dict):
                                            text_piece = c["delta"].get("content")
                                        else:
                                            text_piece = c.get("text") or c.get("content")
                            except Exception:
                                text_piece = None

                    if text_piece is not None:
                        yield text_piece.encode() + b"\n"
                    else:
                        # fallback: return the JSON line verbatim
                        yield json.dumps(obj).encode() + b"\n"

                except Exception as e:
                    shell.print_red_message(f"Error parsing Ollama stream line: {e}")
                    continue

