from __future__ import annotations

from typing import Any, Dict, List

import requests


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 180


def is_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [item.get("name", "") for item in models if item.get("name")]
    except requests.RequestException:
        return []


def generate_answer(question: str, context: str, model_name: str, temperature: float = 0.2, base_url: str = OLLAMA_BASE_URL) -> str:
    clean_context = (context or "").strip()

    if clean_context:
        prompt = (
            "You are a helpful assistant. Use the provided document context when it is relevant, and use general knowledge when the retrieved context is missing or unrelated to the user's question. "
            "Be transparent about the source of the answer, and keep it concise and useful. Include source names or page numbers when available.\n\n"
            f"Question: {question}\n\nRetrieved context:\n{clean_context}"
        )
    else:
        prompt = (
            "You are a helpful general-purpose assistant. Answer the user's question using general knowledge when no relevant document context is available. "
            "Do not refuse unrelated questions just because there is no uploaded PDF or document context.\n\n"
            f"Question: {question}"
        )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=OLLAMA_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No answer generated.")
    except requests.Timeout:
        return (
            f"Ollama timed out while generating the answer with '{model_name}'. "
            "The model may still be loading for the first time. Please wait a minute and retry, "
            "or run 'ollama pull {model_name}' to ensure it is downloaded and ready."
        )
    except requests.RequestException as exc:
        return (
            f"Ollama request failed for '{model_name}': {exc}. "
            "Please verify that Ollama is running and that the model is installed locally."
        )
