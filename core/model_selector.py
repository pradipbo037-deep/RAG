from __future__ import annotations

from typing import List, Optional

from core.ollama_client import list_ollama_models


def detect_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    return list_ollama_models(base_url=base_url)


def pick_best_model(models: List[str]) -> Optional[str]:
    if not models:
        return None

    lowered = [name.lower() for name in models]
    qwen_candidates = [models[i] for i, name in enumerate(lowered) if "qwen" in name]
    if qwen_candidates:
        return qwen_candidates[0]

    llama_candidates = [models[i] for i, name in enumerate(lowered) if "llama" in name]
    if llama_candidates:
        return llama_candidates[0]

    return None
