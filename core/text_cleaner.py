from __future__ import annotations

import re


def clean_text(raw_text: str) -> str:
    """Clean extracted document text while preserving meaning."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    return text.strip()
