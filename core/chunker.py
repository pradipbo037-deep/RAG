from __future__ import annotations

import re
from typing import Any, Dict, List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100, source_name: str = "", doc_metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Split text into useful chunks with a configurable overlap."""
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[Dict[str, Any]] = []
    working_text = ""
    chunk_id = 1

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not working_text:
                    working_text = sentence
                elif len(working_text) + 1 + len(sentence) <= chunk_size:
                    working_text = f"{working_text} {sentence}"
                else:
                    chunks.append(_build_chunk(working_text, source_name, chunk_id, doc_metadata))
                    chunk_id += 1
                    working_text = sentence
            continue

        if not working_text:
            working_text = paragraph
        elif len(working_text) + 1 + len(paragraph) <= chunk_size:
            working_text = f"{working_text} {paragraph}"
        else:
            chunks.append(_build_chunk(working_text, source_name, chunk_id, doc_metadata))
            chunk_id += 1
            working_text = paragraph

    if working_text:
        chunks.append(_build_chunk(working_text, source_name, chunk_id, doc_metadata))

    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap, source_name)

    return chunks


def _build_chunk(text: str, source_name: str, chunk_id: int, doc_metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    page_number = None
    if doc_metadata and "page_texts" in doc_metadata:
        page_number = 1
    return {
        "chunk_id": chunk_id,
        "source_name": source_name,
        "page_number": page_number,
        "text": text.strip(),
    }


def _apply_overlap(chunks: List[Dict[str, Any]], overlap: int, source_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            out.append(chunk)
            continue

        previous_chunk = out[-1]["text"]
        overlap_text = previous_chunk[-overlap:] if overlap < len(previous_chunk) else previous_chunk
        merged = f"{overlap_text} {chunk['text']}".strip()

        if index == 0:
            out.append(chunk)
        else:
            chunk["text"] = merged
            out.append(chunk)
    return out
