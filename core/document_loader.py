from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def extract_document_text(file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a PDF or DOCX file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use a PDF or DOCX file.")


def _extract_pdf_text(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    import fitz

    doc = fitz.open(file_path)
    text_parts = []
    page_count = len(doc)
    page_texts = {}

    for page_number in range(page_count):
        page = doc.load_page(page_number)
        page_text = page.get_text("text")
        page_texts[page_number + 1] = page_text
        text_parts.append(f"\n--- Page {page_number + 1} ---\n{page_text}")

    full_text = "\n".join(text_parts).strip()
    metadata = {
        "file_name": file_path.name,
        "page_count": page_count,
        "page_texts": page_texts,
        "source_type": "pdf",
    }
    return full_text, metadata


def _extract_docx_text(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    from docx import Document

    doc = Document(file_path)
    paragraphs = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)
    metadata = {
        "file_name": file_path.name,
        "page_count": 1,
        "page_texts": {1: full_text},
        "source_type": "docx",
    }
    return full_text, metadata
