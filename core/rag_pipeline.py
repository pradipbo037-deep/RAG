from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.chunker import chunk_text
from core.document_loader import extract_document_text
from core.embeddings import embed_query, embed_texts, load_embedding_model
from core.ollama_client import generate_answer as generate_ollama_answer
from core.retriever import retrieve_top_k
from core.text_cleaner import clean_text


class RAGPipeline:
    def __init__(self, chunk_size: int = 500, overlap: int = 100, top_k: int = 4, min_similarity: float = 0.15, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.documents: Dict[str, Dict[str, Any]] = {}

    def ensure_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = load_embedding_model(self.embedding_model_name)
        return self.embedding_model

    def process_document(self, file_path: str | Path, file_name: Optional[str] = None) -> Dict[str, Any]:
        source_path = Path(file_path)
        document_name = file_name or source_path.name

        raw_text, metadata = extract_document_text(source_path)
        cleaned_text = clean_text(raw_text)

        chunks = chunk_text(
            cleaned_text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            source_name=document_name,
            doc_metadata=metadata,
        )

        if not chunks:
            return {
                "document_name": document_name,
                "chunk_count": 0,
                "char_count": len(cleaned_text),
                "page_count": metadata.get("page_count", 1),
                "warnings": ["The document was empty or could not be interpreted."],
            }

        model = self.ensure_embedding_model()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts, model)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        self.documents[document_name] = {
            "path": str(source_path),
            "metadata": metadata,
            "chunks": chunks,
            "embeddings": embeddings,
        }

        return {
            "document_name": document_name,
            "chunk_count": len(chunks),
            "char_count": len(cleaned_text),
            "page_count": metadata.get("page_count", 1),
            "warnings": [],
        }

    def retrieve(self, question: str, top_k: Optional[int] = None, min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        all_chunks = []
        for document in self.documents.values():
            all_chunks.extend(document["chunks"])

        if not all_chunks:
            return []

        query_embedding = embed_query(question, self.ensure_embedding_model())
        results = retrieve_top_k(
            chunks=all_chunks,
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            min_similarity=min_similarity if min_similarity is not None else self.min_similarity,
            allow_fallback=True,
        )
        return results

    def generate_answer(self, question: str, context: str, model_name: str, temperature: float = 0.2) -> str:
        return generate_ollama_answer(question=question, context=context, model_name=model_name, temperature=temperature)

    def get_stats(self) -> Dict[str, Any]:
        documents = len(self.documents)
        chunks = sum(len(doc["chunks"]) for doc in self.documents.values())
        return {
            "document_count": documents,
            "chunk_count": chunks,
            "embedding_ready": self.embedding_model is not None,
            "retrieval_ready": documents > 0,
        }
