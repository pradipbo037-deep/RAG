from core.chunker import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = " ".join([f"Sentence {index}." for index in range(1, 21)])
    chunks = chunk_text(text, chunk_size=80, overlap=20, source_name="demo.txt")

    assert len(chunks) >= 2
    assert all(chunk["text"].strip() for chunk in chunks)
    assert chunks[0]["source_name"] == "demo.txt"
