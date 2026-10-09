# Local RAG Workshop App

A lightweight, beginner-friendly local Retrieval-Augmented Generation (RAG) application built for a classroom workshop. It allows you to upload PDF or DOCX files, extract text, split it into chunks, generate local embeddings, retrieve relevant context using cosine similarity, and ask a locally running Ollama model a grounded question.

## What this app demonstrates

- PDF and DOCX extraction
- Text cleaning and chunking
- Local embedding generation
- Cosine similarity scoring
- Top-k retrieval
- Context construction for LLM prompts
- Ollama model discovery and fallback logic
- Transparent, grounded responses

## Architecture

```text
PDF/DOCX
  -> Document extraction
  -> Text cleaning
  -> Chunking
  -> Embedding generation
  -> Cosine similarity retrieval
  -> Prompt with retrieved context
  -> Ollama model
  -> Grounded answer
```

## Tech stack

- Python 3.11
- Streamlit
- PyMuPDF
- python-docx
- sentence-transformers
- scikit-learn
- NumPy
- Ollama

## Quick start

1. Install Python 3.11.
2. Create a virtual environment.
3. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

4. Install and start Ollama.
5. Pull or install a local model, such as Qwen or Llama.
6. Run the app:

```bash
streamlit run app.py
```

## Ollama setup

Download and install Ollama from https://ollama.com.

Then check models locally:

```bash
ollama list
```

Common model choices:

```bash
ollama pull qwen2.5
# or
ollama pull llama3.2
```

The app will detect the best available model automatically, preferring Qwen and falling back to Llama if needed.

## Model selection policy

The application follows this order:

1. Prefer Qwen if a suitable local Qwen model is installed.
2. Otherwise use Llama if a compatible local Llama model is installed.
3. Otherwise show a clear setup message in the UI.

## Demo workflow

1. Upload a syllabus, contract, or course notes PDF/DOCX.
2. Process the document.
3. Ask a grounded question such as:
   - What are the eligibility requirements?
   - Which deadlines are mentioned?
   - What are the key deliverables?
4. Inspect retrieval results and similarity scores.
5. Review the answer produced from the retrieved context.

## Workshop teaching points

- Embeddings represent semantic meaning rather than exact keyword matching.
- Cosine similarity measures how closely aligned two vectors are.
- Retrieval finds the most relevant document chunks.
- The LLM receives the question plus the retrieved context.
- Good grounding reduces hallucinations.
- The application must never invent unsupported information.

## Troubleshooting

### Ollama not detected

- Make sure Ollama is installed and the service is running.
- Check `http://localhost:11434` is reachable.

### No Qwen or Llama model found

Run one of the following:

```bash
ollama pull qwen2.5
```

or

```bash
ollama pull llama3.2
```

### Document not extracting

- Ensure the file is a valid PDF or DOCX.
- Try a clean, non-encrypted document.
- Check that the file is not empty.

## Project structure

```text
rag_workshop/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── core/
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── ollama_client.py
│   ├── model_selector.py
│   └── rag_pipeline.py
├── ui/
│   ├── components.py
│   └── styles.py
├── data/
│   └── uploads/
└── tests/
    ├── test_chunking.py
    ├── test_embeddings.py
    ├── test_retrieval.py
    └── test_model_selection.py
```

## License

This project is intended for educational demo and workshop use.
