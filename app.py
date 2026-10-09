import os
from pathlib import Path

import streamlit as st

from core.model_selector import pick_best_model, detect_ollama_models
from core.ollama_client import is_ollama_available
from core.rag_pipeline import RAGPipeline
from ui.styles import get_app_styles

DATA_DIR = Path(__file__).resolve().parent / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="Local RAG Workshop", page_icon="📚", layout="wide")
st.markdown(get_app_styles(), unsafe_allow_html=True)

# Keep a fixed local port for the workshop app to avoid stale browser sessions.
# This app is expected to run on http://localhost:802


if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


def set_status(message: str, kind: str = "info"):
    st.session_state.status_message = {"text": message, "kind": kind}


def get_ollama_status():
    available = is_ollama_available()
    models = detect_ollama_models() if available else []
    selected_model = pick_best_model(models)
    return available, models, selected_model


with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Local RAG Workshop")
        st.caption("Upload a PDF or DOCX, retrieve the most relevant sections, and ask a grounded question using a local LLM.")
    with col2:
        available, models, selected_model = get_ollama_status()
        status_text = "Connected" if available else "Not connected"
        status_color = "success" if available else "error"
        st.markdown(f"<div class='status-pill {status_color}'>{status_text}</div>", unsafe_allow_html=True)
        if selected_model:
            st.caption(f"Active model: {selected_model}")
        else:
            st.caption("No local Qwen or Llama model detected.")


st.markdown("---")

left_col, right_col = st.columns([1.6, 1.1])

with left_col:
    st.subheader("1. Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_path = DATA_DIR / uploaded_file.name
        file_path.write_bytes(uploaded_file.read())
        st.session_state.document_name = uploaded_file.name
        with st.spinner("Processing document..."):
            result = st.session_state.rag_pipeline.process_document(file_path, uploaded_file.name)
        st.success(f"Document indexed: {result['document_name']} | {result['chunk_count']} chunks created")
        st.write(f"Extracted characters: {result['char_count']} | Pages: {result['page_count']}")

        if result.get("warnings"):
            for warning in result["warnings"]:
                st.warning(warning)

    if st.session_state.document_name:
        st.info(f"Current document: {st.session_state.document_name}")

    st.subheader("2. Ask a question")
    question = st.text_input("Ask anything", placeholder="What is the capital of France? or ask about the uploaded document")

    if st.button("Search and answer"):
        if not is_ollama_available():
            st.error("Ollama is not running or not installed. Start Ollama and confirm that a local model is available.")
        else:
            selected_model = pick_best_model(detect_ollama_models())
            if not selected_model:
                st.warning("No Qwen or Llama model was found locally. Install one with `ollama pull qwen2.5` or `ollama pull llama3.2`.")
            else:
                with st.spinner("Generating an answer..."):
                    if st.session_state.document_name:
                        results = st.session_state.rag_pipeline.retrieve(question, top_k=4, min_similarity=0.15)
                        if not results:
                            st.info("No closely matching document chunks were found, so I’m answering from general knowledge instead.")
                            answer = st.session_state.rag_pipeline.generate_answer(
                                question=question,
                                context="",
                                model_name=selected_model,
                                temperature=0.2,
                            )
                            results = []
                        else:
                            min_score = min(item["score"] for item in results)
                            if min_score < 0.15:
                                st.info("The uploaded document does not closely match this question, so I’m using general knowledge for the answer.")
                                answer = st.session_state.rag_pipeline.generate_answer(
                                    question=question,
                                    context="",
                                    model_name=selected_model,
                                    temperature=0.2,
                                )
                            else:
                                context_parts = []
                                for item in results:
                                    source_text = f"[Source: {item['source_name']} | Page {item.get('page_number', 'N/A')} | Chunk {item['chunk_id']}]\n{item['text']}"
                                    context_parts.append(source_text)
                                context = "\n\n---\n\n".join(context_parts)

                                answer = st.session_state.rag_pipeline.generate_answer(
                                    question=question,
                                    context=context,
                                    model_name=selected_model,
                                    temperature=0.2,
                                )

                            with st.expander("View retrieved context and scores"):
                                for item in results:
                                    st.markdown(f"**Score:** {item['score']:.3f} | **Document:** {item['source_name']} | **Page:** {item.get('page_number', 'N/A')} | **Chunk:** {item['chunk_id']}")
                                    st.write(item['text'])
                                    st.markdown("---")
                    else:
                        st.info("No document uploaded, so I’m answering from general knowledge.")
                        answer = st.session_state.rag_pipeline.generate_answer(
                            question=question,
                            context="",
                            model_name=selected_model,
                            temperature=0.2,
                        )
                        results = []

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "results": results,
                })

                st.markdown("### Answer")
                st.write(answer)

    if st.session_state.chat_history:
        st.subheader("3. Conversation history")
        for i, turn in enumerate(reversed(st.session_state.chat_history[-5:]), start=1):
            with st.expander(f"Question {len(st.session_state.chat_history) - i + 1}: {turn['question']}"):
                st.write(turn["answer"])

with right_col:
    st.subheader("Knowledge Base status")
    stats = st.session_state.rag_pipeline.get_stats()
    st.metric("Document count", stats["document_count"])
    st.metric("Chunk count", stats["chunk_count"])
    st.metric("Embedding status", "Ready" if stats["embedding_ready"] else "Not ready")
    st.metric("Retrieval status", "Ready" if stats["retrieval_ready"] else "Not ready")

    st.subheader("Model status")
    available, models, selected_model = get_ollama_status()
    if available:
        st.success("Ollama connection looks healthy")
        if models:
            st.write("Available local models:")
            for model_name in models:
                st.code(model_name, language=None)
        else:
            st.warning("Ollama is running but no models were detected.")
    else:
        st.error("Ollama is not reachable on localhost:11434")

    st.subheader("Settings")
    st.session_state.rag_pipeline.chunk_size = st.slider("Chunk size", min_value=150, max_value=1200, value=st.session_state.rag_pipeline.chunk_size, step=50)
    st.session_state.rag_pipeline.overlap = st.slider("Chunk overlap", min_value=25, max_value=300, value=st.session_state.rag_pipeline.overlap, step=25)
    st.session_state.rag_pipeline.top_k = st.slider("Top-K", min_value=1, max_value=10, value=st.session_state.rag_pipeline.top_k, step=1)
    st.session_state.rag_pipeline.min_similarity = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=st.session_state.rag_pipeline.min_similarity, step=0.05)

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")
st.caption("Important: the LLM should only answer using retrieved context from the uploaded document. It must not invent unsupported facts.")
