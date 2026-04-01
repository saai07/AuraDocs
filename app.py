import streamlit as st
import os
import time
from core import (
    extract_text_from_file,
    chunk_text,
    Embedder,
    Retriever,
    generate_answer_stream,
    TOP_K,
)

st.set_page_config(page_title="AuraDocs", layout="wide")

# --- Set API key as env var for google-genai Client ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("Google API key not found. Set it in `.streamlit/secrets.toml` or as an env variable.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# --- Initialize session state ---
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()
    st.session_state.retriever = Retriever()
    st.session_state.processed_files = set()
    st.session_state.messages = []
    st.session_state.metrics = {
        "files_processed": 0,
        "total_chunks": 0,
        "questions_asked": 0,
        "avg_response_time": 0,
        "response_times": [],
        "source_counts": {},
    }

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📄 AuraDocs")
    st.caption("AI-powered document Q&A with RAG")
    st.divider()

    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, CSV, Excel",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        text = extract_text_from_file(file)
                        if not text.strip():
                            st.warning(f"No text extracted from {file.name}.")
                            continue
                        chunks = chunk_text(text)
                        if not chunks:
                            st.warning(f"No chunks generated from {file.name}.")
                            continue
                        embeddings = st.session_state.embedder.embed(chunks)
                        st.session_state.retriever.add_chunks(chunks, embeddings, file.name)
                        st.session_state.processed_files.add(file.name)

                        # Update metrics
                        st.session_state.metrics["files_processed"] += 1
                        st.session_state.metrics["total_chunks"] = len(st.session_state.retriever.chunks)

                        st.success(f"✅ {file.name} — {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Show processed files
    if st.session_state.processed_files:
        st.divider()
        st.markdown(f"**📊 {len(st.session_state.retriever.chunks)} chunks indexed**")
        for fname in st.session_state.processed_files:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "TXT": "📝", "CSV": "📊", "XLSX": "📗", "XLS": "📗"}.get(ext, "📄")
            st.caption(f"{icon} {fname}")

    # Clear button
    st.divider()
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.embedder = Embedder()
        st.session_state.retriever = Retriever()
        st.session_state.processed_files = set()
        st.session_state.messages = []
        st.session_state.metrics = {
            "files_processed": 0, "total_chunks": 0,
            "questions_asked": 0, "avg_response_time": 0,
            "response_times": [], "source_counts": {},
        }
        st.rerun()

# --- Main Chat Area ---
if not st.session_state.processed_files:
    # Welcome screen
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📂 Upload")
        st.markdown("Drop your PDFs, text files, CSVs, or Excel sheets in the sidebar.")
    with col2:
        st.markdown("### 🧠 Process")
        st.markdown("Documents are chunked, embedded, and indexed automatically.")
    with col3:
        st.markdown("### 💬 Ask")
        st.markdown("Ask questions and get AI answers grounded in your documents.")
    st.markdown("---")
    st.info("👈 Start by uploading documents in the sidebar!")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, (chunk, src) in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}** from `{src}`")
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                        if i < len(msg["sources"]) - 1:
                            st.divider()

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate streaming answer
        with st.chat_message("assistant"):
            start_time = time.time()

            query_emb = st.session_state.embedder.embed([question])
            results = st.session_state.retriever.search(query_emb, k=TOP_K)

            if not results:
                answer = "No relevant chunks found in the uploaded documents."
                st.markdown(answer)
                sources = []
            else:
                # Stream the response
                stream = generate_answer_stream(
                    question,
                    results,
                    chat_history=st.session_state.messages,
                )
                answer = st.write_stream(stream)
                sources = results

                # Show sources
                with st.expander("📚 Sources"):
                    for i, (chunk, src) in enumerate(results):
                        st.markdown(f"**Chunk {i+1}** from `{src}`")
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                        if i < len(results) - 1:
                            st.divider()

            # Update metrics
            elapsed = time.time() - start_time
            m = st.session_state.metrics
            m["questions_asked"] += 1
            m["response_times"].append(elapsed)
            m["avg_response_time"] = sum(m["response_times"]) / len(m["response_times"])
            for _, src in results:
                m["source_counts"][src] = m["source_counts"].get(src, 0) + 1

        # Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })