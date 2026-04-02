import streamlit as st
import os
from core import (
    extract_text_from_file,
    chunk_text,
    Embedder,
    Retriever,
    generate_answer_stream,
    ask_about_image_stream,
    TOP_K,
    SUPPORTED_IMAGE_EXTENSIONS,
)

st.set_page_config(page_title="AuraDocs", page_icon="📄", layout="wide")

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
    st.session_state.doc_messages = []
    st.session_state.image_messages = []
    st.session_state.image_uploader_key = 0

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📄 AuraDocs")
    st.caption("Multimodal AI Assistant for Documents & Images")
    st.divider()

    # --- Mode Selection ---
    mode = st.radio("🚀 Action", ["Document Q&A", "Image Chat"], horizontal=True)
    st.divider()

    if mode == "Document Q&A":
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

    else:
        st.header("🖼️ Upload Image")
        uploaded_image = st.file_uploader(
            "PNG, JPG",
            type=[ext.strip(".") for ext in SUPPORTED_IMAGE_EXTENSIONS],
            key=f"image_uploader_{st.session_state.image_uploader_key}",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Current Image", use_container_width=True)
            if st.button("🗑️ Clear Image", use_container_width=True):
                st.session_state.image_uploader_key += 1
                st.rerun()

    # Clear all data
    st.divider()
    if st.button("Clear History & Data", use_container_width=True):
        if mode == "Document Q&A":
            st.session_state.doc_messages = []
            st.session_state.processed_files = set()
            st.session_state.retriever = Retriever()
        else:
            st.session_state.image_messages = []
            st.session_state.image_uploader_key += 1
        st.rerun()

# --- Main Area ---
if mode == "Document Q&A" and not st.session_state.processed_files:
    # Welcome screen
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📂 Upload")
        st.markdown("Drop your documents in the sidebar.")
    with col2:
        st.markdown("### 🧠 RAG")
        st.markdown("Context-aware answers from your local vector index.")
    with col3:
        st.markdown("### 💬 Chat")
        st.markdown("Ask deep questions about your data.")
    st.markdown("---")
    st.info("👈 Please upload a document in the sidebar to start!")

elif mode == "Image Chat" and not uploaded_image:
    st.markdown("---")
    st.info("👈 Please upload an image in the sidebar to start chatting with it!")

else:
    # Get mode-scoped messages
    messages = st.session_state.doc_messages if mode == "Document Q&A" else st.session_state.image_messages

    # Display chat history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, (chunk, src) in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}** from `{src}`")
                        st.caption(chunk[:300] + "...")
                        if i < len(msg["sources"]) - 1: st.divider()

    # Chat input
    question = st.chat_input("Ask a question...")

    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            sources = []

            if mode == "Image Chat":
                if not uploaded_image:
                    st.error("Please upload an image first!")
                    st.stop()
                img_bytes = uploaded_image.getvalue()
                stream = ask_about_image_stream(question, img_bytes, filename=uploaded_image.name)
                answer = st.write_stream(stream)
            else:
                # Handle RAG
                query_emb = st.session_state.embedder.embed([question])
                results = st.session_state.retriever.search(query_emb, k=TOP_K)
                
                if not results:
                    answer = "No relevant information found in documents."
                    st.markdown(answer)
                else:
                    stream = generate_answer_stream(question, results, messages)
                    answer = st.write_stream(stream)
                    sources = results
                    with st.expander("📚 Sources"):
                        for i, (chunk, src) in enumerate(results):
                            st.markdown(f"**Chunk {i+1}** from `{src}`")
                            st.caption(chunk[:300] + "...")
                            if i < len(results) - 1: st.divider()

        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })