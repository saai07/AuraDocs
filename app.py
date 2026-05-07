import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from core import (
    extract_text_from_file,
    chunk_text,
    Embedder,
    Retriever,
    generate_answer_stream,
    ask_about_image_stream,
    TOP_K,
    SUPPORTED_IMAGE_EXTENSIONS,
    CONTENT_REGISTRY_URL,
    fetch_classes,
    fetch_subjects,
    fetch_chapters,
    fetch_content_items,
    build_registry_context,
    upload_and_extract_stream,
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

for key, default in [
    ("doc_messages", []),
    ("image_messages", []),
    ("processed_files", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📄 AuraDocs")
    st.caption("Multimodal AI Assistant for Documents & Images")
    st.divider()

    # --- Mode Selection ---
    mode = st.radio("🚀 Action", ["Document Q&A", "Image Chat"], horizontal=True)
    st.divider()

    # ── Document Q&A Sidebar ─────────────────────────────────────────
    if mode == "Document Q&A":

        # ── Section 1: Local File Upload ────────────────────────────
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
            st.markdown(f"**📊 {len(st.session_state.retriever.chunks)} chunks indexed**")
            for fname in st.session_state.processed_files:
                ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "REG"
                icon = {"PDF": "📕", "TXT": "📝", "CSV": "📊", "XLSX": "📗", "XLS": "📗"}.get(ext, "📄")
                st.caption(f"{icon} {fname}")

        st.divider()

        # ── Section 2: Content Registry ──────────────────────────────
        with st.expander("📚 Content Registry", expanded=False):
            st.caption(f"`{CONTENT_REGISTRY_URL}`")

            # ──────────────────────────────────────────────────────────
            # Upload, Extract & Chat — unified SSE flow
            # ──────────────────────────────────────────────────────────
            st.subheader("🚀 Upload, Extract & Chat")

            reg_file = st.file_uploader(
                "Image or PDF",
                type=["png", "jpg", "jpeg", "pdf"],
                key="reg_uploader",
            )
            reg_class_name = st.text_input("Class", placeholder="e.g. Class 10", key="reg_up_class")
            reg_subject    = st.text_input("Subject", placeholder="e.g. Science", key="reg_up_subject")
            reg_chapter    = st.text_input("Chapter", placeholder="e.g. Light", key="reg_up_chapter")
            reg_tags       = st.text_input("Tags (optional)", placeholder="e.g. optics, refraction", key="reg_up_tags")

            if st.button("🚀 Upload, Extract & Chat", use_container_width=True, type="primary", key="reg_go_btn"):
                if not reg_file:
                    st.warning("Please select a file to upload.")
                elif not reg_class_name or not reg_subject or not reg_chapter:
                    st.warning("Please fill in Class, Subject, and Chapter.")
                else:
                    progress = st.progress(0, text="📤 Uploading & starting extraction...")
                    status_text = st.empty()

                    # Stream SSE events from the backend
                    final_event = None
                    for event in upload_and_extract_stream(
                        file_bytes=reg_file.read(),
                        filename=reg_file.name,
                        class_name=reg_class_name,
                        subject=reg_subject,
                        chapter=reg_chapter,
                        tags=f'["{reg_tags}"]' if reg_tags else "[]",
                    ):
                        pct = event.get("percent", 0)
                        msg = event.get("message", "Processing...")
                        status = event.get("status", "processing")

                        if status == "failed":
                            progress.progress(100, text=f"❌ {msg}")
                            st.error(msg)
                            final_event = event
                            break

                        # Map backend 0-100% to our 5-85% (leave room for indexing step)
                        display_pct = min(5 + int(pct * 0.80), 85) if pct < 100 else 88
                        progress.progress(display_pct, text=f"🔍 {msg}")
                        final_event = event

                    # Process the final result
                    if final_event and final_event.get("status") == "completed":
                        extracted_text = final_event.get("extracted_text", "")
                        content_id = final_event.get("content_id", "")

                        if not extracted_text or not extracted_text.strip():
                            progress.progress(100, text="⚠️ No text extracted.")
                            st.warning("Extraction completed but no text was found.")
                        else:
                            progress.progress(92, text="📦 Chunking & indexing into RAG...")

                            source = f"{reg_class_name} › {reg_subject} › {reg_chapter}"
                            chunks = chunk_text(extracted_text)
                            if chunks:
                                embeddings = st.session_state.embedder.embed(chunks)
                                st.session_state.retriever.add_chunks(chunks, embeddings, source)
                                st.session_state.processed_files.add(source)

                            progress.progress(100, text="✅ Ready to chat!")
                            st.success(f"✅ **{len(chunks)}** chunks indexed from **{source}** — start chatting!")

                            # Clear caches so Load Existing sees the new content
                            fetch_classes.clear()
                            fetch_subjects.clear()
                            fetch_chapters.clear()
                            fetch_content_items.clear()

                    elif final_event and final_event.get("status") == "failed":
                        pass  # Already handled above
                    else:
                        progress.progress(100, text="⚠️ No response from server.")
                        st.warning("Extraction did not complete. Try again.")


    # ── Image Chat Sidebar ───────────────────────────────────────────
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


# ── Main Area ────────────────────────────────────────────────────────

if mode == "Document Q&A" and not st.session_state.processed_files:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📂 Upload")
        st.markdown("Drop your documents in the sidebar.")
    with col2:
        st.markdown("### 🧠 RAG")
        st.markdown("Context-aware answers from your documents.")
    with col3:
        st.markdown("### 💬 Chat")
        st.markdown("Ask deep questions about your data.")
    st.markdown("---")
    st.info("👈 Upload a document **or** use the Content Registry in the sidebar to start!")

elif mode == "Image Chat" and not uploaded_image:
    st.markdown("---")
    st.info("👈 Please upload an image in the sidebar to start chatting with it!")

else:
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
                        if i < len(msg["sources"]) - 1:
                            st.divider()

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
                # Document Q&A RAG
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
                            if i < len(results) - 1:
                                st.divider()

        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
