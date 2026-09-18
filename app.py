"""
app.py
------
Streamlit UI for the PDF RAG Chatbot.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from rag_pipeline import load_and_split_pdf, build_vector_store, build_qa_chain, ask_question

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("A Retrieval-Augmented Generation (RAG) chatbot — upload a PDF and ask questions about it.")

# --- Sidebar: API key + upload ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com/keys",
    )
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_btn = st.button("Process PDF", type="primary")

# --- Session state ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Process PDF ---
if process_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    elif not api_key:
        st.sidebar.error("Please enter your Groq API key.")
    else:
        with st.spinner("Reading and indexing PDF... (first run downloads the embedding model)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            vector_store = build_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store, groq_api_key=api_key)
            st.session_state.messages = []
            os.unlink(tmp_path)
        st.sidebar.success("PDF processed! Ask a question below.")

# --- Chat interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask something about your PDF..."):
    if not st.session_state.qa_chain:
        st.error("Please upload and process a PDF first (sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state.qa_chain, question)
                st.write(result["answer"])
                with st.expander("Sources used"):
                    for i, src in enumerate(result["sources"], 1):
                        st.caption(f"**Chunk {i}:** {src}...")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
