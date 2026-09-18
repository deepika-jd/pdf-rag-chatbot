"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) logic:
1. Load a PDF and split it into chunks
2. Embed chunks locally using sentence-transformers
3. Store/search embeddings using a FAISS vector index
4. Retrieve relevant chunks + send them to an LLM (Groq) to answer questions
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Loads a PDF and splits it into overlapping text chunks.
    Overlap helps preserve context across chunk boundaries.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    print(f"Loaded '{pdf_path}' -> {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks, save_path: str = "faiss_index"):
    """
    Embeds chunks locally (free, no API calls) and builds a FAISS index.
    The index is saved to disk so we don't have to re-embed every run.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"Vector store saved to '{save_path}/'")
    return vector_store


def load_vector_store(save_path: str = "faiss_index"):
    """Loads a previously saved FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


def build_qa_chain(vector_store, groq_api_key: str = None):
    """
    Builds a RetrievalQA chain:
    - Retriever pulls the top-k most relevant chunks from FAISS
    - Groq's LLM generates an answer grounded in those chunks
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in a .env file (see .env.example) "
            "or pass it directly to build_qa_chain()."
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",  # fast + free-tier friendly
        temperature=0.2,
    )

    prompt_template = """You are a helpful assistant answering questions based ONLY on the
provided document context. If the answer isn't in the context, say you don't know —
do not make up information.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def ask_question(qa_chain, question: str):
    """Runs a question through the QA chain and returns the answer + source chunks."""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]],
    }
