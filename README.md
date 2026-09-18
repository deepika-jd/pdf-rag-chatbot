<<<<<<< HEAD
# pdf-rag-chatbot
"Built a RAG-based chatbot that answers questions from PDF documents using local embeddings (sentence-transformers), FAISS vector search, and Groq's LLM API — reducing hallucination by grounding all answers in retrieved document context."
=======
# 📄 Chat with your PDF — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask
questions about its content in natural language. Built to understand and implement
core GenAI engineering concepts: embeddings, vector search, and grounded LLM responses.

## How it works

1. **Ingest** — the PDF is loaded and split into overlapping text chunks
2. **Embed** — each chunk is converted into a vector using a local, free embedding
   model (`sentence-transformers/all-MiniLM-L6-v2`) — no API cost
3. **Store & Retrieve** — chunks are indexed in a **FAISS** vector store; when you ask
   a question, the most relevant chunks are retrieved by similarity search
4. **Generate** — the retrieved chunks + your question are sent to an LLM
   (**Groq's Llama 3.1**, free tier) which generates an answer grounded in the document

```
PDF → Chunking → Embeddings → FAISS Vector Store
                                      ↓
User Question → Embed → Similarity Search → Top-K Chunks
                                      ↓
                        LLM (Groq) + Context → Answer
```

## Tech stack

- **LangChain** — orchestration
- **FAISS** — vector similarity search
- **sentence-transformers** — local embeddings (no API cost)
- **Groq API** — fast LLM inference (free tier)
- **Streamlit** — web UI

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd pdf-rag-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# then edit .env and paste your key (free at https://console.groq.com/keys)

# 5. Run the app
streamlit run app.py
```

## Usage

1. Open the app in your browser (Streamlit opens it automatically)
2. Enter your Groq API key in the sidebar (or set it in `.env`)
3. Upload a PDF and click **Process PDF**
4. Ask questions in the chat box — answers are grounded in the document, with
   source chunks shown for transparency

## Project structure

```
pdf-rag-chatbot/
├── app.py             # Streamlit UI
├── rag_pipeline.py     # Core RAG logic (chunking, embeddings, retrieval, QA chain)
├── requirements.txt
├── .env.example
└── README.md
```

## Possible extensions

- Swap FAISS for a hosted vector DB (Pinecone, Qdrant, Weaviate)
- Add conversation memory for multi-turn follow-up questions
- Support multiple PDFs / a document library
- Add citation highlighting (show exact page number per answer)
- Deploy on Streamlit Community Cloud or Hugging Face Spaces

## License

MIT
>>>>>>> e3abf32 (pdf-rag-chatbot)
