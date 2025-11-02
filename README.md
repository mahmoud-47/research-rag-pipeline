# Research Paper RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline developed while working as a Research Assistant to improve literature review efficiency across large volumes of research papers.  
It ingests academic documents, generates embeddings, stores them in a vector database, and retrieves context for LLM-based summarization.

---

## 🚀 Features

- Multi-format document ingestion (PDF, DOCX, XLSX, JSON, TXT, CSV)
- Semantic chunking with overlapping windows
- Embedding generation via SentenceTransformers
- Persistent similarity search using FAISS
- RAG workflow with LLM summarization (ChatGroq)
- Parallel ingestion for scalability
- Automatic index rebuild when outdated/missing

---

## 🧠 Workflow

1. Load documents
2. Chunk content into semantically coherent windows
3. Embed chunks and persist metadata
4. Store vectors inside FAISS (disk-backed)
5. Retrieve top-K relevant chunks at query time
6. Summarize results via LLM

---

## 🛠️ Tools & Technologies

**Vector Databases**
- FAISS (similarity search, index persistence)

**LLMs & Orchestration**
- Groq / ChatGroq API
- Prompt engineering for contextual summarization

**Embeddings**
- SentenceTransformers (`all-MiniLM-L6-v2`)

**Document Processing**
- LangChain community loaders (PDF, text, Excel, JSON, etc.)
- RecursiveCharacterTextSplitter

**Index Lifecycle / MLOps-Adjacent**
- Persistent disk storage (`faiss.index`, metadata)
- Rebuild triggers based on data changes
- Embedding serialization

**Parallelism & Utilities**
- ThreadPoolExecutor
- dotenv, logging, numpy, pathlib

---

## 🔍 Usage

Build vectors:
```bash
python app.py
