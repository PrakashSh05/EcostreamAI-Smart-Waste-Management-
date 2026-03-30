# rag/config.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 3  # tighter context window for higher precision retrieval
CHROMA_PERSIST_DIR = "chroma_store"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
FAST_LLM_MODEL = "llama-3.1-8b-instant"  # used for RAG generation (latency KPI)