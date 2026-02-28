import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
# EMBED_MODEL = "all-minilm:l12-v2"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

RETRIEVER_K = 3
RETRIEVER_FETCH_K = 10
RETRIEVER_SCORE_THRESHOLD = 0.4

OLLAMA_BASE_URL = "http://localhost:11435"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
