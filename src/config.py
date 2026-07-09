from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CODE_TRAVAIL_URL = (
    "https://raw.githubusercontent.com/SocialGouv/legi-data/master/"
    "data/LEGITEXT000006072050.json"
)
CODE_TRAVAIL_ID = "LEGITEXT000006072050"
PRIMARY_SOURCE = "Légifrance"
TECHNICAL_SOURCE = "SocialGouv/legi-data"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

RAW_CODE_TRAVAIL_FILE = RAW_DATA_DIR / "code_du_travail.json"
METADATA_CODE_TRAVAIL_FILE = METADATA_DIR / "code_du_travail_metadata.json"
PROCESSED_CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks_code_du_travail.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
VECTOR_INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
VECTOR_CHUNKS_METADATA_FILE = VECTOR_STORE_DIR / "chunks_metadata.json"
VECTOR_STORE_METADATA_FILE = VECTOR_STORE_DIR / "vector_store_metadata.json"

REQUEST_TIMEOUT = 30

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_REQUEST_TIMEOUT = 30
