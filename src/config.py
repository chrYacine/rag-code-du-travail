from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

REQUEST_TIMEOUT = 30
