from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.config import (
        EMBEDDING_MODEL,
        PROCESSED_CHUNKS_FILE,
        VECTOR_CHUNKS_METADATA_FILE,
        VECTOR_INDEX_FILE,
        VECTOR_STORE_METADATA_FILE,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (  # type: ignore
        EMBEDDING_MODEL,
        PROCESSED_CHUNKS_FILE,
        VECTOR_CHUNKS_METADATA_FILE,
        VECTOR_INDEX_FILE,
        VECTOR_STORE_METADATA_FILE,
    )

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be built safely."""


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_vector_store_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    loaded = read_json_file(path)
    if not isinstance(loaded, dict):
        return None
    return loaded


def load_chunks(path: Path) -> list[dict[str, Any]]:
    loaded = read_json_file(path)
    if not isinstance(loaded, list):
        raise VectorStoreError("Le fichier de chunks doit contenir une liste JSON.")
    return loaded


def validate_chunk(chunk: dict[str, Any], position: int) -> None:
    if not isinstance(chunk.get("id"), str) or not chunk["id"]:
        raise VectorStoreError(f"Chunk invalide à la position {position}: id manquant.")

    if not isinstance(chunk.get("content"), str) or not chunk["content"].strip():
        raise VectorStoreError(
            f"Chunk invalide à la position {position}: contenu manquant."
        )

    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        raise VectorStoreError(
            f"Chunk invalide à la position {position}: metadata manquante."
        )

    if not all(isinstance(key, str) for key in metadata):
        raise VectorStoreError(
            f"Chunk invalide à la position {position}: clés metadata non textuelles."
        )

    if not all(isinstance(value, str) for value in metadata.values()):
        raise VectorStoreError(
            f"Chunk invalide à la position {position}: valeurs metadata non textuelles."
        )


def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        raise VectorStoreError("Aucun chunk à vectoriser.")

    for position, chunk in enumerate(chunks):
        validate_chunk(chunk, position)


def build_parallel_metadata(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "vector_id": position,
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        }
        for position, chunk in enumerate(chunks)
    ]


def should_rebuild_vector_store(
    *,
    chunks_hash: str,
    index_path: Path,
    chunks_metadata_path: Path,
    vector_metadata_path: Path,
    force: bool = False,
) -> bool:
    if force:
        return True

    if not index_path.exists() or not chunks_metadata_path.exists():
        return True

    metadata = read_vector_store_metadata(vector_metadata_path)
    if not metadata:
        return True

    return metadata.get("chunks_hash_sha256") != chunks_hash


def load_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts(model: Any, texts: list[str]) -> Any:
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: Any) -> Any:
    import faiss

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def write_faiss_index(index: Any, index_path: Path) -> None:
    import faiss

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def build_vector_store_metadata(
    *,
    chunks_hash: str,
    chunks_count: int,
    embedding_dimension: int,
    embedding_model: str,
    chunks_file: Path,
    index_file: Path,
    chunks_metadata_file: Path,
    forced: bool,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunks_count": chunks_count,
        "chunks_hash_sha256": chunks_hash,
        "chunks_file": str(chunks_file),
        "index_file": str(index_file),
        "chunks_metadata_file": str(chunks_metadata_file),
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine_similarity_on_normalized_embeddings",
        "forced": forced,
    }


def build_status(
    *,
    status: str,
    rebuilt: bool,
    chunks_count: int,
    message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "rebuilt": rebuilt,
        "chunks_count": chunks_count,
        "message": message,
    }


def build_vector_store(
    *,
    force: bool = False,
    embedding_model: str = EMBEDDING_MODEL,
    chunks_path: Path = PROCESSED_CHUNKS_FILE,
    index_path: Path = VECTOR_INDEX_FILE,
    chunks_metadata_path: Path = VECTOR_CHUNKS_METADATA_FILE,
    vector_metadata_path: Path = VECTOR_STORE_METADATA_FILE,
) -> dict[str, Any]:
    logger.info("Vérification de la base vectorielle en cours...")

    chunks_hash = compute_file_sha256(chunks_path)
    chunks = load_chunks(chunks_path)
    validate_chunks(chunks)

    if not should_rebuild_vector_store(
        chunks_hash=chunks_hash,
        index_path=index_path,
        chunks_metadata_path=chunks_metadata_path,
        vector_metadata_path=vector_metadata_path,
        force=force,
    ):
        message = "Aucun changement détecté dans les chunks. La base vectorielle n'est pas reconstruite."
        logger.info(message)
        return build_status(
            status="unchanged",
            rebuilt=False,
            chunks_count=len(chunks),
            message=message,
        )

    logger.info(
        "Construction de la base vectorielle avec le modèle %s", embedding_model
    )
    texts = [chunk["content"] for chunk in chunks]
    model = load_embedding_model(embedding_model)
    embeddings = encode_texts(model, texts)
    index = build_faiss_index(embeddings)

    write_faiss_index(index, index_path)
    write_json_file(chunks_metadata_path, build_parallel_metadata(chunks))

    vector_metadata = build_vector_store_metadata(
        chunks_hash=chunks_hash,
        chunks_count=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
        embedding_model=embedding_model,
        chunks_file=chunks_path,
        index_file=index_path,
        chunks_metadata_file=chunks_metadata_path,
        forced=force,
    )
    write_json_file(vector_metadata_path, vector_metadata)

    message = "Base vectorielle construite et sauvegardée."
    logger.info(message)
    return build_status(
        status="updated" if not force else "forced",
        rebuilt=True,
        chunks_count=len(chunks),
        message=message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit la base vectorielle FAISS du Code du travail."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruit la base vectorielle même si les chunks n'ont pas changé.",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help="Modèle sentence-transformers à utiliser pour les embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    build_vector_store(force=args.force, embedding_model=args.model)
