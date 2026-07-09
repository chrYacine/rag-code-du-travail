from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

try:
    from src.config import (
        VECTOR_CHUNKS_METADATA_FILE,
        VECTOR_INDEX_FILE,
        VECTOR_STORE_METADATA_FILE,
    )
    from src.retrieval.contracts import RetrievedChunk
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import (  # type: ignore
        VECTOR_CHUNKS_METADATA_FILE,
        VECTOR_INDEX_FILE,
        VECTOR_STORE_METADATA_FILE,
    )
    from src.retrieval.contracts import RetrievedChunk  # type: ignore

logger = logging.getLogger(__name__)


class VectorRetrievalError(RuntimeError):
    """Raised when the FAISS vector retriever cannot be loaded safely."""


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vector_store_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise VectorRetrievalError(f"Metadata de base vectorielle introuvable: {path}")

    loaded = read_json_file(path)
    if not isinstance(loaded, dict):
        raise VectorRetrievalError(
            "Le fichier de metadata vectorielle doit contenir un objet JSON."
        )

    embedding_model = loaded.get("embedding_model")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise VectorRetrievalError(
            "Le modèle d'embedding utilisé par l'index est manquant."
        )

    return loaded


def load_chunks_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise VectorRetrievalError(f"Metadata des chunks introuvable: {path}")

    loaded = read_json_file(path)
    if not isinstance(loaded, list):
        raise VectorRetrievalError(
            "Le fichier de metadata des chunks doit contenir une liste JSON."
        )

    for position, item in enumerate(loaded):
        validate_chunk_metadata_item(item, position)

    return loaded


def validate_chunk_metadata_item(item: Any, position: int) -> None:
    if not isinstance(item, dict):
        raise VectorRetrievalError(
            f"Metadata de chunk invalide à la position {position}."
        )

    if not isinstance(item.get("vector_id"), int):
        raise VectorRetrievalError(
            f"vector_id manquant ou invalide à la position {position}."
        )

    if not isinstance(item.get("content"), str) or not item["content"].strip():
        raise VectorRetrievalError(
            f"Contenu du chunk manquant à la position {position}."
        )

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise VectorRetrievalError(
            f"Metadata juridique manquante à la position {position}."
        )

    if not all(isinstance(key, str) for key in metadata):
        raise VectorRetrievalError(
            f"Clés metadata non textuelles à la position {position}."
        )

    if not all(isinstance(value, str) for value in metadata.values()):
        raise VectorRetrievalError(
            f"Valeurs metadata non textuelles à la position {position}."
        )


def load_faiss_index(path: Path) -> Any:
    if not path.exists():
        raise VectorRetrievalError(f"Index FAISS introuvable: {path}")

    import faiss

    return faiss.read_index(str(path))


def load_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_query(model: Any, question: str) -> Any:
    embeddings = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def validate_index_alignment(
    index: Any, chunks_metadata: Sequence[dict[str, Any]]
) -> None:
    index_size = getattr(index, "ntotal", None)
    if index_size is not None and int(index_size) != len(chunks_metadata):
        raise VectorRetrievalError(
            "L'index FAISS et les metadata de chunks ne contiennent pas "
            "le même nombre d'éléments."
        )


def build_retrieved_chunk(
    *,
    item: dict[str, Any],
    vector_score: float,
) -> RetrievedChunk:
    metadata = item["metadata"]
    return RetrievedChunk(
        content=item["content"],
        score=vector_score,
        metadata=metadata,
        score_details={"vector_score": vector_score},
    )


class VectorRetrievalEngine:
    """Vector-only retrieval engine backed by FAISS and parallel metadata."""

    def __init__(
        self,
        *,
        index_path: Path = VECTOR_INDEX_FILE,
        chunks_metadata_path: Path = VECTOR_CHUNKS_METADATA_FILE,
        vector_metadata_path: Path = VECTOR_STORE_METADATA_FILE,
    ) -> None:
        self.vector_store_metadata = load_vector_store_metadata(vector_metadata_path)
        self.embedding_model_name = self.vector_store_metadata["embedding_model"]
        self.chunks_metadata = load_chunks_metadata(chunks_metadata_path)
        self.index = load_faiss_index(index_path)
        validate_index_alignment(self.index, self.chunks_metadata)
        self.model = load_embedding_model(self.embedding_model_name)

    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        clean_question = question.strip()
        if not clean_question or top_k <= 0:
            return []

        query_embedding = encode_query(self.model, clean_question)
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[RetrievedChunk] = []
        for score, index_position in zip(scores[0], indices[0]):
            vector_id = int(index_position)
            if vector_id < 0:
                continue
            if vector_id >= len(self.chunks_metadata):
                raise VectorRetrievalError(
                    f"FAISS a retourné un vector_id inconnu: {vector_id}."
                )

            results.append(
                build_retrieved_chunk(
                    item=self.chunks_metadata[vector_id],
                    vector_score=float(score),
                )
            )

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interroge la base vectorielle FAISS du Code du travail."
    )
    parser.add_argument("question", help="Question à rechercher dans le corpus.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre maximal de chunks à retourner.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    warnings.filterwarnings(
        "ignore", message=".*You are sending unauthenticated requests.*"
    )
    for logger_name in (
        "faiss",
        "huggingface_hub",
        "httpx",
        "sentence_transformers",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    engine = VectorRetrievalEngine()
    for result in engine.search(args.question, args.top_k):
        article_id = result.metadata.get("article_id", "article inconnu")
        logger.info("%s | score=%.4f", article_id, result.score)
