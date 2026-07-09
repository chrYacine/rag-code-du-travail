from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from src.config import (
        BM25_ARTICLE_ID_BOOST,
        BM25_B,
        BM25_K1,
        PROCESSED_CHUNKS_FILE,
    )
    from src.retrieval.contracts import RetrievedChunk
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import (  # type: ignore
        BM25_ARTICLE_ID_BOOST,
        BM25_B,
        BM25_K1,
        PROCESSED_CHUNKS_FILE,
    )
    from src.retrieval.contracts import RetrievedChunk  # type: ignore

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
ARTICLE_PATTERN = re.compile(r"\b([A-Za-z])\s*(\d+(?:-\d+)+)\b")


class BM25RetrievalError(RuntimeError):
    """Raised when the BM25 retriever cannot be loaded safely."""


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_accents.casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def extract_article_ids(text: str) -> set[str]:
    return {
        f"{letter.upper()}{numbers}"
        for letter, numbers in ARTICLE_PATTERN.findall(text)
    }


def normalize_article_id(article_id: str) -> str:
    match = ARTICLE_PATTERN.search(article_id)
    if not match:
        return article_id.strip().upper()
    letter, numbers = match.groups()
    return f"{letter.upper()}{numbers}"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BM25RetrievalError(f"Fichier de chunks introuvable: {path}")

    loaded = read_json_file(path)
    if not isinstance(loaded, list):
        raise BM25RetrievalError("Le fichier de chunks doit contenir une liste JSON.")

    for position, item in enumerate(loaded):
        validate_chunk(item, position)

    return loaded


def validate_chunk(item: Any, position: int) -> None:
    if not isinstance(item, dict):
        raise BM25RetrievalError(f"Chunk invalide à la position {position}.")

    if not isinstance(item.get("content"), str) or not item["content"].strip():
        raise BM25RetrievalError(f"Contenu manquant à la position {position}.")

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise BM25RetrievalError(f"Metadata manquante à la position {position}.")

    if not all(isinstance(key, str) for key in metadata):
        raise BM25RetrievalError(
            f"Clés metadata non textuelles à la position {position}."
        )

    if not all(isinstance(value, str) for value in metadata.values()):
        raise BM25RetrievalError(
            f"Valeurs metadata non textuelles à la position {position}."
        )


class BM25Scorer:
    """Small BM25 Okapi scorer for the already prepared article chunks."""

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if not documents:
            raise BM25RetrievalError("Aucun document à indexer avec BM25.")

        self.k1 = k1
        self.b = b
        self.document_frequencies = [Counter(document) for document in documents]
        self.document_lengths = [len(document) for document in documents]
        self.average_document_length = sum(self.document_lengths) / len(
            self.document_lengths
        )
        self.inverse_document_frequencies = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.document_frequencies)
        term_document_counts: Counter[str] = Counter()
        for document_frequency in self.document_frequencies:
            term_document_counts.update(document_frequency.keys())

        return {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in term_document_counts.items()
        }

    def score(self, query_tokens: Sequence[str]) -> list[float]:
        if not query_tokens:
            return [0.0 for _ in self.document_frequencies]

        unique_query_tokens = set(query_tokens)
        scores: list[float] = []
        for document_frequency, document_length in zip(
            self.document_frequencies, self.document_lengths
        ):
            score = 0.0
            for token in unique_query_tokens:
                token_frequency = document_frequency.get(token, 0)
                if token_frequency == 0:
                    continue

                idf = self.inverse_document_frequencies.get(token, 0.0)
                denominator = token_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                score += idf * token_frequency * (self.k1 + 1) / denominator

            scores.append(score)

        return scores


def build_retrieved_chunk(
    *,
    item: dict[str, Any],
    bm25_score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=item["content"],
        score=bm25_score,
        metadata=item["metadata"],
        score_details={"bm25_score": bm25_score},
    )


class BM25RetrievalEngine:
    """BM25 retrieval engine over processed article chunks."""

    def __init__(
        self,
        *,
        chunks_path: Path = PROCESSED_CHUNKS_FILE,
        k1: float = BM25_K1,
        b: float = BM25_B,
        article_id_boost: float = BM25_ARTICLE_ID_BOOST,
    ) -> None:
        self.chunks = load_chunks(chunks_path)
        self.article_id_boost = article_id_boost
        tokenized_documents = [tokenize(chunk["content"]) for chunk in self.chunks]
        self.scorer = BM25Scorer(tokenized_documents, k1=k1, b=b)

    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        clean_question = question.strip()
        if not clean_question or top_k <= 0:
            return []

        query_tokens = tokenize(clean_question)
        requested_article_ids = extract_article_ids(clean_question)
        scores = self.scorer.score(query_tokens)

        ranked: list[tuple[float, dict[str, Any]]] = []
        for score, chunk in zip(scores, self.chunks):
            final_score = score + self._article_id_bonus(chunk, requested_article_ids)
            if final_score <= 0:
                continue
            ranked.append((final_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            build_retrieved_chunk(item=chunk, bm25_score=score)
            for score, chunk in ranked[:top_k]
        ]

    def _article_id_bonus(
        self,
        chunk: dict[str, Any],
        requested_article_ids: set[str],
    ) -> float:
        if not requested_article_ids:
            return 0.0

        article_id = chunk["metadata"].get("article_id", "")
        if normalize_article_id(article_id) in requested_article_ids:
            return self.article_id_boost

        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interroge les chunks du Code du travail avec BM25."
    )
    parser.add_argument("question", help="Question à rechercher dans le corpus.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre maximal de chunks à retourner.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    engine = BM25RetrievalEngine()
    for result in engine.search(args.question, args.top_k):
        article_id = result.metadata.get("article_id", "article inconnu")
        logger.info("%s | score=%.4f", article_id, result.score)
