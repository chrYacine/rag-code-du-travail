from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from src.config import HYBRID_ALPHA, HYBRID_BETA
    from src.retrieval.bm25_retriever import (
        BM25RetrievalEngine,
        extract_article_ids,
        normalize_article_id,
    )
    from src.retrieval.contracts import RetrievedChunk, RetrievalEngine
    from src.retrieval.vector_retriever import VectorRetrievalEngine
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import HYBRID_ALPHA, HYBRID_BETA  # type: ignore
    from src.retrieval.bm25_retriever import (  # type: ignore
        BM25RetrievalEngine,
        extract_article_ids,
        normalize_article_id,
    )
    from src.retrieval.contracts import RetrievedChunk, RetrievalEngine  # type: ignore
    from src.retrieval.vector_retriever import VectorRetrievalEngine  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HybridCandidate:
    content: str
    metadata: dict[str, str]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    article_id_match: bool = False


def deduplication_key(chunk: RetrievedChunk) -> str:
    article_id = chunk.metadata.get("article_id", "").strip()
    if article_id:
        return f"article:{article_id.casefold()}"

    legi_id = chunk.metadata.get("legi_id", "").strip()
    if legi_id:
        return f"legi:{legi_id.casefold()}"

    return f"content:{chunk.content.strip().casefold()}"


def normalize_scores(results: Sequence[RetrievedChunk]) -> dict[str, float]:
    if not results:
        return {}

    scores_by_key = {deduplication_key(chunk): chunk.score for chunk in results}
    scores = list(scores_by_key.values())
    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        normalized_value = 1.0 if maximum > 0 else 0.0
        return {key: normalized_value for key in scores_by_key}

    return {
        key: (score - minimum) / (maximum - minimum)
        for key, score in scores_by_key.items()
    }


def metadata_to_strings(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


class HybridRetrievalEngine:
    """Hybrid retrieval engine combining vector search and BM25."""

    def __init__(
        self,
        *,
        vector_engine: RetrievalEngine | None = None,
        bm25_engine: RetrievalEngine | None = None,
        alpha: float = HYBRID_ALPHA,
        beta: float = HYBRID_BETA,
        candidate_multiplier: int = 2,
    ) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha et beta doivent être positifs ou nuls.")
        if alpha == 0 and beta == 0:
            raise ValueError("alpha et beta ne peuvent pas être tous les deux nuls.")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier doit être supérieur ou égal à 1.")

        self.vector_engine = vector_engine or VectorRetrievalEngine()
        self.bm25_engine = bm25_engine or BM25RetrievalEngine()
        self.alpha = alpha
        self.beta = beta
        self.candidate_multiplier = candidate_multiplier

    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        return self.search_with_queries(
            original_query=question,
            vector_query=question,
            top_k=top_k,
        )

    def search_with_queries(
        self,
        *,
        original_query: str,
        vector_query: str,
        top_k: int,
    ) -> Sequence[RetrievedChunk]:
        if top_k <= 0:
            return []

        clean_original_query = original_query.strip()
        clean_vector_query = vector_query.strip()
        if not clean_original_query and not clean_vector_query:
            return []

        candidate_top_k = top_k * self.candidate_multiplier
        vector_results = (
            list(self.vector_engine.search(clean_vector_query, candidate_top_k))
            if clean_vector_query
            else []
        )
        bm25_results = (
            list(self.bm25_engine.search(clean_original_query, candidate_top_k))
            if clean_original_query
            else []
        )
        requested_article_ids = extract_article_ids(clean_original_query)

        vector_scores = normalize_scores(vector_results)
        bm25_scores = normalize_scores(bm25_results)
        candidates = self._merge_candidates(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            requested_article_ids=requested_article_ids,
        )

        ranked = sorted(
            candidates.values(),
            key=self._hybrid_score,
            reverse=True,
        )
        return [self._to_retrieved_chunk(candidate) for candidate in ranked][:top_k]

    def _merge_candidates(
        self,
        *,
        vector_results: Sequence[RetrievedChunk],
        bm25_results: Sequence[RetrievedChunk],
        vector_scores: dict[str, float],
        bm25_scores: dict[str, float],
        requested_article_ids: set[str],
    ) -> dict[str, HybridCandidate]:
        candidates: dict[str, HybridCandidate] = {}

        for chunk in vector_results:
            key = deduplication_key(chunk)
            candidates[key] = HybridCandidate(
                content=chunk.content,
                metadata=metadata_to_strings(chunk.metadata),
                vector_score=vector_scores.get(key, 0.0),
                article_id_match=self._matches_requested_article_id(
                    chunk, requested_article_ids
                ),
            )

        for chunk in bm25_results:
            key = deduplication_key(chunk)
            candidate = candidates.get(key)
            if candidate is None:
                candidates[key] = HybridCandidate(
                    content=chunk.content,
                    metadata=metadata_to_strings(chunk.metadata),
                    bm25_score=bm25_scores.get(key, 0.0),
                    article_id_match=self._matches_requested_article_id(
                        chunk, requested_article_ids
                    ),
                )
                continue

            candidate.bm25_score = bm25_scores.get(key, 0.0)
            candidate.article_id_match = (
                candidate.article_id_match
                or self._matches_requested_article_id(chunk, requested_article_ids)
            )
            if len(chunk.content) > len(candidate.content):
                candidate.content = chunk.content
                candidate.metadata = metadata_to_strings(chunk.metadata)

        return candidates

    def _hybrid_score(self, candidate: HybridCandidate) -> float:
        weighted_score = (
            self.alpha * candidate.vector_score + self.beta * candidate.bm25_score
        )
        if candidate.article_id_match:
            return max(weighted_score, 1.0)
        return weighted_score

    @staticmethod
    def _matches_requested_article_id(
        chunk: RetrievedChunk,
        requested_article_ids: set[str],
    ) -> bool:
        if not requested_article_ids:
            return False
        article_id = chunk.metadata.get("article_id", "")
        return normalize_article_id(article_id) in requested_article_ids

    def _to_retrieved_chunk(self, candidate: HybridCandidate) -> RetrievedChunk:
        hybrid_score = self._hybrid_score(candidate)
        return RetrievedChunk(
            content=candidate.content,
            score=hybrid_score,
            metadata=candidate.metadata,
            score_details={
                "vector_score": candidate.vector_score,
                "bm25_score": candidate.bm25_score,
                "hybrid_score": hybrid_score,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interroge les chunks du Code du travail avec retrieval hybride."
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
    engine = HybridRetrievalEngine()
    for result in engine.search(args.question, args.top_k):
        article_id = result.metadata.get("article_id", "article inconnu")
        logger.info("%s | score=%.4f", article_id, result.score)
