from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.retrieval.contracts import RetrievedChunk


@dataclass(frozen=True)
class ResultAggregator:
    """Deduplicate multi-query results and keep the strongest chunks."""

    def aggregate(
        self,
        result_sets: Iterable[Sequence[RetrievedChunk]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")

        best_by_article: dict[str, RetrievedChunk] = {}
        for result_set in result_sets:
            for chunk in result_set:
                key = self._deduplication_key(chunk)
                current = best_by_article.get(key)
                if current is None or chunk.score > current.score:
                    best_by_article[key] = chunk

        ranked = sorted(
            best_by_article.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _deduplication_key(chunk: RetrievedChunk) -> str:
        article_id = chunk.metadata.get("article_id", "").strip()
        if article_id:
            return f"article:{article_id.casefold()}"

        legi_id = chunk.metadata.get("legi_id", "").strip()
        if legi_id:
            return f"legi:{legi_id.casefold()}"

        return f"content:{chunk.content.strip().casefold()}"
