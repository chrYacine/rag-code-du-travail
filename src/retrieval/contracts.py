from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk returned by the retrieval engine with its legal metadata."""

    content: str
    score: float
    metadata: Mapping[str, str] = field(default_factory=dict)
    score_details: Mapping[str, float] = field(default_factory=dict)

    @property
    def article_id(self) -> str:
        return self.metadata.get("article_id", "unknown article")


class RetrievalEngine(Protocol):
    """Contract implemented by the retrieval component owned by indexing/retrieval."""

    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        """Return the most relevant chunks for a user question.

        The final `score` may come from vector retrieval, BM25 retrieval, or a
        hybrid formula such as alpha * vector_score + beta * bm25_score.
        Optional score components can be exposed through `score_details`.
        """
