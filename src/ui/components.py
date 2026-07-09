from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.rag.orchestrator import RAGAnswer
from src.retrieval.contracts import RetrievedChunk


@dataclass(frozen=True)
class SourceView:
    article_id: str
    content: str
    score: str
    score_details: Mapping[str, str]
    theme: str = ""
    section: str = ""
    legi_id: str = ""
    source: str = ""


def answer_without_duplicate_warning(answer: RAGAnswer) -> str:
    """Keep the legal warning in its dedicated UI block only."""

    normalized_answer = answer.answer.strip()
    warning = answer.legal_warning.strip()
    if warning and normalized_answer.endswith(warning):
        normalized_answer = normalized_answer[: -len(warning)].rstrip()
    return normalized_answer


def build_source_views(chunks: Sequence[RetrievedChunk]) -> list[SourceView]:
    return [_build_source_view(chunk) for chunk in chunks]


def _build_source_view(chunk: RetrievedChunk) -> SourceView:
    return SourceView(
        article_id=chunk.article_id,
        content=chunk.content,
        score=format_score(chunk.score),
        score_details={
            name: format_score(value) for name, value in chunk.score_details.items()
        },
        theme=chunk.metadata.get("theme", ""),
        section=chunk.metadata.get("section", ""),
        legi_id=chunk.metadata.get("legi_id", ""),
        source=chunk.metadata.get("primary_source") or chunk.metadata.get("source", ""),
    )


def format_score(value: float) -> str:
    if not math.isfinite(value):
        return "indisponible"
    return f"{value:.4f}"
