from __future__ import annotations

import pytest

from src.retrieval.contracts import RetrievedChunk
from src.retrieval.result_aggregator import ResultAggregator


def chunk(
    article_id: str,
    score: float,
    *,
    content: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=content or f"Article {article_id}",
        score=score,
        metadata={"article_id": article_id} if article_id else {},
        score_details={"hybrid_score": score},
    )


def test_aggregate_deduplicates_articles_and_keeps_best_score() -> None:
    aggregator = ResultAggregator()

    results = aggregator.aggregate(
        [
            [chunk("L3121-1", 0.60), chunk("L3141-1", 0.75)],
            [chunk("L3121-1", 0.92), chunk("L1237-11", 0.70)],
        ],
        top_k=5,
    )

    assert [result.article_id for result in results] == [
        "L3121-1",
        "L3141-1",
        "L1237-11",
    ]
    assert results[0].score == 0.92


def test_aggregate_sorts_and_applies_global_top_k() -> None:
    results = ResultAggregator().aggregate(
        [[chunk("L1-1", 0.20), chunk("L2-1", 0.90), chunk("L3-1", 0.70)]],
        top_k=2,
    )

    assert [result.article_id for result in results] == ["L2-1", "L3-1"]


def test_chunks_without_article_id_use_legi_id_then_content() -> None:
    with_legi_id = RetrievedChunk(
        content="Premier texte",
        score=0.4,
        metadata={"legi_id": "LEGI-1"},
    )
    better_same_legi_id = RetrievedChunk(
        content="Premier texte enrichi",
        score=0.8,
        metadata={"legi_id": "LEGI-1"},
    )
    content_only = RetrievedChunk(content="Autre texte", score=0.6)

    results = ResultAggregator().aggregate(
        [[with_legi_id, better_same_legi_id, content_only]],
        top_k=5,
    )

    assert results == [better_same_legi_id, content_only]


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="supérieur ou égal à 1"):
        ResultAggregator().aggregate([], top_k=0)
