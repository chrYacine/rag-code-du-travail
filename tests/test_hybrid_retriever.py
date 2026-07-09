from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.retrieval.contracts import RetrievedChunk
from src.retrieval.hybrid_retriever import (
    HybridRetrievalEngine,
    deduplication_key,
    normalize_scores,
)


class FakeRetrievalEngine:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        self.calls.append((question, top_k))
        return self.results[:top_k]


def chunk(
    *,
    article_id: str,
    legi_id: str,
    score: float,
    score_details: dict[str, float],
    content: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=content or f"Article {article_id}\nTexte",
        score=score,
        metadata={
            "article_id": article_id,
            "legi_id": legi_id,
            "theme": "test",
        },
        score_details=score_details,
    )


def test_deduplication_key_prefers_article_id() -> None:
    retrieved = chunk(
        article_id="L3121-1",
        legi_id="LEGIARTI1",
        score=0.5,
        score_details={"vector_score": 0.5},
    )

    assert deduplication_key(retrieved) == "article:l3121-1"


def test_normalize_scores_scales_values_between_zero_and_one() -> None:
    results = [
        chunk(
            article_id="L1-1",
            legi_id="LEGIARTI1",
            score=2.0,
            score_details={"bm25_score": 2.0},
        ),
        chunk(
            article_id="L1-2",
            legi_id="LEGIARTI2",
            score=6.0,
            score_details={"bm25_score": 6.0},
        ),
    ]

    normalized = normalize_scores(results)

    assert normalized["article:l1-1"] == 0.0
    assert normalized["article:l1-2"] == 1.0


def test_normalize_scores_handles_single_positive_result() -> None:
    results = [
        chunk(
            article_id="L1-1",
            legi_id="LEGIARTI1",
            score=2.0,
            score_details={"bm25_score": 2.0},
        )
    ]

    assert normalize_scores(results) == {"article:l1-1": 1.0}


def test_hybrid_retrieval_engine_combines_vector_and_bm25_scores() -> None:
    vector_engine = FakeRetrievalEngine(
        [
            chunk(
                article_id="L3121-1",
                legi_id="LEGIARTI1",
                score=0.9,
                score_details={"vector_score": 0.9},
            ),
            chunk(
                article_id="L3141-1",
                legi_id="LEGIARTI2",
                score=0.6,
                score_details={"vector_score": 0.6},
            ),
        ]
    )
    bm25_engine = FakeRetrievalEngine(
        [
            chunk(
                article_id="L3121-1",
                legi_id="LEGIARTI1",
                score=4.0,
                score_details={"bm25_score": 4.0},
            ),
            chunk(
                article_id="L1237-11",
                legi_id="LEGIARTI3",
                score=2.0,
                score_details={"bm25_score": 2.0},
            ),
        ]
    )
    engine = HybridRetrievalEngine(
        vector_engine=vector_engine,
        bm25_engine=bm25_engine,
        alpha=0.7,
        beta=0.3,
    )

    results = engine.search("duree travail", top_k=2)

    assert [result.metadata["article_id"] for result in results] == [
        "L3121-1",
        "L3141-1",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score_details == {
        "vector_score": 1.0,
        "bm25_score": 1.0,
        "hybrid_score": 1.0,
    }
    assert results[1].score_details["hybrid_score"] == pytest.approx(0.0)


def test_hybrid_retrieval_engine_uses_separate_original_and_vector_queries() -> None:
    vector_engine = FakeRetrievalEngine(
        [
            chunk(
                article_id="L3121-1",
                legi_id="LEGIARTI1",
                score=0.8,
                score_details={"vector_score": 0.8},
            )
        ]
    )
    bm25_engine = FakeRetrievalEngine(
        [
            chunk(
                article_id="L1237-11",
                legi_id="LEGIARTI2",
                score=3.0,
                score_details={"bm25_score": 3.0},
            )
        ]
    )
    engine = HybridRetrievalEngine(
        vector_engine=vector_engine,
        bm25_engine=bm25_engine,
        alpha=0.5,
        beta=0.5,
        candidate_multiplier=3,
    )

    results = engine.search_with_queries(
        original_query="Que dit L1237-11 ?",
        vector_query="passage juridique hypothetique",
        top_k=2,
    )

    assert vector_engine.calls == [("passage juridique hypothetique", 6)]
    assert bm25_engine.calls == [("Que dit L1237-11 ?", 6)]
    assert results[0].metadata["article_id"] == "L1237-11"
    assert results[0].score_details["hybrid_score"] == pytest.approx(1.0)
    assert {result.metadata["article_id"] for result in results} == {
        "L3121-1",
        "L1237-11",
    }


def test_hybrid_retrieval_engine_returns_empty_list_for_empty_query() -> None:
    engine = HybridRetrievalEngine(
        vector_engine=FakeRetrievalEngine([]),
        bm25_engine=FakeRetrievalEngine([]),
    )

    assert engine.search("   ", top_k=3) == []
    assert engine.search("question", top_k=0) == []


def test_hybrid_retrieval_engine_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="tous les deux nuls"):
        HybridRetrievalEngine(
            vector_engine=FakeRetrievalEngine([]),
            bm25_engine=FakeRetrievalEngine([]),
            alpha=0,
            beta=0,
        )
