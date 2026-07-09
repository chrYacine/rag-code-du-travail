from __future__ import annotations

import math

from src.rag.orchestrator import LEGAL_WARNING, RAGAnswer
from src.retrieval.contracts import RetrievedChunk
from src.ui.components import (
    answer_without_duplicate_warning,
    build_source_views,
    format_score,
)


def test_answer_warning_is_not_duplicated_in_ui_body() -> None:
    result = RAGAnswer(
        answer=f"Réponse juridique.\n\n{LEGAL_WARNING}",
        sources=[],
    )

    assert answer_without_duplicate_warning(result) == "Réponse juridique."


def test_answer_without_appended_warning_is_preserved() -> None:
    result = RAGAnswer(answer="Réponse juridique.", sources=[])

    assert answer_without_duplicate_warning(result) == "Réponse juridique."


def test_build_source_views_exposes_legal_metadata_and_scores() -> None:
    chunk = RetrievedChunk(
        content="Article L3121-1\nTexte juridique.",
        score=0.91234,
        metadata={
            "article_id": "L3121-1",
            "legi_id": "LEGIARTI0001",
            "theme": "duree du travail",
            "section": "Code du travail > Durée du travail",
            "primary_source": "Légifrance",
        },
        score_details={
            "vector_score": 0.8,
            "bm25_score": 0.7,
            "hybrid_score": 0.91234,
        },
    )

    views = build_source_views([chunk])

    assert len(views) == 1
    assert views[0].article_id == "L3121-1"
    assert views[0].legi_id == "LEGIARTI0001"
    assert views[0].source == "Légifrance"
    assert views[0].score == "0.9123"
    assert views[0].score_details == {
        "vector_score": "0.8000",
        "bm25_score": "0.7000",
        "hybrid_score": "0.9123",
    }


def test_technical_source_is_used_when_primary_source_is_missing() -> None:
    chunk = RetrievedChunk(
        content="Texte",
        score=0.5,
        metadata={"article_id": "L1-1", "source": "SocialGouv/legi-data"},
    )

    assert build_source_views([chunk])[0].source == "SocialGouv/legi-data"


def test_non_finite_scores_are_not_displayed_as_numbers() -> None:
    assert format_score(math.nan) == "indisponible"
    assert format_score(math.inf) == "indisponible"
