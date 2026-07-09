from __future__ import annotations

import pytest

from src.query_code_travail import (
    format_answer,
    format_score_details,
    main,
    parse_args,
)
from src.rag.orchestrator import LEGAL_WARNING, RAGAnswer
from src.retrieval.contracts import RetrievedChunk


def source_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="Article L1237-11\nTexte",
        score=0.98,
        metadata={
            "article_id": "L1237-11",
            "legi_id": "LEGIARTI123",
            "theme": "rupture conventionnelle",
            "section": "Code du travail > Rupture",
        },
        score_details={
            "vector_score": 0.7,
            "bm25_score": 1.0,
            "hybrid_score": 0.98,
        },
    )


def test_parse_args_reads_question_and_top_k() -> None:
    args = parse_args(["Que dit L1237-11 ?", "--top-k", "3"])

    assert args.question == "Que dit L1237-11 ?"
    assert args.top_k == 3


def test_format_score_details_displays_available_scores() -> None:
    assert (
        format_score_details(
            {
                "vector_score": 0.75,
                "bm25_score": 1.0,
                "hybrid_score": 0.825,
            }
        )
        == "vector_score=0.7500 | bm25_score=1.0000 | hybrid_score=0.8250"
    )


def test_format_answer_displays_answer_sources_and_warning() -> None:
    result = RAGAnswer(
        answer="La rupture conventionnelle est encadree.",
        sources=[source_chunk()],
    )

    formatted = format_answer(result)

    assert "Reponse" in formatted
    assert "La rupture conventionnelle est encadree." in formatted
    assert "Article L1237-11" in formatted
    assert "legi_id: LEGIARTI123" in formatted
    assert "hybrid_score=0.9800" in formatted
    assert LEGAL_WARNING in formatted


def test_format_answer_does_not_duplicate_warning() -> None:
    result = RAGAnswer(
        answer=f"Reponse.\n\n{LEGAL_WARNING}",
        sources=[],
    )

    formatted = format_answer(result)

    assert formatted.count(LEGAL_WARNING) == 1


def test_main_rejects_empty_question(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["   "])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "question ne peut pas etre vide" in captured.out


def test_main_prints_formatted_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_run_query(question: str, *, top_k: int) -> RAGAnswer:
        calls.append((question, top_k))
        return RAGAnswer(answer="Reponse.", sources=[source_chunk()])

    monkeypatch.setattr("src.query_code_travail.run_query", fake_run_query)

    exit_code = main(["Que dit L1237-11 ?", "--top-k", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [("Que dit L1237-11 ?", 2)]
    assert "Reponse." in captured.out
    assert "Article L1237-11" in captured.out


def test_main_returns_error_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run_query(question: str, *, top_k: int) -> RAGAnswer:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.query_code_travail.run_query", fail_run_query)

    exit_code = main(["Question valide ?"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "question n'a pas pu etre traitee" in captured.out
