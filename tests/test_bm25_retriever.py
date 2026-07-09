from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import (
    BM25RetrievalEngine,
    BM25RetrievalError,
    BM25Scorer,
    build_retrieved_chunk,
    extract_article_ids,
    load_chunks,
    tokenize,
)
from src.retrieval.contracts import RetrievedChunk


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sample_chunks() -> list[dict[str, object]]:
    return [
        {
            "id": "LEGIARTI1",
            "content": "Article L3121-1\nLa duree du travail effectif est definie.",
            "metadata": {
                "article_id": "L3121-1",
                "legi_id": "LEGIARTI1",
                "theme": "duree du travail et heures supplementaires",
            },
        },
        {
            "id": "LEGIARTI2",
            "content": "Article L3141-1\nTout salarie a droit aux conges payes.",
            "metadata": {
                "article_id": "L3141-1",
                "legi_id": "LEGIARTI2",
                "theme": "conges payes",
            },
        },
        {
            "id": "LEGIARTI3",
            "content": "Article L1237-11\nLa rupture conventionnelle est encadree.",
            "metadata": {
                "article_id": "L1237-11",
                "legi_id": "LEGIARTI3",
                "theme": "rupture conventionnelle",
            },
        },
    ]


def test_tokenize_normalizes_accents_and_keeps_article_numbers() -> None:
    assert tokenize("Durée du travail - Article L3121-1") == [
        "duree",
        "du",
        "travail",
        "article",
        "l3121-1",
    ]


def test_extract_article_ids_supports_suffix_articles() -> None:
    assert extract_article_ids("Voir l'article L1221-10-1 et L1237-11") == {
        "L1221-10-1",
        "L1237-11",
    }


def test_load_chunks_rejects_non_string_metadata(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.json"
    write_json(
        chunks_path,
        [
            {
                "content": "Article L3121-1\nTexte",
                "metadata": {"article_id": "L3121-1", "score": 1},
            }
        ],
    )

    with pytest.raises(BM25RetrievalError, match="Valeurs metadata"):
        load_chunks(chunks_path)


def test_bm25_scorer_ranks_matching_document_first() -> None:
    documents = [
        tokenize("duree travail effectif"),
        tokenize("conges payes salarie"),
    ]
    scorer = BM25Scorer(documents)

    scores = scorer.score(tokenize("conges payes"))

    assert scores[1] > scores[0]


def test_build_retrieved_chunk_uses_contract_metadata() -> None:
    chunk = sample_chunks()[0]

    retrieved = build_retrieved_chunk(item=chunk, bm25_score=1.25)

    assert isinstance(retrieved, RetrievedChunk)
    assert retrieved.content.startswith("Article L3121-1")
    assert retrieved.score == 1.25
    assert retrieved.metadata["article_id"] == "L3121-1"
    assert retrieved.metadata["legi_id"] == "LEGIARTI1"
    assert retrieved.score_details == {"bm25_score": 1.25}


def test_bm25_retrieval_engine_returns_ranked_chunks(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks_code_du_travail.json"
    write_json(chunks_path, sample_chunks())

    engine = BM25RetrievalEngine(chunks_path=chunks_path)
    results = engine.search("conges payes", top_k=2)

    assert results[0].metadata["article_id"] == "L3141-1"
    assert results[0].score_details["bm25_score"] == results[0].score


def test_bm25_retrieval_engine_prioritizes_exact_article_id(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks_code_du_travail.json"
    write_json(chunks_path, sample_chunks())

    engine = BM25RetrievalEngine(chunks_path=chunks_path, article_id_boost=10.0)
    results = engine.search("Que dit L1237-11 sur les conges ?", top_k=2)

    assert results[0].metadata["article_id"] == "L1237-11"
    assert results[0].score_details["bm25_score"] == results[0].score


def test_bm25_retrieval_engine_returns_empty_list_for_empty_question(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks_code_du_travail.json"
    write_json(chunks_path, sample_chunks())

    engine = BM25RetrievalEngine(chunks_path=chunks_path)

    assert engine.search("   ", top_k=2) == []
    assert engine.search("conges payes", top_k=0) == []
