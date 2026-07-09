from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.retrieval import vector_retriever
from src.retrieval.contracts import RetrievedChunk
from src.retrieval.vector_retriever import (
    VectorRetrievalEngine,
    VectorRetrievalError,
    build_retrieved_chunk,
    load_vector_store_metadata,
)


class FakeModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert texts == ["temps de travail"]
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return np.array([[0.1, 0.2, 0.3]], dtype="float32")


class FakeIndex:
    ntotal = 2

    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert query_embedding.shape == (1, 3)
        assert top_k == 2
        return (
            np.array([[0.91, 0.72]], dtype="float32"),
            np.array([[1, 0]], dtype="int64"),
        )


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sample_parallel_metadata() -> list[dict[str, object]]:
    return [
        {
            "vector_id": 0,
            "id": "LEGIARTI1",
            "content": "Article L3141-1\nTexte conges",
            "metadata": {
                "article_id": "L3141-1",
                "legi_id": "LEGIARTI1",
                "theme": "conges payes",
            },
        },
        {
            "vector_id": 1,
            "id": "LEGIARTI2",
            "content": "Article L3121-1\nTexte duree",
            "metadata": {
                "article_id": "L3121-1",
                "legi_id": "LEGIARTI2",
                "theme": "duree du travail et heures supplementaires",
            },
        },
    ]


def test_load_vector_store_metadata_requires_embedding_model(tmp_path: Path) -> None:
    metadata_path = tmp_path / "vector_store_metadata.json"
    write_json(metadata_path, {"chunks_count": 2})

    with pytest.raises(VectorRetrievalError, match="modèle d'embedding"):
        load_vector_store_metadata(metadata_path)


def test_build_retrieved_chunk_uses_contract_metadata() -> None:
    item = sample_parallel_metadata()[0]

    retrieved = build_retrieved_chunk(item=item, vector_score=0.8)

    assert isinstance(retrieved, RetrievedChunk)
    assert retrieved.content.startswith("Article L3141-1")
    assert retrieved.score == 0.8
    assert retrieved.metadata["article_id"] == "L3141-1"
    assert retrieved.metadata["legi_id"] == "LEGIARTI1"
    assert retrieved.score_details == {"vector_score": 0.8}


def test_vector_retrieval_engine_returns_ranked_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.faiss"
    chunks_metadata_path = tmp_path / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store_metadata.json"
    index_path.write_bytes(b"fake index")
    write_json(chunks_metadata_path, sample_parallel_metadata())
    write_json(
        vector_metadata_path,
        {"embedding_model": "model-from-index", "chunks_count": 2},
    )

    loaded_models: list[str] = []

    def fake_load_model(model_name: str) -> FakeModel:
        loaded_models.append(model_name)
        return FakeModel(model_name)

    monkeypatch.setattr(vector_retriever, "load_faiss_index", lambda path: FakeIndex())
    monkeypatch.setattr(vector_retriever, "load_embedding_model", fake_load_model)

    engine = VectorRetrievalEngine(
        index_path=index_path,
        chunks_metadata_path=chunks_metadata_path,
        vector_metadata_path=vector_metadata_path,
    )
    results = engine.search("temps de travail", top_k=2)

    assert loaded_models == ["model-from-index"]
    assert [result.metadata["article_id"] for result in results] == [
        "L3121-1",
        "L3141-1",
    ]
    assert results[0].score == pytest.approx(0.91)
    assert results[0].score_details["vector_score"] == pytest.approx(0.91)


def test_vector_retrieval_engine_returns_empty_list_for_empty_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.faiss"
    chunks_metadata_path = tmp_path / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store_metadata.json"
    index_path.write_bytes(b"fake index")
    write_json(chunks_metadata_path, sample_parallel_metadata())
    write_json(vector_metadata_path, {"embedding_model": "model-from-index"})

    monkeypatch.setattr(vector_retriever, "load_faiss_index", lambda path: FakeIndex())
    monkeypatch.setattr(
        vector_retriever,
        "load_embedding_model",
        lambda model_name: FakeModel(model_name),
    )

    engine = VectorRetrievalEngine(
        index_path=index_path,
        chunks_metadata_path=chunks_metadata_path,
        vector_metadata_path=vector_metadata_path,
    )

    assert engine.search("   ", top_k=2) == []
    assert engine.search("temps de travail", top_k=0) == []
