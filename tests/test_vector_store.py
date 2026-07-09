from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval import vector_store
from src.retrieval.vector_store import (
    VectorStoreError,
    build_parallel_metadata,
    build_vector_store,
    compute_file_sha256,
    should_rebuild_vector_store,
    validate_chunks,
)


class FakeEmbeddings:
    shape = (2, 3)

    def astype(self, dtype: str) -> "FakeEmbeddings":
        assert dtype == "float32"
        return self


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sample_chunks() -> list[dict[str, object]]:
    return [
        {
            "id": "LEGIARTI1",
            "content": "Article L3121-1\nTexte un",
            "metadata": {
                "article_id": "L3121-1",
                "legi_id": "LEGIARTI1",
                "theme": "duree du travail et heures supplementaires",
            },
        },
        {
            "id": "LEGIARTI2",
            "content": "Article L3141-1\nTexte deux",
            "metadata": {
                "article_id": "L3141-1",
                "legi_id": "LEGIARTI2",
                "theme": "conges payes",
            },
        },
    ]


def test_compute_file_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "chunks.json"
    path.write_text("same content", encoding="utf-8")

    assert compute_file_sha256(path) == compute_file_sha256(path)


def test_should_rebuild_vector_store_detects_unchanged_store(tmp_path: Path) -> None:
    index_path = tmp_path / "index.faiss"
    chunks_metadata_path = tmp_path / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store_metadata.json"
    index_path.write_bytes(b"index")
    write_json(chunks_metadata_path, [])
    write_json(vector_metadata_path, {"chunks_hash_sha256": "abc"})

    assert (
        should_rebuild_vector_store(
            chunks_hash="abc",
            index_path=index_path,
            chunks_metadata_path=chunks_metadata_path,
            vector_metadata_path=vector_metadata_path,
        )
        is False
    )


def test_should_rebuild_vector_store_when_hash_changes(tmp_path: Path) -> None:
    index_path = tmp_path / "index.faiss"
    chunks_metadata_path = tmp_path / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store_metadata.json"
    index_path.write_bytes(b"index")
    write_json(chunks_metadata_path, [])
    write_json(vector_metadata_path, {"chunks_hash_sha256": "old"})

    assert (
        should_rebuild_vector_store(
            chunks_hash="new",
            index_path=index_path,
            chunks_metadata_path=chunks_metadata_path,
            vector_metadata_path=vector_metadata_path,
        )
        is True
    )


def test_should_rebuild_vector_store_when_force_is_enabled(tmp_path: Path) -> None:
    assert (
        should_rebuild_vector_store(
            chunks_hash="abc",
            index_path=tmp_path / "index.faiss",
            chunks_metadata_path=tmp_path / "chunks_metadata.json",
            vector_metadata_path=tmp_path / "metadata.json",
            force=True,
        )
        is True
    )


def test_validate_chunks_rejects_non_string_metadata() -> None:
    chunks = [
        {
            "id": "LEGIARTI1",
            "content": "Article L3121-1\nTexte",
            "metadata": {"article_id": "L3121-1", "score": 1},
        }
    ]

    with pytest.raises(VectorStoreError, match="valeurs metadata"):
        validate_chunks(chunks)


def test_build_parallel_metadata_preserves_vector_order() -> None:
    metadata = build_parallel_metadata(sample_chunks())

    assert [item["vector_id"] for item in metadata] == [0, 1]
    assert metadata[0]["metadata"]["article_id"] == "L3121-1"
    assert metadata[1]["metadata"]["article_id"] == "L3141-1"


def test_build_vector_store_writes_index_and_parallel_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks_path = tmp_path / "processed" / "chunks_code_du_travail.json"
    index_path = tmp_path / "vector_store" / "index.faiss"
    chunks_metadata_path = tmp_path / "vector_store" / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store" / "vector_store_metadata.json"
    write_json(chunks_path, sample_chunks())

    monkeypatch.setattr(
        vector_store, "load_embedding_model", lambda model_name: object()
    )
    monkeypatch.setattr(
        vector_store, "encode_texts", lambda model, texts: FakeEmbeddings()
    )
    monkeypatch.setattr(
        vector_store, "build_faiss_index", lambda embeddings: {"index": True}
    )

    def fake_write_index(index: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake faiss")

    monkeypatch.setattr(vector_store, "write_faiss_index", fake_write_index)

    status = build_vector_store(
        embedding_model="fake-model",
        chunks_path=chunks_path,
        index_path=index_path,
        chunks_metadata_path=chunks_metadata_path,
        vector_metadata_path=vector_metadata_path,
    )

    assert status["status"] == "updated"
    assert status["rebuilt"] is True
    assert index_path.exists()

    parallel_metadata = json.loads(chunks_metadata_path.read_text(encoding="utf-8"))
    assert parallel_metadata[0]["vector_id"] == 0
    assert parallel_metadata[0]["content"].startswith("Article L3121-1")

    vector_metadata = json.loads(vector_metadata_path.read_text(encoding="utf-8"))
    assert vector_metadata["embedding_model"] == "fake-model"
    assert vector_metadata["embedding_dimension"] == 3
    assert vector_metadata["chunks_count"] == 2


def test_build_vector_store_skips_when_chunks_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks_path = tmp_path / "processed" / "chunks_code_du_travail.json"
    index_path = tmp_path / "vector_store" / "index.faiss"
    chunks_metadata_path = tmp_path / "vector_store" / "chunks_metadata.json"
    vector_metadata_path = tmp_path / "vector_store" / "vector_store_metadata.json"
    write_json(chunks_path, sample_chunks())
    chunks_hash = compute_file_sha256(chunks_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"index")
    write_json(chunks_metadata_path, [])
    write_json(vector_metadata_path, {"chunks_hash_sha256": chunks_hash})

    def fail_if_called(model_name: str) -> object:
        raise AssertionError("model should not be loaded when chunks are unchanged")

    monkeypatch.setattr(vector_store, "load_embedding_model", fail_if_called)

    status = build_vector_store(
        chunks_path=chunks_path,
        index_path=index_path,
        chunks_metadata_path=chunks_metadata_path,
        vector_metadata_path=vector_metadata_path,
    )

    assert status["status"] == "unchanged"
    assert status["rebuilt"] is False
