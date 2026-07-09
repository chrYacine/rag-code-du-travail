from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from src import download_code_travail
from src.download_code_travail import (
    STATUS_FORCED,
    STATUS_INITIAL,
    STATUS_UNCHANGED,
    build_metadata,
    compute_sha256,
    download_code_du_travail,
    has_source_changed,
)


class FakeResponse:
    def __init__(
        self,
        payload: object | None = None,
        *,
        content: bytes = b"{}",
        status_error: Exception | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.content = content
        self.status_error = status_error
        self.json_error = json_error

    def raise_for_status(self) -> None:
        if self.status_error:
            raise self.status_error

    def json(self) -> object:
        if self.json_error:
            raise self.json_error
        return self.payload


def valid_payload() -> dict[str, object]:
    return {"type": "code", "data": {}, "children": [{"type": "article"}]}


def patch_output_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    raw_file = tmp_path / "raw" / "code_du_travail.json"
    metadata_file = tmp_path / "metadata" / "code_du_travail_metadata.json"
    monkeypatch.setattr(download_code_travail, "RAW_CODE_TRAVAIL_FILE", raw_file)
    monkeypatch.setattr(
        download_code_travail, "METADATA_CODE_TRAVAIL_FILE", metadata_file
    )
    return raw_file, metadata_file


def patch_response(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: bytes,
    payload: object | None = None,
    status_error: Exception | None = None,
    json_error: Exception | None = None,
) -> None:
    monkeypatch.setattr(
        download_code_travail.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            payload if payload is not None else valid_payload(),
            content=content,
            status_error=status_error,
            json_error=json_error,
        ),
    )


def write_previous_metadata(metadata_file: Path, content_hash: str) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps({"content_hash_sha256": content_hash}),
        encoding="utf-8",
    )


def test_compute_sha256_is_stable_for_identical_content() -> None:
    assert compute_sha256(b"same content") == compute_sha256(b"same content")


def test_compute_sha256_differs_for_different_content() -> None:
    assert compute_sha256(b"first content") != compute_sha256(b"second content")


def test_has_source_changed_returns_false_for_same_hash(tmp_path: Path) -> None:
    content_hash = compute_sha256(b"remote")
    metadata_file = tmp_path / "metadata.json"
    write_previous_metadata(metadata_file, content_hash)

    assert has_source_changed(content_hash, metadata_file) is False


def test_has_source_changed_returns_true_for_new_hash(tmp_path: Path) -> None:
    metadata_file = tmp_path / "metadata.json"
    write_previous_metadata(metadata_file, compute_sha256(b"old"))

    assert has_source_changed(compute_sha256(b"new"), metadata_file) is True


def test_download_code_du_travail_initializes_without_previous_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_file, metadata_file = patch_output_paths(monkeypatch, tmp_path)
    patch_response(monkeypatch, content=b'{"version": 1}')

    with caplog.at_level(logging.INFO):
        status = download_code_du_travail()

    assert status["status"] == "updated"
    assert status["downloaded"] is True
    assert status["changed"] is True
    assert status["forced"] is False
    assert STATUS_INITIAL in caplog.text
    assert json.loads(raw_file.read_text(encoding="utf-8")) == valid_payload()

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["source"] == download_code_travail.TECHNICAL_SOURCE
    assert metadata["url"] == download_code_travail.CODE_TRAVAIL_URL
    assert metadata["primary_source"] == download_code_travail.PRIMARY_SOURCE
    assert metadata["code_id"] == download_code_travail.CODE_TRAVAIL_ID
    assert metadata["content_hash_sha256"] == compute_sha256(b'{"version": 1}')
    assert metadata["has_changed"] is True
    assert metadata["forced"] is False
    assert "retrieved_at" in metadata


def test_download_code_du_travail_detects_unchanged_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_file, metadata_file = patch_output_paths(monkeypatch, tmp_path)
    content = b'{"version": 1}'
    write_previous_metadata(metadata_file, compute_sha256(content))
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("previous local file", encoding="utf-8")
    patch_response(monkeypatch, content=content)

    with caplog.at_level(logging.INFO):
        status = download_code_du_travail()

    assert status["status"] == "unchanged"
    assert status["downloaded"] is True
    assert status["changed"] is False
    assert status["forced"] is False
    assert status["message"] == STATUS_UNCHANGED
    assert raw_file.read_text(encoding="utf-8") == "previous local file"
    assert STATUS_UNCHANGED in caplog.text


def test_download_code_du_travail_detects_changed_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_file, metadata_file = patch_output_paths(monkeypatch, tmp_path)
    write_previous_metadata(metadata_file, compute_sha256(b"old content"))
    patch_response(monkeypatch, content=b"new content")

    status = download_code_du_travail()

    assert status["status"] == "updated"
    assert status["downloaded"] is True
    assert status["changed"] is True
    assert json.loads(raw_file.read_text(encoding="utf-8")) == valid_payload()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["content_hash_sha256"] == compute_sha256(b"new content")
    assert metadata["has_changed"] is True


def test_download_code_du_travail_force_rewrites_unchanged_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_file, metadata_file = patch_output_paths(monkeypatch, tmp_path)
    content = b'{"version": 1}'
    write_previous_metadata(metadata_file, compute_sha256(content))
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("previous local file", encoding="utf-8")
    patch_response(monkeypatch, content=content)

    with caplog.at_level(logging.INFO):
        status = download_code_du_travail(force=True)

    assert status["status"] == "forced"
    assert status["downloaded"] is True
    assert status["changed"] is False
    assert status["forced"] is True
    assert status["message"] == STATUS_FORCED
    assert json.loads(raw_file.read_text(encoding="utf-8")) == valid_payload()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["forced"] is True
    assert metadata["has_changed"] is False
    assert STATUS_FORCED in caplog.text


def test_download_code_du_travail_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_response(
        monkeypatch,
        content=b"",
        status_error=requests.exceptions.HTTPError("404"),
    )

    status = download_code_du_travail()

    assert status["status"] == "error"
    assert status["downloaded"] is False
    assert status["changed"] is None
    assert "Erreur HTTP" in status["message"]


def test_download_code_du_travail_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_connection_error(*args: object, **kwargs: object) -> None:
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(download_code_travail.requests, "get", raise_connection_error)

    status = download_code_du_travail()

    assert status["status"] == "error"
    assert status["downloaded"] is False
    assert status["changed"] is None
    assert "Erreur réseau" in status["message"]


def test_download_code_du_travail_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_response(
        monkeypatch,
        content=b"invalid",
        json_error=ValueError("invalid json"),
    )

    status = download_code_du_travail()

    assert status["status"] == "error"
    assert status["downloaded"] is False
    assert status["changed"] is None
    assert "JSON valide" in status["message"]


def test_build_metadata_contains_required_fields(tmp_path: Path) -> None:
    content_hash = compute_sha256(b"content")
    metadata = build_metadata(
        retrieved_at="2026-07-09T10:00:00+00:00",
        raw_file=tmp_path / "code_du_travail.json",
        content_size_bytes=123,
        content_hash_sha256=content_hash,
        has_changed=True,
        forced=False,
    )

    assert metadata["retrieved_at"] == "2026-07-09T10:00:00+00:00"
    assert metadata["source"] == "SocialGouv/legi-data"
    assert metadata["technical_source"] == "SocialGouv/legi-data"
    assert metadata["url"]
    assert metadata["source_url"]
    assert metadata["primary_source"] == "Légifrance"
    assert metadata["code_id"] == "LEGITEXT000006072050"
    assert metadata["content_hash_sha256"] == content_hash
    assert metadata["has_changed"] is True
    assert metadata["forced"] is False
    assert metadata["content_size_bytes"] == 123
