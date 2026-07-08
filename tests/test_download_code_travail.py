from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src import download_code_travail
from src.download_code_travail import (
    CodeTravailDownloadError,
    build_metadata,
    download_code_du_travail,
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


def test_download_code_du_travail_valid_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_file = tmp_path / "raw" / "code_du_travail.json"
    metadata_file = tmp_path / "metadata" / "code_du_travail_metadata.json"

    monkeypatch.setattr(download_code_travail, "RAW_CODE_TRAVAIL_FILE", raw_file)
    monkeypatch.setattr(
        download_code_travail, "METADATA_CODE_TRAVAIL_FILE", metadata_file
    )
    monkeypatch.setattr(
        download_code_travail.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(valid_payload(), content=b'{"ok": true}'),
    )

    data = download_code_du_travail()

    assert data == valid_payload()
    assert json.loads(raw_file.read_text(encoding="utf-8")) == valid_payload()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["source_url"] == download_code_travail.CODE_TRAVAIL_URL
    assert metadata["technical_source"] == download_code_travail.TECHNICAL_SOURCE
    assert metadata["primary_source"] == download_code_travail.PRIMARY_SOURCE
    assert metadata["code_id"] == download_code_travail.CODE_TRAVAIL_ID
    assert "retrieved_at" in metadata


def test_download_code_du_travail_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        download_code_travail.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            valid_payload(),
            status_error=requests.exceptions.HTTPError("404"),
        ),
    )

    with pytest.raises(CodeTravailDownloadError, match="Erreur HTTP"):
        download_code_du_travail()


def test_download_code_du_travail_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        download_code_travail.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(json_error=ValueError("invalid json")),
    )

    with pytest.raises(CodeTravailDownloadError, match="JSON valide"):
        download_code_du_travail()


def test_build_metadata_contains_required_fields(tmp_path: Path) -> None:
    metadata = build_metadata(
        retrieved_at="2026-07-09T10:00:00+00:00",
        raw_file=tmp_path / "code_du_travail.json",
        content_size_bytes=123,
    )

    assert metadata["retrieved_at"] == "2026-07-09T10:00:00+00:00"
    assert metadata["source_url"]
    assert metadata["technical_source"] == "SocialGouv/legi-data"
    assert metadata["primary_source"] == "Légifrance"
    assert metadata["code_id"] == "LEGITEXT000006072050"
    assert metadata["content_size_bytes"] == 123
