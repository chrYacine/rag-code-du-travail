from __future__ import annotations

from typing import Any

import pytest
import requests

from src.llm.groq_client import GroqClient, GroqClientError


class FakeResponse:
    def __init__(
        self,
        body: Any = None,
        *,
        status_code: int = 200,
        json_error: ValueError | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> Any:
        if self.json_error:
            raise self.json_error
        return self.body


def test_generate_returns_assistant_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {"choices": [{"message": {"content": "  Réponse juridique.  "}}]}
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = GroqClient(api_key="test-key", model="test-model", timeout=12)

    result = client.generate([{"role": "user", "content": "Ma question"}])

    assert result == "Réponse juridique."
    assert captured["json"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Ma question"}],
    }
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 12


def test_missing_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    expected_message = (
        "Clé GROQ_API_KEY manquante. " "Ajoutez votre clé dans le fichier .env local."
    )

    with pytest.raises(GroqClientError, match="GROQ_API_KEY") as error:
        GroqClient()

    assert str(error.value) == expected_message


def test_empty_messages_are_rejected() -> None:
    client = GroqClient(api_key="test-key")

    with pytest.raises(GroqClientError, match="message"):
        client.generate([])


def test_timeout_is_wrapped_without_exposing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        raise requests.Timeout("secret-test-key")

    monkeypatch.setattr(requests, "post", fake_post)
    client = GroqClient(api_key="secret-test-key")

    with pytest.raises(GroqClientError, match="temps") as error:
        client.generate([{"role": "user", "content": "Question"}])

    assert "secret-test-key" not in str(error.value)


def test_http_error_is_wrapped_with_status_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code=429),
    )
    client = GroqClient(api_key="secret-test-key")

    with pytest.raises(GroqClientError, match="429") as error:
        client.generate([{"role": "user", "content": "Question"}])

    assert "secret-test-key" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(json_error=ValueError("invalid JSON")),
        FakeResponse({}),
        FakeResponse({"choices": [{"message": {"content": ""}}]}),
    ],
)
def test_invalid_responses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeResponse,
) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    client = GroqClient(api_key="test-key")

    with pytest.raises(GroqClientError):
        client.generate([{"role": "user", "content": "Question"}])
