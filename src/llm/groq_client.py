from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import requests

from src.config import GROQ_API_URL, GROQ_MODEL, GROQ_REQUEST_TIMEOUT


class GroqClientError(RuntimeError):
    """Raised when Groq cannot generate a usable response."""


@dataclass
class GroqClient:
    """Small REST client implementing the project's LLMClient contract."""

    api_key: str | None = None
    model: str = GROQ_MODEL
    api_url: str = GROQ_API_URL
    timeout: int = GROQ_REQUEST_TIMEOUT

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise GroqClientError(
                "Clé GROQ_API_KEY manquante. "
                "Ajoutez votre clé dans le fichier .env local."
            )

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        payload_messages = self._validate_messages(messages)

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "messages": payload_messages},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise GroqClientError("Le service Groq n'a pas répondu à temps.") from exc
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", "inconnu")
            raise GroqClientError(
                f"Le service Groq a retourné une erreur HTTP {status_code}."
            ) from exc
        except requests.RequestException as exc:
            raise GroqClientError("Le service Groq est indisponible.") from exc

        return self._extract_content(response)

    @staticmethod
    def _validate_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            raise GroqClientError("Au moins un message est requis.")

        validated: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if (
                not isinstance(role, str)
                or not role.strip()
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise GroqClientError(
                    "Chaque message doit contenir un rôle et un contenu textuels."
                )
            validated.append({"role": role, "content": content})
        return validated

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            body = response.json()
        except ValueError as exc:
            raise GroqClientError(
                "Le service Groq a retourné une réponse JSON invalide."
            ) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqClientError(
                "Le service Groq a retourné une réponse incomplète."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise GroqClientError("Le service Groq a retourné une réponse vide.")
        return content.strip()
