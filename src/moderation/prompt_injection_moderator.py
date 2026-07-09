from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar, Pattern

from src.moderation.contracts import ModerationResult

EMPTY_REASON = "La question est vide."
TOO_LONG_REASON = "La question dépasse la longueur maximale autorisée."
UNSAFE_REASON = "La question contient des instructions potentiellement malveillantes."


@dataclass(frozen=True)
class PromptInjectionModerator:
    """Block explicit attempts to override or expose application instructions."""

    max_length: int = 4_000

    _INJECTION_PATTERNS: ClassVar[tuple[Pattern[str], ...]] = (
        re.compile(
            r"\b(?:ignore|forget|disregard|override)\b.{0,60}"
            r"\b(?:previous|prior|system|developer|hidden)\b.{0,30}"
            r"\b(?:instruction|instructions|message|prompt|rules?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\b(?:ignore|oublie|oubliez|contourne|contournez|remplace|remplacez)"
            r"\b.{0,60}\b(?:instruction|instructions|consigne|consignes|règle|règles)"
            r"\b.{0,40}\b(?:précédent|précédente|précédentes|système|caché|cachées)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\b(?:révèle|révélez|affiche|affichez|montre|montrez|donne|donnez|"
            r"reveal|show|display|print)\b.{0,60}"
            r"\b(?:system prompt|prompt système|instructions? système|"
            r"instructions? cachées?|developer message)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\b(?:act as|agis comme|fais comme si)\b.{0,60}"
            r"\b(?:no rules|sans règles|sans restrictions?|unrestricted|dan)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\b(?:jailbreak|mode développeur|developer mode)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:<\|(?:system|assistant|developer)\|>|"
            r"\[(?:system|assistant|developer)\]|"
            r"###\s*(?:system|assistant|developer)\s*:)",
            re.IGNORECASE,
        ),
    )

    def validate(self, text: str) -> ModerationResult:
        normalized = self._normalize(text)
        if not normalized:
            return ModerationResult(is_allowed=False, reason=EMPTY_REASON)

        if len(normalized) > self.max_length:
            return ModerationResult(is_allowed=False, reason=TOO_LONG_REASON)

        if self._contains_forbidden_control_character(normalized):
            return ModerationResult(is_allowed=False, reason=UNSAFE_REASON)

        if any(pattern.search(normalized) for pattern in self._INJECTION_PATTERNS):
            return ModerationResult(is_allowed=False, reason=UNSAFE_REASON)

        return ModerationResult(is_allowed=True)

    @staticmethod
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        return " ".join(normalized.split())

    @staticmethod
    def _contains_forbidden_control_character(text: str) -> bool:
        return any(
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\r", "\t"}
            for character in text
        )
