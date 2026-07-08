from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModerationResult:
    """Result returned by a moderation component."""

    is_allowed: bool
    reason: str = ""


class UserInputModerator(Protocol):
    """Contract for user input moderation before any LLM call."""

    def validate(self, text: str) -> ModerationResult:
        """Validate a user question."""
