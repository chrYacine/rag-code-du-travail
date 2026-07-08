from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class LLMClient(Protocol):
    """Contract for text generation clients such as Groq."""

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate an answer from chat-style messages."""
