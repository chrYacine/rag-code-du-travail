from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.llm.contracts import LLMClient

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "hyde_prompt.md"


@dataclass(frozen=True)
class HyDEQuery:
    """Original lexical query paired with its vector-search expansion."""

    original_query: str
    vector_query: str
    used_hyde: bool


@dataclass
class HyDEGenerator:
    """Generate a hypothetical legal passage for dense retrieval."""

    llm_client: LLMClient
    prompt_path: Path = field(default=PROMPT_PATH)

    def expand(self, question: str) -> HyDEQuery:
        normalized_question = self._normalize(question)
        if not normalized_question:
            return HyDEQuery(
                original_query="",
                vector_query="",
                used_hyde=False,
            )

        try:
            hypothetical_document = self.llm_client.generate(
                [
                    {"role": "system", "content": self._load_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Question juridique à représenter pour la recherche :\n"
                            f"{normalized_question}"
                        ),
                    },
                ]
            )
        except Exception:
            return self._fallback(normalized_question)

        normalized_document = self._normalize(hypothetical_document)
        if not normalized_document:
            return self._fallback(normalized_question)

        return HyDEQuery(
            original_query=normalized_question,
            vector_query=normalized_document,
            used_hyde=normalized_document.casefold() != normalized_question.casefold(),
        )

    def _load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _fallback(question: str) -> HyDEQuery:
        return HyDEQuery(
            original_query=question,
            vector_query=question,
            used_hyde=False,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split())
