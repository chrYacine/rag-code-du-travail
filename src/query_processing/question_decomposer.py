from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.llm.contracts import LLMClient

PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "question_decomposition_prompt.md"
)

COMPARISON_MARKERS = (
    "compare",
    "comparaison",
    "différence entre",
    "différences entre",
    "versus",
    " vs ",
    "d'une part",
    "d’autre part",
    "d'autre part",
)

TOPIC_HINTS = (
    ("harcèlement", "discrimination"),
    ("contrat de travail", "cdd", "cdi"),
    ("période d'essai", "periode d'essai"),
    ("licenciement",),
    ("rupture conventionnelle",),
    ("représentation du personnel", "cse"),
    ("durée du travail", "heures supplémentaires", "temps de travail"),
    ("congés payés",),
    ("salaire minimum", "smic", "rémunération"),
)


@dataclass
class QuestionDecomposer:
    """Turn a complex legal question into a few focused retrieval queries."""

    llm_client: LLMClient
    max_sub_questions: int = 4
    prompt_path: Path = field(default=PROMPT_PATH)

    def __post_init__(self) -> None:
        if not 2 <= self.max_sub_questions <= 4:
            raise ValueError("max_sub_questions doit être compris entre 2 et 4.")

    def decompose(self, question: str) -> list[str]:
        normalized_question = self._normalize(question)
        if not normalized_question:
            return []

        if not self.should_decompose(normalized_question):
            return [normalized_question]

        try:
            response = self.llm_client.generate(
                [
                    {"role": "system", "content": self._load_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Décompose la question suivante :\n"
                            f"{normalized_question}"
                        ),
                    },
                ]
            )
            sub_questions = self._parse_response(response)
        except Exception:
            return [normalized_question]

        return sub_questions or [normalized_question]

    @staticmethod
    def should_decompose(question: str) -> bool:
        lowered = question.casefold()
        if any(marker in lowered for marker in COMPARISON_MARKERS):
            return True

        matched_topics = sum(
            any(hint in lowered for hint in topic_hints) for topic_hints in TOPIC_HINTS
        )
        return matched_topics >= 2

    def _load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8").strip()

    def _parse_response(self, response: str) -> list[str]:
        cleaned_response = self._strip_code_fence(response)
        try:
            parsed = json.loads(cleaned_response)
        except (json.JSONDecodeError, TypeError):
            return []

        if isinstance(parsed, dict):
            candidates = parsed.get("sub_questions")
        else:
            candidates = parsed

        if not isinstance(candidates, list):
            return []

        unique_questions: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            normalized = self._normalize(candidate)
            deduplication_key = normalized.casefold()
            if not normalized or deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            unique_questions.append(normalized)
            if len(unique_questions) == self.max_sub_questions:
                break

        if len(unique_questions) < 2:
            return []
        return unique_questions

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        stripped = value.strip()
        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else stripped

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split())
