from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from src.query_processing.hyde_generator import HyDEGenerator, HyDEQuery


@dataclass
class FakeLLMClient:
    response: str = ""
    error: Exception | None = None
    calls: list[Sequence[Mapping[str, str]]] = field(default_factory=list)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "hyde_prompt.md"
    path.write_text("Rédige un passage juridique hypothétique.", encoding="utf-8")
    return path


def test_expand_keeps_original_and_generates_vector_query(
    prompt_file: Path,
) -> None:
    llm = FakeLLMClient(
        response=(
            "Les heures effectuées au-delà de la durée légale du travail "
            "constituent des heures supplémentaires."
        )
    )
    generator = HyDEGenerator(llm, prompt_path=prompt_file)

    result = generator.expand("Comment fonctionnent les heures supplémentaires ?")

    assert result == HyDEQuery(
        original_query="Comment fonctionnent les heures supplémentaires ?",
        vector_query=(
            "Les heures effectuées au-delà de la durée légale du travail "
            "constituent des heures supplémentaires."
        ),
        used_hyde=True,
    )
    assert llm.calls[0][0] == {
        "role": "system",
        "content": "Rédige un passage juridique hypothétique.",
    }


def test_original_query_can_be_used_for_bm25(prompt_file: Path) -> None:
    llm = FakeLLMClient(response="Document hypothétique pour la recherche.")
    generator = HyDEGenerator(llm, prompt_path=prompt_file)

    result = generator.expand("Que dit l'article L3121-1 ?")

    assert result.original_query == "Que dit l'article L3121-1 ?"
    assert result.vector_query == "Document hypothétique pour la recherche."


@pytest.mark.parametrize("response", ["", "  \n "])
def test_empty_llm_response_falls_back_to_original_query(
    prompt_file: Path,
    response: str,
) -> None:
    generator = HyDEGenerator(
        FakeLLMClient(response=response),
        prompt_path=prompt_file,
    )
    question = "Quels congés sont prévus ?"

    assert generator.expand(question) == HyDEQuery(
        original_query=question,
        vector_query=question,
        used_hyde=False,
    )


def test_llm_error_falls_back_to_original_query(prompt_file: Path) -> None:
    generator = HyDEGenerator(
        FakeLLMClient(error=RuntimeError("Groq indisponible")),
        prompt_path=prompt_file,
    )
    question = "Comment fonctionne le licenciement ?"

    assert generator.expand(question) == HyDEQuery(
        original_query=question,
        vector_query=question,
        used_hyde=False,
    )


def test_missing_prompt_falls_back_to_original_query(tmp_path: Path) -> None:
    llm = FakeLLMClient(response="Réponse qui ne sera pas utilisée.")
    generator = HyDEGenerator(llm, prompt_path=tmp_path / "missing.md")
    question = "Quelle est la durée légale du travail ?"

    assert generator.expand(question) == HyDEQuery(
        original_query=question,
        vector_query=question,
        used_hyde=False,
    )


def test_empty_question_does_not_call_llm(prompt_file: Path) -> None:
    llm = FakeLLMClient(response="Texte")
    generator = HyDEGenerator(llm, prompt_path=prompt_file)

    assert generator.expand("   ") == HyDEQuery("", "", False)
    assert llm.calls == []


def test_identical_generated_text_is_not_marked_as_hyde(
    prompt_file: Path,
) -> None:
    question = "Que prévoit le Code du travail ?"
    generator = HyDEGenerator(
        FakeLLMClient(response=f"  {question}  "),
        prompt_path=prompt_file,
    )

    result = generator.expand(question)

    assert result.vector_query == question
    assert result.used_hyde is False
