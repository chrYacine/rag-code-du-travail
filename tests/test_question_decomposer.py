from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from src.query_processing.question_decomposer import QuestionDecomposer


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
    path = tmp_path / "question_decomposition_prompt.md"
    path.write_text("Retourne uniquement du JSON.", encoding="utf-8")
    return path


def test_simple_question_is_returned_without_llm_call(prompt_file: Path) -> None:
    llm = FakeLLMClient()
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    result = decomposer.decompose("  Que dit l'article L3121-1 ?  ")

    assert result == ["Que dit l'article L3121-1 ?"]
    assert llm.calls == []


def test_comparative_question_is_decomposed(prompt_file: Path) -> None:
    llm = FakeLLMClient(
        response=(
            '{"sub_questions": ['
            '"Quelles sont les règles des congés payés ?", '
            '"Quelles sont les règles des heures supplémentaires ?"'
            "]}"
        )
    )
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    result = decomposer.decompose(
        "Compare les congés payés et les heures supplémentaires."
    )

    assert result == [
        "Quelles sont les règles des congés payés ?",
        "Quelles sont les règles des heures supplémentaires ?",
    ]
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == {
        "role": "system",
        "content": "Retourne uniquement du JSON.",
    }


def test_three_detected_topics_trigger_decomposition(prompt_file: Path) -> None:
    llm = FakeLLMClient(response="""```json
        {
          "sub_questions": [
            "Quelles règles concernent les congés payés ?",
            "Quelles règles concernent les heures supplémentaires ?",
            "Quelles règles concernent la rupture conventionnelle ?"
          ]
        }
        ```""")
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    result = decomposer.decompose(
        "Explique les congés payés, les heures supplémentaires "
        "et la rupture conventionnelle."
    )

    assert len(result) == 3


def test_duplicate_and_empty_questions_are_removed(prompt_file: Path) -> None:
    llm = FakeLLMClient(
        response=(
            "["
            '"Question sur les congés ?", '
            '"  question sur les congés ?  ", '
            '"", '
            '"Question sur le licenciement ?"'
            "]"
        )
    )
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    result = decomposer.decompose("Compare les congés et le licenciement.")

    assert result == [
        "Question sur les congés ?",
        "Question sur le licenciement ?",
    ]


def test_result_is_limited_to_four_sub_questions(prompt_file: Path) -> None:
    llm = FakeLLMClient(
        response=(
            "["
            '"Question 1 ?", "Question 2 ?", "Question 3 ?", '
            '"Question 4 ?", "Question 5 ?"'
            "]"
        )
    )
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    result = decomposer.decompose("Compare plusieurs règles du travail.")

    assert result == [
        "Question 1 ?",
        "Question 2 ?",
        "Question 3 ?",
        "Question 4 ?",
    ]


@pytest.mark.parametrize(
    "response",
    [
        "pas du json",
        "{}",
        '{"sub_questions": []}',
        '{"sub_questions": ["Une seule question ?"]}',
    ],
)
def test_invalid_llm_output_falls_back_to_original_question(
    prompt_file: Path,
    response: str,
) -> None:
    llm = FakeLLMClient(response=response)
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)
    question = "Compare le licenciement et la rupture conventionnelle."

    assert decomposer.decompose(question) == [question]


def test_llm_error_falls_back_to_original_question(prompt_file: Path) -> None:
    llm = FakeLLMClient(error=RuntimeError("service indisponible"))
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)
    question = "Compare le licenciement et les congés payés."

    assert decomposer.decompose(question) == [question]


def test_empty_question_does_not_call_llm(prompt_file: Path) -> None:
    llm = FakeLLMClient()
    decomposer = QuestionDecomposer(llm, prompt_path=prompt_file)

    assert decomposer.decompose("   ") == []
    assert llm.calls == []


def test_max_sub_questions_must_stay_between_two_and_four(
    prompt_file: Path,
) -> None:
    with pytest.raises(ValueError, match="compris entre 2 et 4"):
        QuestionDecomposer(
            FakeLLMClient(),
            max_sub_questions=5,
            prompt_path=prompt_file,
        )
