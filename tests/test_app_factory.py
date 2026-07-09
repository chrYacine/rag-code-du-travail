from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pytest

import src.app_factory as app_factory
from src.app_factory import build_rag_service
from src.moderation.contracts import ModerationResult
from src.retrieval.contracts import RetrievedChunk


@dataclass
class FakeLLMClient:
    response: str = "Réponse."

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self.response


class FakeRetrievalEngine:
    def search(self, question: str, top_k: int) -> Sequence[RetrievedChunk]:
        return []


class FakeModerator:
    def validate(self, text: str) -> ModerationResult:
        return ModerationResult(is_allowed=True)


def test_build_rag_service_wires_all_application_components() -> None:
    llm = FakeLLMClient()
    retriever = FakeRetrievalEngine()
    moderator = FakeModerator()

    service = build_rag_service(
        llm_client=llm,
        retrieval_engine=retriever,
        input_moderator=moderator,
        top_k=3,
    )

    assert service.llm_client is llm
    assert service.retrieval_engine is retriever
    assert service.input_moderator is moderator
    assert service.top_k == 3
    assert service.question_decomposer is not None
    assert service.question_decomposer.llm_client is llm
    assert service.hyde_generator is not None
    assert service.hyde_generator.llm_client is llm


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="supérieur ou égal à 1"):
        build_rag_service(
            llm_client=FakeLLMClient(),
            retrieval_engine=FakeRetrievalEngine(),
            input_moderator=FakeModerator(),
            top_k=0,
        )


def test_default_retrieval_engine_is_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetrievalEngine()
    monkeypatch.setattr(
        app_factory,
        "HybridRetrievalEngine",
        lambda: retriever,
    )

    service = build_rag_service(
        llm_client=FakeLLMClient(),
        input_moderator=FakeModerator(),
    )

    assert service.retrieval_engine is retriever
