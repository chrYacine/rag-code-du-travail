from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.llm.contracts import LLMClient
from src.llm.prompt_builder import PromptBuilder
from src.moderation.contracts import UserInputModerator
from src.retrieval.contracts import RetrievedChunk, RetrievalEngine


LEGAL_WARNING = (
    "Avertissement juridique : cette reponse est fournie a titre informatif "
    "et ne remplace pas le conseil d'un professionnel du droit."
)


@dataclass(frozen=True)
class RAGAnswer:
    """Final answer returned by the RAG pipeline."""

    answer: str
    sources: Sequence[RetrievedChunk]
    legal_warning: str = LEGAL_WARNING


@dataclass
class RAGOrchestrator:
    """Coordinates moderation, retrieval, prompt building and LLM generation."""

    retrieval_engine: RetrievalEngine
    llm_client: LLMClient
    input_moderator: UserInputModerator
    prompt_builder: PromptBuilder
    top_k: int = 5

    def answer(self, question: str) -> RAGAnswer:
        moderation_result = self.input_moderator.validate(question)
        if not moderation_result.is_allowed:
            return RAGAnswer(
                answer=(
                    "Je ne peux pas traiter cette demande. "
                    f"Raison : {moderation_result.reason}"
                ),
                sources=[],
            )

        chunks = self.retrieval_engine.search(question=question, top_k=self.top_k)
        messages = self.prompt_builder.build_messages(question=question, chunks=chunks)
        generated_answer = self.llm_client.generate(messages)

        return RAGAnswer(
            answer=self._ensure_legal_warning(generated_answer),
            sources=chunks,
        )

    def _ensure_legal_warning(self, answer: str) -> str:
        if LEGAL_WARNING in answer:
            return answer
        return f"{answer}\n\n{LEGAL_WARNING}"
