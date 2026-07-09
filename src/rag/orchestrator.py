from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.llm.contracts import LLMClient
from src.llm.prompt_builder import PromptBuilder
from src.moderation.contracts import UserInputModerator
from src.query_processing.hyde_generator import HyDEGenerator
from src.query_processing.question_decomposer import QuestionDecomposer
from src.retrieval.contracts import RetrievedChunk, RetrievalEngine
from src.retrieval.result_aggregator import ResultAggregator

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
    question_decomposer: QuestionDecomposer | None = None
    hyde_generator: HyDEGenerator | None = None
    result_aggregator: ResultAggregator = field(default_factory=ResultAggregator)

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

        chunks = self._retrieve(question)
        messages = self.prompt_builder.build_messages(question=question, chunks=chunks)
        generated_answer = self.llm_client.generate(messages)

        return RAGAnswer(
            answer=self._ensure_legal_warning(generated_answer),
            sources=chunks,
        )

    def _retrieve(self, question: str) -> Sequence[RetrievedChunk]:
        sub_questions = (
            self.question_decomposer.decompose(question)
            if self.question_decomposer
            else [question]
        )
        if not sub_questions:
            sub_questions = [question]

        result_sets: list[Sequence[RetrievedChunk]] = []
        for sub_question in sub_questions:
            queries = self._build_retrieval_queries(sub_question)
            for query in queries:
                result_sets.append(
                    self.retrieval_engine.search(
                        question=query,
                        top_k=self.top_k,
                    )
                )

        return self.result_aggregator.aggregate(result_sets, top_k=self.top_k)

    def _build_retrieval_queries(self, question: str) -> list[str]:
        if not self.hyde_generator:
            return [question]

        expanded = self.hyde_generator.expand(question)
        queries = [expanded.original_query]
        if (
            expanded.vector_query
            and expanded.vector_query.casefold() != expanded.original_query.casefold()
        ):
            queries.append(expanded.vector_query)
        return queries

    def _ensure_legal_warning(self, answer: str) -> str:
        if LEGAL_WARNING in answer:
            return answer
        return f"{answer}\n\n{LEGAL_WARNING}"
