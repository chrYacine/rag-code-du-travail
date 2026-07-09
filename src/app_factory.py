from __future__ import annotations

from src.config import TOP_K
from src.llm.contracts import LLMClient
from src.llm.groq_client import GroqClient
from src.llm.prompt_builder import PromptBuilder
from src.moderation.contracts import UserInputModerator
from src.moderation.prompt_injection_moderator import PromptInjectionModerator
from src.query_processing.hyde_generator import HyDEGenerator
from src.query_processing.question_decomposer import QuestionDecomposer
from src.rag.orchestrator import RAGOrchestrator
from src.retrieval.contracts import RetrievalEngine
from src.retrieval.vector_retriever import VectorRetrievalEngine


def build_rag_service(
    *,
    llm_client: LLMClient | None = None,
    retrieval_engine: RetrievalEngine | None = None,
    input_moderator: UserInputModerator | None = None,
    top_k: int = TOP_K,
) -> RAGOrchestrator:
    """Assemble the application without leaking infrastructure into the UI."""

    if top_k < 1:
        raise ValueError("TOP_K doit être supérieur ou égal à 1.")

    llm = llm_client or GroqClient()
    retriever = retrieval_engine or VectorRetrievalEngine()
    moderator = input_moderator or PromptInjectionModerator()

    return RAGOrchestrator(
        retrieval_engine=retriever,
        llm_client=llm,
        input_moderator=moderator,
        prompt_builder=PromptBuilder(),
        top_k=top_k,
        question_decomposer=QuestionDecomposer(llm),
        hyde_generator=HyDEGenerator(llm),
    )
