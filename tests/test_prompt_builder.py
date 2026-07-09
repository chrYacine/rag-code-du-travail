from __future__ import annotations

from src.llm.prompt_builder import SYSTEM_PROMPT, PromptBuilder
from src.retrieval.contracts import RetrievedChunk


def test_system_prompt_covers_legal_safety_requirements() -> None:
    assert "N'invente jamais un article" in SYSTEM_PROMPT
    assert "convention collective" in SYSTEM_PROMPT
    assert "licenciement est abusif" in SYSTEM_PROMPT
    assert "professionnel du droit" in SYSTEM_PROMPT
    assert "textes peuvent avoir évolué" in SYSTEM_PROMPT


def test_context_contains_traceability_metadata() -> None:
    chunk = RetrievedChunk(
        content="Article L3121-1\nTexte juridique.",
        score=0.91,
        metadata={
            "article_id": "L3121-1",
            "theme": "duree du travail",
            "section": "Code du travail > Durée du travail",
            "retrieved_at": "2026-07-09T10:00:00+00:00",
        },
    )

    messages = PromptBuilder().build_messages(
        question="Quelle est la durée légale ?",
        chunks=[chunk],
    )
    user_prompt = messages[1]["content"]

    assert "Article : L3121-1" in user_prompt
    assert "Thème : duree du travail" in user_prompt
    assert "Section : Code du travail > Durée du travail" in user_prompt
    assert "Date de récupération du corpus : 2026-07-09" in user_prompt
    assert "Texte : Article L3121-1" in user_prompt


def test_empty_context_explicitly_reports_missing_sources() -> None:
    messages = PromptBuilder().build_messages(
        question="Question hors corpus",
        chunks=[],
    )

    assert "Aucun contexte juridique pertinent" in messages[1]["content"]
