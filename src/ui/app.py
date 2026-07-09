from __future__ import annotations

import logging
from typing import Protocol

from src.rag.orchestrator import RAGAnswer
from src.ui.components import answer_without_duplicate_warning, build_source_views

logger = logging.getLogger(__name__)


class RAGService(Protocol):
    def answer(self, question: str) -> RAGAnswer:
        """Return a legal-information answer and its retrieved sources."""


def render_app(service: RAGService, *, configure_page: bool = True) -> None:
    """Render the UI while keeping infrastructure outside this module."""

    import streamlit as st

    if configure_page:
        _configure_page(st)
    st.title("Assistant Code du travail")
    st.caption(
        "Posez une question sur le droit du travail français. "
        "Les réponses sont fondées sur les articles retrouvés dans le corpus."
    )

    question = st.text_area(
        "Votre question",
        placeholder=(
            "Exemple : comparez les règles des congés payés "
            "et des heures supplémentaires."
        ),
        max_chars=4_000,
    )

    if not st.button("Rechercher", type="primary", use_container_width=True):
        return

    if not question.strip():
        st.warning("Saisissez une question avant de lancer la recherche.")
        return

    try:
        with st.spinner("Recherche des articles pertinents..."):
            result = service.answer(question)
    except Exception:
        logger.exception("Échec du traitement de la question dans l'interface.")
        st.error(
            "La demande n'a pas pu être traitée. "
            "Vérifiez la configuration puis réessayez."
        )
        return

    st.subheader("Réponse")
    st.write(answer_without_duplicate_warning(result))
    st.warning(result.legal_warning)

    sources = build_source_views(result.sources)
    st.subheader(f"Sources utilisées ({len(sources)})")
    if not sources:
        st.info("Aucun article pertinent n’a été retrouvé.")
        return

    for source in sources:
        with st.expander(
            f"Article {source.article_id} — score {source.score}",
            expanded=False,
        ):
            if source.theme:
                st.write(f"**Thème :** {source.theme}")
            if source.section:
                st.write(f"**Section :** {source.section}")
            if source.source:
                st.write(f"**Source :** {source.source}")
            if source.legi_id:
                st.write(f"**Identifiant Légifrance :** {source.legi_id}")
            if source.score_details:
                formatted_scores = " · ".join(
                    f"{name}: {value}" for name, value in source.score_details.items()
                )
                st.caption(formatted_scores)
            st.write(source.content)


def main() -> None:
    """Build the cached RAG service and launch the Streamlit application."""

    import streamlit as st

    _configure_page(st)
    try:
        from src.ui.runtime import get_rag_service

        with st.spinner("Chargement du moteur RAG..."):
            service = get_rag_service()
    except Exception:
        logger.exception("Échec de l'initialisation du service RAG.")
        st.error(
            "Le moteur RAG n'a pas pu démarrer. Vérifiez la clé Groq, "
            "les chunks et l'index FAISS."
        )
        return

    render_app(service, configure_page=False)


def _configure_page(st: object) -> None:
    """Apply the Streamlit page configuration before any other UI call."""

    st.set_page_config(
        page_title="Assistant Code du travail",
        page_icon="⚖️",
        layout="centered",
    )


if __name__ == "__main__":
    main()
