from __future__ import annotations

import streamlit as st

from src.app_factory import build_rag_service
from src.rag.orchestrator import RAGOrchestrator


@st.cache_resource(show_spinner=False)
def get_rag_service() -> RAGOrchestrator:
    """Build heavy models and indexes once per Streamlit process."""

    return build_rag_service()
