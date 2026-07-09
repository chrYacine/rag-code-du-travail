from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Sequence

try:
    from src.app_factory import build_rag_service
    from src.config import TOP_K
    from src.rag.orchestrator import RAGAnswer
    from src.retrieval.contracts import RetrievedChunk
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.app_factory import build_rag_service  # type: ignore
    from src.config import TOP_K  # type: ignore
    from src.rag.orchestrator import RAGAnswer  # type: ignore
    from src.retrieval.contracts import RetrievedChunk  # type: ignore

logger = logging.getLogger(__name__)


def format_score_details(score_details: object) -> str:
    if not isinstance(score_details, dict) or not score_details:
        return "scores indisponibles"

    formatted_scores: list[str] = []
    for key in ("vector_score", "bm25_score", "hybrid_score"):
        value = score_details.get(key)
        if isinstance(value, int | float):
            formatted_scores.append(f"{key}={value:.4f}")

    return " | ".join(formatted_scores) if formatted_scores else "scores indisponibles"


def format_source(index: int, chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    article_id = metadata.get("article_id", "article inconnu")
    legi_id = metadata.get("legi_id", "legi_id inconnu")
    theme = metadata.get("theme", "theme non renseigne")
    section = metadata.get("section", "section non renseignee")
    scores = format_score_details(chunk.score_details)

    return (
        f"{index}. Article {article_id}\n"
        f"   legi_id: {legi_id}\n"
        f"   theme: {theme}\n"
        f"   section: {section}\n"
        f"   score: {chunk.score:.4f}\n"
        f"   details: {scores}"
    )


def format_answer(result: RAGAnswer) -> str:
    lines = [
        "Reponse",
        "=======",
        result.answer.strip(),
        "",
        "Sources",
        "=======",
    ]

    if not result.sources:
        lines.append("Aucune source retrouvee.")
    else:
        for index, source in enumerate(result.sources, start=1):
            lines.append(format_source(index, source))
            lines.append("")

    if result.legal_warning not in result.answer:
        lines.extend(["", result.legal_warning])

    return "\n".join(lines).strip()


def run_query(question: str, *, top_k: int = TOP_K) -> RAGAnswer:
    service = build_rag_service(top_k=top_k)
    return service.answer(question)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interroge le pipeline RAG du Code du travail sans reconstruire "
            "le corpus ni l'index FAISS."
        )
    )
    parser.add_argument("question", help="Question a poser au moteur RAG.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Nombre maximal de sources a utiliser.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not args.question.strip():
        print("Erreur: la question ne peut pas etre vide.")
        return 2

    if args.top_k < 1:
        print("Erreur: --top-k doit etre superieur ou egal a 1.")
        return 2

    try:
        result = run_query(args.question, top_k=args.top_k)
    except Exception:
        logger.exception("Echec de l'interrogation du pipeline RAG.")
        print(
            "Erreur: la question n'a pas pu etre traitee. "
            "Verifiez la cle Groq, les chunks et l'index FAISS."
        )
        return 1

    print(format_answer(result))
    return 0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    warnings.filterwarnings(
        "ignore", message=".*You are sending unauthenticated requests.*"
    )
    for logger_name in (
        "faiss",
        "huggingface_hub",
        "httpx",
        "sentence_transformers",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
