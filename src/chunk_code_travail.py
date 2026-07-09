from __future__ import annotations

import json
import logging
import re
from html import unescape
from pathlib import Path
from typing import Any, Iterable

try:
    from src.config import (
        CODE_TRAVAIL_ID,
        METADATA_CODE_TRAVAIL_FILE,
        PRIMARY_SOURCE,
        PROCESSED_CHUNKS_FILE,
        RAW_CODE_TRAVAIL_FILE,
        TECHNICAL_SOURCE,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from config import (  # type: ignore
        CODE_TRAVAIL_ID,
        METADATA_CODE_TRAVAIL_FILE,
        PRIMARY_SOURCE,
        PROCESSED_CHUNKS_FILE,
        RAW_CODE_TRAVAIL_FILE,
        TECHNICAL_SOURCE,
    )

logger = logging.getLogger(__name__)


THEME_ARTICLE_RANGES: dict[str, tuple[str, str]] = {
    "rupture de la periode d'essai": ("L1221-19", "L1221-26"),
    "rupture conventionnelle": ("L1237-11", "L1237-19"),
    "licenciement": ("L1231-1", "L1237-20"),
    "harcelement et discrimination": ("L1152-1", "L1155-2"),
    "representation du personnel": ("L2311-1", "L2316-26"),
    "duree du travail et heures supplementaires": ("L3121-1", "L3121-36"),
    "conges payes": ("L3141-1", "L3141-32"),
    "salaire minimum smic": ("L3231-1", "L3232-9"),
    "contrat de travail": ("L1221-1", "L1248-11"),
}


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    text = unescape(value)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_fil_ariane(parts: Iterable[str]) -> str:
    cleaned_parts = [clean_text(part) for part in parts if clean_text(part)]
    return " > ".join(cleaned_parts)


def article_sort_key(article_num: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"([A-Z])(\d+)-(\d+)", article_num.strip())
    if not match:
        return (article_num, -1, -1)
    prefix, book_number, article_number = match.groups()
    return (prefix, int(book_number), int(article_number))


def is_article_between(article_num: str, start: str, end: str) -> bool:
    current_key = article_sort_key(article_num)
    start_key = article_sort_key(start)
    end_key = article_sort_key(end)
    return start_key <= current_key <= end_key


def infer_theme(article_num: str) -> str | None:
    for theme, (start, end) in THEME_ARTICLE_RANGES.items():
        if is_article_between(article_num, start, end):
            return theme
    return None


def extract_article_chunks(
    data: dict[str, Any], retrieved_at: str | None = None
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    def visit(node: Any, path: list[str]) -> None:
        if not isinstance(node, dict):
            return

        node_data = node.get("data") if isinstance(node.get("data"), dict) else {}
        node_type = node.get("type")
        title = node_data.get("title")
        article_num = node_data.get("num")

        next_path = path
        if isinstance(title, str) and title.strip():
            next_path = [*path, title]

        if node_type == "article" and isinstance(article_num, str):
            text = clean_text(node_data.get("texte") or node_data.get("texteHtml"))
            if text:
                legi_id = str(
                    node_data.get("id") or node_data.get("cid") or article_num
                )
                fil_ariane = normalize_fil_ariane(next_path)
                content = f"Article {article_num}\n{text}"
                chunks.append(
                    {
                        "id": legi_id,
                        "article_num": article_num,
                        "article_id": article_num,
                        "legi_id": legi_id,
                        "theme": infer_theme(article_num) or "",
                        "fil_ariane": fil_ariane,
                        "content": content,
                        "source": TECHNICAL_SOURCE,
                        "primary_source": PRIMARY_SOURCE,
                        "code_id": CODE_TRAVAIL_ID,
                        "retrieved_at": retrieved_at or "",
                        "etat": str(node_data.get("etat") or ""),
                    }
                )

        for child in node.get("children") or []:
            visit(child, next_path)

    visit(data, [])
    return chunks


def is_indexable_chunk(chunk: dict[str, Any]) -> bool:
    return chunk.get("etat") == "VIGUEUR" and bool(chunk.get("theme"))


def to_processed_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "article_id": str(chunk.get("article_id") or ""),
        "legi_id": str(chunk.get("legi_id") or chunk.get("id") or ""),
        "section": str(chunk.get("fil_ariane") or ""),
        "theme": str(chunk.get("theme") or ""),
        "etat": str(chunk.get("etat") or ""),
        "source": str(chunk.get("source") or TECHNICAL_SOURCE),
        "primary_source": str(chunk.get("primary_source") or PRIMARY_SOURCE),
        "code_id": str(chunk.get("code_id") or CODE_TRAVAIL_ID),
        "retrieved_at": str(chunk.get("retrieved_at") or ""),
    }

    return {
        "id": metadata["legi_id"],
        "content": str(chunk.get("content") or ""),
        "metadata": metadata,
    }


def build_processed_chunks(
    data: dict[str, Any], retrieved_at: str | None = None
) -> list[dict[str, Any]]:
    raw_chunks = extract_article_chunks(data, retrieved_at=retrieved_at)
    return [
        to_processed_chunk(chunk) for chunk in raw_chunks if is_indexable_chunk(chunk)
    ]


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_retrieved_at(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return ""

    metadata = read_json_file(metadata_path)
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("retrieved_at") or "")


def write_processed_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_chunks(
    *,
    raw_path: Path = RAW_CODE_TRAVAIL_FILE,
    metadata_path: Path = METADATA_CODE_TRAVAIL_FILE,
    output_path: Path = PROCESSED_CHUNKS_FILE,
) -> dict[str, Any]:
    data = read_json_file(raw_path)
    if not isinstance(data, dict):
        raise ValueError("Le fichier brut doit contenir un objet JSON racine.")

    retrieved_at = read_retrieved_at(metadata_path)
    chunks = build_processed_chunks(data, retrieved_at=retrieved_at)
    write_processed_chunks(output_path, chunks)

    themes = sorted({chunk["metadata"]["theme"] for chunk in chunks})
    status = {
        "status": "success",
        "chunks_count": len(chunks),
        "themes_count": len(themes),
        "themes": themes,
        "output_path": str(output_path),
    }
    logger.info(
        "Chunking terminé: %s chunks écrits dans %s.",
        status["chunks_count"],
        output_path,
    )
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    generate_chunks()
