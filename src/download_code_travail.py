from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from src.config import (
        CODE_TRAVAIL_ID,
        CODE_TRAVAIL_URL,
        METADATA_CODE_TRAVAIL_FILE,
        PRIMARY_SOURCE,
        RAW_CODE_TRAVAIL_FILE,
        REQUEST_TIMEOUT,
        TECHNICAL_SOURCE,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from config import (  # type: ignore
        CODE_TRAVAIL_ID,
        CODE_TRAVAIL_URL,
        METADATA_CODE_TRAVAIL_FILE,
        PRIMARY_SOURCE,
        RAW_CODE_TRAVAIL_FILE,
        REQUEST_TIMEOUT,
        TECHNICAL_SOURCE,
    )

logger = logging.getLogger(__name__)


class CodeTravailDownloadError(RuntimeError):
    """Raised when the Code du travail corpus cannot be downloaded or parsed."""


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def validate_code_travail_data(data: Any) -> None:
    if not isinstance(data, dict):
        raise CodeTravailDownloadError("Le fichier JSON doit contenir un objet racine.")

    children = data.get("children")
    if not isinstance(children, list) or not children:
        raise CodeTravailDownloadError(
            "Le fichier JSON ne contient pas d'arborescence exploitable."
        )

    if data.get("type") != "code":
        logger.warning("Type racine inattendu dans le JSON: %s", data.get("type"))


def build_metadata(
    *,
    retrieved_at: str,
    raw_file: Path,
    content_size_bytes: int,
) -> dict[str, Any]:
    return {
        "retrieved_at": retrieved_at,
        "source_url": CODE_TRAVAIL_URL,
        "technical_source": TECHNICAL_SOURCE,
        "primary_source": PRIMARY_SOURCE,
        "code_id": CODE_TRAVAIL_ID,
        "raw_file": str(raw_file),
        "content_size_bytes": content_size_bytes,
    }


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_code_du_travail() -> dict[str, Any]:
    logger.info("Telechargement du Code du travail depuis %s", TECHNICAL_SOURCE)

    try:
        response = requests.get(CODE_TRAVAIL_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise CodeTravailDownloadError("Timeout pendant le telechargement.") from exc
    except requests.exceptions.HTTPError as exc:
        raise CodeTravailDownloadError(
            f"Erreur HTTP pendant le telechargement: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise CodeTravailDownloadError(
            f"Erreur reseau pendant le telechargement: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise CodeTravailDownloadError(
            "La reponse recue n'est pas un JSON valide."
        ) from exc

    validate_code_travail_data(data)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    write_json_file(RAW_CODE_TRAVAIL_FILE, data)

    metadata = build_metadata(
        retrieved_at=retrieved_at,
        raw_file=RAW_CODE_TRAVAIL_FILE,
        content_size_bytes=len(response.content),
    )
    write_json_file(METADATA_CODE_TRAVAIL_FILE, metadata)

    logger.info("Fichier brut sauvegarde: %s", RAW_CODE_TRAVAIL_FILE)
    logger.info("Metadonnees sauvegardees: %s", METADATA_CODE_TRAVAIL_FILE)

    return data


if __name__ == "__main__":
    configure_logging()
    download_code_du_travail()
