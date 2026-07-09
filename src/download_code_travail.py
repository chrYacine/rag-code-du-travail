from __future__ import annotations

import argparse
import hashlib
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

STATUS_START = "Vérification de la source Code du travail en cours..."
STATUS_DOWNLOAD_OK = "Téléchargement du fichier distant réussi."
STATUS_INITIAL = "Aucune version locale précédente détectée. Initialisation du corpus."
STATUS_UNCHANGED = (
    "Aucune modification détectée dans la source. Le fichier local, le chunking "
    "et l’indexation ne sont pas relancés."
)
STATUS_UPDATED = (
    "Modification détectée dans la source. Le fichier brut est mis à jour et les "
    "étapes suivantes peuvent être relancées : extraction, chunking, nettoyage, "
    "indexation."
)
STATUS_FORCED = (
    "Mode forcé activé. Le corpus est régénéré même si aucun changement n’a été "
    "détecté."
)
STATUS_NETWORK_ERROR = (
    "Erreur réseau lors du téléchargement du Code du travail. La version locale "
    "existante est conservée."
)
STATUS_HTTP_ERROR = (
    "Erreur HTTP lors de l’accès à la source distante. La mise à jour est interrompue."
)
STATUS_JSON_ERROR = (
    "Le fichier téléchargé n’est pas un JSON valide. La version locale existante "
    "est conservée."
)
STATUS_SUCCESS = "Vérification terminée avec succès."


class CodeTravailDownloadError(RuntimeError):
    """Raised when the Code du travail corpus cannot be validated."""


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Métadonnées locales illisibles: %s", path)
        return None

    if not isinstance(loaded, dict):
        return None
    return loaded


def has_source_changed(new_hash: str, metadata_path: Path) -> bool:
    metadata = read_json_file(metadata_path)
    if not metadata:
        return True

    previous_hash = metadata.get("content_hash_sha256")
    return previous_hash != new_hash


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
    content_hash_sha256: str,
    has_changed: bool,
    forced: bool,
) -> dict[str, Any]:
    return {
        "source": TECHNICAL_SOURCE,
        "technical_source": TECHNICAL_SOURCE,
        "primary_source": PRIMARY_SOURCE,
        "code_id": CODE_TRAVAIL_ID,
        "url": CODE_TRAVAIL_URL,
        "source_url": CODE_TRAVAIL_URL,
        "retrieved_at": retrieved_at,
        "content_hash_sha256": content_hash_sha256,
        "has_changed": has_changed,
        "forced": forced,
        "raw_file": str(raw_file),
        "content_size_bytes": content_size_bytes,
    }


def build_status(
    *,
    status: str,
    downloaded: bool,
    changed: bool | None,
    forced: bool,
    message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "downloaded": downloaded,
        "changed": changed,
        "forced": forced,
        "message": message,
    }


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_remote_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise CodeTravailDownloadError(STATUS_JSON_ERROR) from exc

    validate_code_travail_data(data)
    return data


def download_code_du_travail(force: bool = False) -> dict[str, Any]:
    logger.info(STATUS_START)

    try:
        response = requests.get(CODE_TRAVAIL_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(STATUS_NETWORK_ERROR)
        return build_status(
            status="error",
            downloaded=False,
            changed=None,
            forced=force,
            message=STATUS_NETWORK_ERROR,
        )
    except requests.exceptions.HTTPError:
        logger.error(STATUS_HTTP_ERROR)
        return build_status(
            status="error",
            downloaded=False,
            changed=None,
            forced=force,
            message=STATUS_HTTP_ERROR,
        )
    except requests.exceptions.RequestException:
        logger.error(STATUS_NETWORK_ERROR)
        return build_status(
            status="error",
            downloaded=False,
            changed=None,
            forced=force,
            message=STATUS_NETWORK_ERROR,
        )

    logger.info(STATUS_DOWNLOAD_OK)

    try:
        data = parse_remote_json(response)
    except CodeTravailDownloadError:
        logger.error(STATUS_JSON_ERROR)
        return build_status(
            status="error",
            downloaded=False,
            changed=None,
            forced=force,
            message=STATUS_JSON_ERROR,
        )

    content_hash = compute_sha256(response.content)
    local_version_exists = (
        METADATA_CODE_TRAVAIL_FILE.exists() and RAW_CODE_TRAVAIL_FILE.exists()
    )
    changed = not local_version_exists or has_source_changed(
        content_hash, METADATA_CODE_TRAVAIL_FILE
    )

    if not local_version_exists:
        logger.info(STATUS_INITIAL)

    if force:
        logger.info(STATUS_FORCED)
    elif not changed:
        logger.info(STATUS_UNCHANGED)
        logger.info(STATUS_SUCCESS)
        return build_status(
            status="unchanged",
            downloaded=True,
            changed=False,
            forced=False,
            message=STATUS_UNCHANGED,
        )
    else:
        logger.info(STATUS_UPDATED)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    write_json_file(RAW_CODE_TRAVAIL_FILE, data)

    metadata = build_metadata(
        retrieved_at=retrieved_at,
        raw_file=RAW_CODE_TRAVAIL_FILE,
        content_size_bytes=len(response.content),
        content_hash_sha256=content_hash,
        has_changed=changed,
        forced=force,
    )
    write_json_file(METADATA_CODE_TRAVAIL_FILE, metadata)

    logger.info("Fichier brut sauvegardé: %s", RAW_CODE_TRAVAIL_FILE)
    logger.info("Métadonnées sauvegardées: %s", METADATA_CODE_TRAVAIL_FILE)
    logger.info(STATUS_SUCCESS)

    if force:
        return build_status(
            status="forced",
            downloaded=True,
            changed=changed,
            forced=True,
            message=STATUS_FORCED,
        )

    return build_status(
        status="updated",
        downloaded=True,
        changed=True,
        forced=False,
        message=STATUS_UPDATED,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vérifie et télécharge le corpus Code du travail."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force la sauvegarde même si la source distante n'a pas changé.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    download_code_du_travail(force=args.force)
