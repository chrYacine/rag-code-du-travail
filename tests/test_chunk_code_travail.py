from __future__ import annotations

import json

from src.chunk_code_travail import (
    build_processed_chunks,
    clean_text,
    extract_article_chunks,
    generate_chunks,
    infer_theme,
    to_processed_chunk,
)


def test_extract_article_chunks_returns_expected_structure() -> None:
    data = {
        "type": "code",
        "data": {"title": "Code du travail"},
        "children": [
            {
                "type": "section",
                "data": {"title": "Durée du travail"},
                "children": [
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI000006902580",
                            "cid": "LEGIARTI000006902580",
                            "num": "L3121-1",
                            "texte": "<p>La durée du travail effectif est le temps...</p>",
                            "etat": "VIGUEUR",
                        },
                    }
                ],
            }
        ],
    }

    chunks = extract_article_chunks(data, retrieved_at="2026-07-09T10:00:00+00:00")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["id"] == "LEGIARTI000006902580"
    assert chunk["article_num"] == "L3121-1"
    assert chunk["article_id"] == "L3121-1"
    assert chunk["legi_id"] == "LEGIARTI000006902580"
    assert chunk["theme"] == "duree du travail et heures supplementaires"
    assert chunk["fil_ariane"] == "Code du travail > Durée du travail"
    assert chunk["content"].startswith("Article L3121-1")
    assert chunk["source"] == "SocialGouv/legi-data"
    assert chunk["primary_source"] == "Légifrance"
    assert chunk["code_id"] == "LEGITEXT000006072050"
    assert chunk["retrieved_at"] == "2026-07-09T10:00:00+00:00"
    assert chunk["etat"] == "VIGUEUR"


def test_clean_text_removes_basic_html() -> None:
    assert (
        clean_text("<p>Texte&nbsp;important<br/>Suite</p>") == "Texte important\nSuite"
    )


def test_infer_theme_uses_article_ranges() -> None:
    assert infer_theme("L1152-1") == "harcelement et discrimination"
    assert infer_theme("L1221-1") == "contrat de travail"
    assert infer_theme("L1221-19") == "rupture de la periode d'essai"
    assert infer_theme("L1221-26") == "rupture de la periode d'essai"
    assert infer_theme("L1232-1") == "licenciement"
    assert infer_theme("L1237-10") == "licenciement"
    assert infer_theme("L1237-11") == "rupture conventionnelle"
    assert infer_theme("L1237-19") == "rupture conventionnelle"
    assert infer_theme("L1237-20") == "licenciement"
    assert infer_theme("L2311-1") == "representation du personnel"
    assert infer_theme("L3121-10") == "duree du travail et heures supplementaires"
    assert infer_theme("L3141-10") == "conges payes"
    assert infer_theme("L3231-1") == "salaire minimum smic"
    assert infer_theme("L9999-1") is None


def test_to_processed_chunk_uses_retrieved_chunk_metadata_contract() -> None:
    chunk = {
        "id": "LEGIARTI000006902580",
        "article_id": "L3121-1",
        "legi_id": "LEGIARTI000006902580",
        "fil_ariane": "Code du travail > Durée du travail",
        "theme": "duree du travail et heures supplementaires",
        "content": "Article L3121-1\nTexte",
        "source": "SocialGouv/legi-data",
        "primary_source": "Légifrance",
        "code_id": "LEGITEXT000006072050",
        "retrieved_at": "2026-07-09T10:00:00+00:00",
        "etat": "VIGUEUR",
    }

    processed = to_processed_chunk(chunk)

    assert processed["id"] == "LEGIARTI000006902580"
    assert processed["content"] == "Article L3121-1\nTexte"
    assert processed["metadata"]["article_id"] == "L3121-1"
    assert processed["metadata"]["legi_id"] == "LEGIARTI000006902580"
    assert processed["metadata"]["section"] == "Code du travail > Durée du travail"
    assert all(isinstance(value, str) for value in processed["metadata"].values())


def test_build_processed_chunks_keeps_only_vigueur_target_articles() -> None:
    data = {
        "type": "code",
        "data": {"title": "Code du travail"},
        "children": [
            {
                "type": "section",
                "data": {"title": "Section test"},
                "children": [
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI1",
                            "num": "L3121-1",
                            "texte": "Texte en vigueur",
                            "etat": "VIGUEUR",
                        },
                    },
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI2",
                            "num": "L3121-2",
                            "texte": "Texte modifié",
                            "etat": "MODIFIE",
                        },
                    },
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI3",
                            "num": "L9999-1",
                            "texte": "Texte hors thème",
                            "etat": "VIGUEUR",
                        },
                    },
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI4",
                            "num": "L1237-11",
                            "texte": "Texte rupture conventionnelle",
                            "etat": "VIGUEUR",
                        },
                    },
                ],
            }
        ],
    }

    chunks = build_processed_chunks(data, retrieved_at="2026-07-09T10:00:00+00:00")

    assert [chunk["metadata"]["article_id"] for chunk in chunks] == [
        "L3121-1",
        "L1237-11",
    ]
    assert chunks[1]["metadata"]["theme"] == "rupture conventionnelle"


def test_generate_chunks_writes_processed_file(tmp_path) -> None:
    raw_path = tmp_path / "raw" / "code_du_travail.json"
    metadata_path = tmp_path / "metadata" / "code_du_travail_metadata.json"
    output_path = tmp_path / "processed" / "chunks_code_du_travail.json"
    raw_path.parent.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)

    raw_path.write_text(
        json.dumps(
            {
                "type": "code",
                "data": {"title": "Code du travail"},
                "children": [
                    {
                        "type": "article",
                        "data": {
                            "id": "LEGIARTI1",
                            "num": "L3141-1",
                            "texte": "Texte congés payés",
                            "etat": "VIGUEUR",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"retrieved_at": "2026-07-09T10:00:00+00:00"}),
        encoding="utf-8",
    )

    status = generate_chunks(
        raw_path=raw_path,
        metadata_path=metadata_path,
        output_path=output_path,
    )

    written_chunks = json.loads(output_path.read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert status["chunks_count"] == 1
    assert written_chunks[0]["metadata"]["article_id"] == "L3141-1"
    assert written_chunks[0]["metadata"]["retrieved_at"] == "2026-07-09T10:00:00+00:00"
