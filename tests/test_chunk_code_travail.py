from __future__ import annotations

from src.chunk_code_travail import clean_text, extract_article_chunks, infer_theme


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
    assert chunk["article_id"] == "LEGIARTI000006902580"
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
    assert infer_theme("L3141-10") == "conges payes"
    assert infer_theme("L1232-1") == "licenciement"
    assert infer_theme("L1237-11") == "rupture conventionnelle"
    assert infer_theme("L1152-1") == "harcelement et discrimination"
    assert infer_theme("L9999-1") is None
