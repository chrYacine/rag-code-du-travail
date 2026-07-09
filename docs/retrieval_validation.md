# Validation du retrieval

Ce jeu de validation sert a verifier rapidement que les trois modes de recherche
retrouvent des articles pertinents:

- recherche vectorielle FAISS;
- recherche lexicale BM25;
- fusion hybride.

Les tests doivent etre relances apres toute modification du chunking, du modele
d'embedding, de BM25 ou de la formule hybride.

## Cas de validation

| Type | Question | Articles attendus | Objectif |
| --- | --- | --- | --- |
| Article exact | Que dit l'article L1237-11 ? | L1237-11 | Verifier que BM25 et l'hybride priorisent l'article explicitement demande. |
| Article avec suffixe | Que prevoit l'article L1221-10-1 ? | L1221-10-1 | Verifier la gestion des numeros d'articles a suffixe. |
| Theme simple | Quelle est la duree legale du travail ? | L3121-* | Verifier le theme duree du travail et heures supplementaires. |
| Theme simple | Quels sont les droits aux conges payes ? | L3141-* | Verifier le theme conges payes. |
| Theme simple | Quelles sont les regles sur le harcelement moral ? | L1152-* | Verifier le theme harcelement et discrimination. |
| Multi-theme | Compare rupture conventionnelle et licenciement. | L1237-11, L1232-* ou L1233-* | Verifier decomposition, retrieval multi-requetes et deduplication. |
| Remuneration | Que dit le Code du travail sur le SMIC ? | L3231-* | Verifier le theme salaire minimum. |

## Commandes de controle

```bash
python src/retrieval/vector_retriever.py "Que dit l'article L1237-11 ?" --top-k 3
python src/retrieval/bm25_retriever.py "Que dit l'article L1237-11 ?" --top-k 3
python src/retrieval/hybrid_retriever.py "Que dit l'article L1237-11 ?" --top-k 3
```

Resultat attendu pour la requete directe `L1237-11`: l'article `L1237-11`
doit apparaitre en premiere position en BM25 et en hybride.

## Points a verifier dans chaque resultat

- `metadata.article_id` contient le numero lisible, par exemple `L1237-11`.
- `metadata.legi_id` contient l'identifiant technique `LEGIARTI...`.
- `metadata.theme` correspond au theme fonctionnel du RAG.
- `metadata.section` contient le fil d'Ariane.
- `score_details.vector_score`, `score_details.bm25_score` et
  `score_details.hybrid_score` sont presents en mode hybride.

