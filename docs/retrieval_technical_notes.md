# Notes techniques retrieval

## Corpus et chunking

Le corpus vient de SocialGouv/legi-data, qui fournit un JSON exploitable sans
authentification. La source juridique primaire reste Legifrance. Le JSON brut est
conserve pour la tracabilite, mais seuls les articles en vigueur et rattaches aux
themes du projet sont transformes en chunks.

Le choix principal est: un chunk = un article complet. Cela evite de casser le
sens juridique d'un article et facilite les citations dans les reponses.

Points difficiles:

- la structure du JSON est arborescente et peut evoluer;
- les numeros d'articles peuvent contenir des suffixes, par exemple
  `L1221-10-1`;
- certaines plages d'articles se chevauchent, notamment autour de `L1237-*`;
- les anciennes versions doivent etre exclues pour eviter des reponses obsoletes.

## FAISS et persistance

FAISS ne stocke que les vecteurs. Le texte et les metadonnees sont donc conserves
dans `data/vector_store/chunks_metadata.json`, avec le meme ordre que les
vecteurs FAISS. Un index FAISS `i` correspond a `chunks_metadata[i]`.

La persistance est controlee par:

- `data/vector_store/index.faiss`;
- `data/vector_store/chunks_metadata.json`;
- `data/vector_store/vector_store_metadata.json`.

L'index n'est pas reconstruit si le hash des chunks et le modele d'embedding
n'ont pas change. Si le modele change, l'index est reconstruit pour eviter
d'interroger des vecteurs incompatibles.

## Modele d'embedding

Le modele retenu est:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Justification:

- modele multilingue adapte aux questions en francais;
- plus leger et plus rapide que `paraphrase-multilingual-mpnet-base-v2`;
- suffisant pour un POC RAG maintenable;
- compatible avec une execution locale sans LLM pour la vectorisation.

Le modele plus lourd `paraphrase-multilingual-mpnet-base-v2` reste une option
future si la priorite devient la qualite fine du retrieval plutot que la vitesse.

## BM25

BM25 complete FAISS avec une recherche lexicale explicable. Il est utile pour:

- les numeros d'articles;
- les mots juridiques exacts;
- les questions courtes;
- les cas ou HyDE ou l'embedding rapprochent un texte semantiquement voisin mais
  pas l'article exact demande.

Le moteur BM25 normalise les accents et gere les numeros d'articles avec suffixe.
Un bonus controle priorise les articles explicitement cites par l'utilisateur.

## Fusion hybride

Les scores FAISS et BM25 ne sont pas directement comparables. Ils sont donc
normalises separement avant fusion:

```text
hybrid_score = alpha * vector_score + beta * bm25_score
```

Par defaut:

- `HYBRID_ALPHA=0.7`;
- `HYBRID_BETA=0.3`.

Ces valeurs favorisent la recherche semantique tout en gardant un signal lexical
fort pour les articles exacts et les termes juridiques precis.

## Ameliorations possibles

Avec davantage de temps, les ameliorations prioritaires seraient:

- evaluer quantitativement le retrieval sur un jeu de questions annote;
- ajuster `HYBRID_ALPHA` et `HYBRID_BETA` par theme ou par type de question;
- tester un modele d'embedding plus qualitatif;
- ajouter une verification periodique avec l'API officielle Legifrance/PISTE;
- enrichir les metadonnees avec des liens Legifrance directs;
- ajouter un cache local du modele pour reduire les logs et temps de chargement.

