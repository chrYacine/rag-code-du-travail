# rag-code-du-travail

Projet final Master 2 MD5 - Assistant juridique RAG sur le Code du travail.

## Objectif

Construire un assistant capable de repondre a des questions sur le Code du travail a partir d'un pipeline RAG implemente manuellement.

## Contraintes principales

- Pas de LangChain ni LlamaIndex.
- Separation stricte entre indexation et interrogation.
- Base vectorielle persistante.
- Citations des articles utilises.
- Avertissement juridique dans les reponses.
- Interface utilisateur separee du moteur RAG.
- Travail Git via `main`, `dev`, branches `feature/*` et Pull Requests.

## Structure

```text
src/
  indexing/
  retrieval/
  rag/
  moderation/
  llm/
  ui/
  config/
tests/
data/
  raw/
  processed/
  vector_store/
docs/
prompts/
```

## Workflow Git

```bash
git checkout dev
git pull origin dev
git checkout -b feature/nom-de-la-tache
```

Avant une Pull Request :

```bash
python -m pytest
python -m ruff check .
python -m black --check .
```

## Réponses aux questions de réflexion

### 1. Granularité du chunking

Indexer un article séparément préserve son unité juridique, permet une citation précise et évite
qu'un regroupement trop long dilue le vocabulaire utile à l'embedding. Cette stratégie facilite
aussi la déduplication par `article_id`. Son inconvénient est qu'un article peut dépendre de
définitions ou d'exceptions situées dans des articles voisins.

Regrouper une section fournit davantage de contexte et aide à comprendre les renvois, mais produit
des chunks plus longs, moins ciblés et difficiles à citer sans ambiguïté. Cela augmente aussi le
risque de transmettre au LLM des dispositions non pertinentes.

Le projet retient donc **un chunk par article complet**, sans coupure interne. Le fil d'Ariane de la
section est conservé dans les métadonnées. Une approche hybride reste envisageable : retrouver les
articles individuellement, puis enrichir le contexte avec les articles voisins ou les références
explicitement citées. L'amélioration actuelle combine plusieurs signaux de recherche (FAISS, BM25,
décomposition et HyDE) tout en gardant l'article comme unité documentaire.

### 2. Traçabilité des articles

Le numéro lisible apparaît à deux endroits :

- dans `content`, sous la forme `Article L3121-1`, donc dans le texte embeddé ;
- dans `metadata["article_id"]`, utilisé pour la déduplication, les citations et l'affichage.

L'identifiant technique Légifrance est conservé dans `metadata["legi_id"]`. Le prompt transmet
explicitement l'identifiant de chaque source et interdit de citer un numéro absent du contexte. Une
citation générée reste néanmoins une sortie probabiliste : l'interface affiche les chunks sources
réels afin que l'utilisateur puisse vérifier la correspondance. Pour un usage juridique, la
disposition doit toujours être contrôlée sur Légifrance.

### 3. Fraîcheur et risque d'obsolescence

Le téléchargement enregistre `retrieved_at` et un hash SHA256 du corpus. Le workflow quotidien
compare ce hash à la source distante et évite une réindexation lorsqu'elle est inchangée. La date de
récupération est propagée dans les métadonnées des chunks, le contexte du LLM et l'affichage des
sources.

Cette date prouve quand la copie technique a été récupérée, pas qu'elle constitue une consolidation
juridique officielle à cet instant. Le système indique donc que les textes peuvent avoir évolué et
documente que SocialGouv/legi-data est une source technique, tandis que Légifrance reste la source
juridique à vérifier. Une version de production devrait utiliser l'API PISTE et surveiller les
dates d'entrée en vigueur.

### 4. Réponses conditionnelles

Le prompt demande de ne pas masquer les conditions dépendant de l'effectif, de la convention
collective, de l'ancienneté ou de la situation du salarié. Lorsque le corpus permet une règle
générale, l'assistant la présente avec des réserves explicites. Si une information manquante change
la réponse, il demande cette précision au lieu de choisir arbitrairement un cas.

Les conventions collectives ne font pas partie de ce corpus. L'assistant doit le signaler et ne
doit pas présenter la règle générale du Code comme nécessairement plus favorable ou exhaustive.

### 5. Frontière du conseil juridique

Une question directement descriptive, telle que la durée légale ou le contenu d'un article, peut
recevoir une synthèse fondée sur les chunks retrouvés. Une demande comme « mon licenciement est-il
abusif ? » exige en revanche des faits, des preuves, une procédure et parfois de la jurisprudence
absents du corpus.

Dans ce second cas, le prompt interdit de rendre un verdict individuel. L'assistant expose les
critères généraux présents dans le Code, précise les informations manquantes et recommande une
vérification par un avocat, un syndicat ou un service compétent. L'orchestrateur ajoute
systématiquement un avertissement indiquant que la réponse est informative et ne remplace pas un
professionnel du droit.

## Etape corpus: recuperation du Code du travail

Cette etape correspond uniquement a la recuperation et a la preparation initiale du corpus. Elle ne couvre pas encore tout le pipeline RAG. Elle prepare la suite: chunking par article, embeddings, indexation vectorielle, evaluation et mise a jour.

### Source retenue

La source juridique primaire reste Legifrance. Pour demarrer le POC sans authentification OAuth, le projet utilise le fichier JSON maintenu par SocialGouv/legi-data:

```text
https://raw.githubusercontent.com/SocialGouv/legi-data/master/data/LEGITEXT000006072050.json
```

Cette source technique permet de recuperer rapidement une version structuree du Code du travail. Elle evite, pour cette premiere brique, la complexite de l'API officielle Legifrance/PISTE: compte PISTE, application, `client_id`, `client_secret`, token OAuth et renouvellement du jeton.

### Limites

- SocialGouv/legi-data n'est pas la source juridique primaire.
- La source primaire reste Legifrance.
- La structure du JSON peut evoluer.
- La disponibilite depend de GitHub.
- Pour un usage juridique critique, une verification avec Legifrance ou une migration future vers PISTE devra etre prevue.

### Bonnes pratiques appliquees

- Timeout reseau.
- Controle HTTP avec `raise_for_status()`.
- Validation JSON minimale.
- Sauvegarde locale separee entre donnees brutes et metadonnees.
- Logs simples, sans secret.
- Aucun scraping de Legifrance.
- Aucun secret et aucune cle API pour cette etape.
- Non-versionnement des gros fichiers JSON generes.
- Base compatible avec une automatisation future via cron ou GitHub Actions.

### Fichiers generes

```text
data/raw/code_du_travail.json
data/metadata/code_du_travail_metadata.json
```

Un futur fichier `data/processed/chunks_code_du_travail.json` pourra etre ajoute lors de l'etape de chunking final.

### Installation et execution

```bash
pip install -r requirements.txt
python src/download_code_travail.py
```

Apres execution, verifier que les fichiers JSON attendus existent dans `data/raw/` et `data/metadata/`.

### Mise a jour hybride du corpus

La recuperation du corpus fonctionne en mode hybride: elle peut etre lancee manuellement par un membre de l'equipe ou automatiquement via GitHub Actions. Dans les deux cas, le script commence par verifier si la source distante a change avant de preparer la suite du pipeline.

Le choix technique retenu est un hash SHA256 calcule sur le contenu distant telecharge. Ce hash est compare au dernier hash conserve dans `data/metadata/code_du_travail_metadata.json`. Cette methode est simple, robuste et reproductible: elle evite de relancer inutilement l'extraction, le chunking, le nettoyage et l'indexation quand le fichier source est identique.

Methodologie appliquee:

- telecharger la source distante avec un timeout reseau;
- valider le statut HTTP et le JSON;
- calculer le hash SHA256 du contenu distant;
- comparer ce hash avec les metadonnees locales;
- ne pas reecrire le fichier brut si aucune modification n'est detectee;
- mettre a jour le fichier brut et les metadonnees si la source change;
- retourner un statut exploitable par les futurs modules du pipeline;
- conserver la possibilite de forcer la regeneration avec `--force`.

Le mode force sert aux cas ou l'equipe veut regenerer volontairement le corpus local, par exemple apres une modification du code de preparation, un doute sur l'etat local ou un test d'integration. Il ne doit pas etre le mode normal d'execution automatisee.

Le workflow `.github/workflows/update_code_travail.yml` lance une verification automatique tous les matins avec le cron `0 5 * * *`. Les cron GitHub Actions sont en UTC: 5h UTC correspond environ a 7h en France en heure d'ete, et l'heure locale peut varier avec le passage heure d'ete / heure d'hiver. Le workflow peut aussi etre lance manuellement depuis l'onglet Actions de GitHub grace a `workflow_dispatch`.

Comme les gros fichiers JSON ne sont pas versionnes, le workflow utilise un cache GitHub Actions pour restaurer les dernieres metadonnees et permettre la comparaison de hash entre deux executions. Ce cache sert uniquement a eviter des traitements inutiles dans l'automatisation; la source de reference reste le JSON distant SocialGouv/legi-data et, juridiquement, Legifrance.

### Messages de statut

Chaque execution affiche un etat clair dans le terminal. Les messages couvrent les cas importants pour l'equipe:

- initialisation du corpus quand aucune version locale precedente n'est detectee;
- telechargement distant reussi;
- aucune modification detectee;
- mise a jour detectee;
- mode force active;
- erreur reseau, HTTP ou JSON;
- fin de traitement avec succes.

Cette logique permet de savoir rapidement si le corpus est a jour, si une action manuelle est necessaire, ou si le pipeline peut continuer vers le chunking et l'indexation.

Le script retourne aussi un dictionnaire de statut exploitable par les futures briques Python. Exemples de statuts: `updated`, `unchanged`, `forced` ou `error`.

### Commandes utiles

```bash
python src/download_code_travail.py
python src/download_code_travail.py --force
python src/chunk_code_travail.py
```

La premiere commande verifie la source et ne met a jour le corpus local que si le hash distant a change. La seconde force la regeneration locale, meme si aucun changement n'est detecte.

### Chunking par article

Le chunking cible le format attendu par le moteur de retrieval et par le contrat `RetrievedChunk`. La strategie retenue est volontairement simple et explicable en soutenance: un chunk correspond a un article complet. Le code ne coupe pas les articles en fragments internes, afin de conserver le sens juridique et les citations.

Le fichier genere est:

```text
data/processed/chunks_code_du_travail.json
```

Chaque entree contient:

- `id`: identifiant technique Legifrance `LEGIARTI...`;
- `content`: texte complet sous la forme `Article Lxxxx-x\n...`;
- `metadata.article_id`: numero lisible de l'article, utilise pour les citations;
- `metadata.legi_id`: identifiant technique;
- `metadata.section`: fil d'Ariane;
- `metadata.theme`: theme fonctionnel du RAG;
- `metadata.etat`: statut juridique, filtre sur `VIGUEUR`;
- `metadata.source`, `metadata.primary_source`, `metadata.code_id`, `metadata.retrieved_at`.

Seuls les articles en vigueur et appartenant aux themes retenus sont exportes vers `data/processed/`. Le JSON brut reste conserve dans `data/raw/` pour la tracabilite, mais il n'est pas indexe directement.

### Base vectorielle FAISS

La vectorisation n'utilise pas de LLM. Elle transforme les chunks en embeddings avec `sentence-transformers`, puis persiste ces vecteurs dans FAISS. Le modele par defaut est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, choisi car il est multilingue, adapte au francais, plus leger que `paraphrase-multilingual-mpnet-base-v2`, et suffisant pour une premiere base RAG maintenable. Le modele reste configurable avec `EMBEDDING_MODEL`.

FAISS ne stocke ni le texte ni les metadonnees. Pour eviter toute ambiguite, le projet sauvegarde donc une liste parallele dans:

```text
data/vector_store/chunks_metadata.json
```

L'ordre de cette liste correspond exactement a l'ordre des vecteurs dans l'index FAISS. Un resultat FAISS `i` permet donc de retrouver `chunks_metadata[i]`, puis de reconstruire un futur `RetrievedChunk`.

Fichiers generes:

```text
data/vector_store/index.faiss
data/vector_store/chunks_metadata.json
data/vector_store/vector_store_metadata.json
```

La mise a jour est controlee par un hash SHA256 de `data/processed/chunks_code_du_travail.json` et par le nom du modele d'embedding enregistre dans `data/vector_store/vector_store_metadata.json`. Si les chunks et le modele n'ont pas change et que les fichiers FAISS existent deja, la base vectorielle n'est pas reconstruite. Si le modele change, l'index est reconstruit automatiquement pour eviter d'interroger des vecteurs produits avec un autre modele. Le mode `--force` permet de reconstruire volontairement la base, par exemple apres un doute sur l'etat local.

```bash
python src/retrieval/vector_store.py
python src/retrieval/vector_store.py --force
python src/retrieval/vector_store.py --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

### Retrieval vectoriel

Le moteur `VectorRetrievalEngine` charge l'index FAISS, la liste parallele `chunks_metadata.json` et les metadonnees du vector store. Il utilise toujours le modele d'embedding enregistre avec l'index, afin de garantir que la question est encodee avec le meme modele que les chunks.

Le contrat expose reste celui attendu par l'orchestration:

```python
retrieval_engine.search(question: str, top_k: int)
```

Chaque resultat est un `RetrievedChunk` avec:

- `content`: texte complet du chunk;
- `score`: score vectoriel FAISS utilise pour le classement;
- `metadata["article_id"]`: numero lisible de l'article, par exemple `L3121-1`;
- `metadata["legi_id"]`: identifiant technique `LEGIARTI...`;
- `score_details["vector_score"]`: detail du score vectoriel.

Commande de verification locale:

```bash
python src/retrieval/vector_retriever.py "Quelle est la duree legale du travail ?" --top-k 3
```

### Retrieval BM25

Le moteur `BM25RetrievalEngine` interroge les chunks textuels avec BM25 Okapi, sans modele LLM et sans dependance externe. Il est prevu pour recevoir la question originale de l'utilisateur ou la sous-question originale produite par l'orchestrateur, tandis que FAISS peut recevoir la variante HyDE.

Le moteur lit par defaut:

```text
data/processed/chunks_code_du_travail.json
```

Chaque resultat est un `RetrievedChunk` avec:

- `content`: texte complet du chunk;
- `score`: score BM25 utilise pour le classement;
- `metadata["article_id"]`: numero lisible de l'article;
- `metadata["legi_id"]`: identifiant technique `LEGIARTI...`;
- `score_details["bm25_score"]`: detail du score BM25.

Les recherches contenant un numero d'article, par exemple `L1237-11`, recoivent un bonus controle pour faire remonter l'article exact. Cela aide les questions directes du type "Que dit l'article L1237-11 ?".

Commande de verification locale:

```bash
python src/retrieval/bm25_retriever.py "Que dit l'article L1237-11 ?" --top-k 3
```

### Retrieval hybride

Le moteur `HybridRetrievalEngine` combine les resultats de `VectorRetrievalEngine` et `BM25RetrievalEngine`. Les scores FAISS et BM25 sont normalises avant fusion, puis le score final est calcule avec:

```text
hybrid_score = alpha * vector_score + beta * bm25_score
```

Les valeurs par defaut sont configurees par `HYBRID_ALPHA` et `HYBRID_BETA`. Le contrat principal reste compatible avec l'orchestration actuelle:

```python
retrieval_engine.search(question: str, top_k: int)
```

Pour l'integration finale avec HyDE, le moteur expose aussi:

```python
retrieval_engine.search_with_queries(
    original_query="question originale",
    vector_query="passage HyDE ou question vectorielle",
    top_k=5,
)
```

Dans ce mode:

- `original_query` est envoyee a BM25;
- `vector_query` est envoyee a FAISS;
- les resultats sont deduplicates par `article_id`, puis `legi_id`;
- `score` contient le `hybrid_score`;
- `score_details` contient `vector_score`, `bm25_score` et `hybrid_score`.

Quand la question contient explicitement un numero d'article, par exemple `L1237-11`, l'article correspondant est priorise dans le score hybride final. Cela evite qu'une similarite vectorielle generale masque une demande juridique exacte.

Commande de verification locale:

```bash
python src/retrieval/hybrid_retriever.py "Que dit l'article L1237-11 ?" --top-k 3
```

### État du pipeline

L'extraction, le filtrage thématique, le chunking, la base FAISS persistante et le retrieval
hybride sont implémentés. La mise à jour du corpus est automatisable. Le travail restant avant le
rendu porte principalement sur l'évaluation documentée du retrieval et la validation de bout en
bout sur une installation propre.

## Lancer l'assistant RAG hybride

Avant le premier lancement, créer le fichier `.env` local à partir de
`.env.example`, télécharger le corpus puis construire les chunks et l'index :

```bash
python src/download_code_travail.py
python src/chunk_code_travail.py
python src/retrieval/vector_store.py
```

Lancer ensuite l'interface :

```bash
streamlit run src/ui/app.py
```

Le service est assemblé avec `HybridRetrievalEngine`. Pour chaque sous-question,
la requête originale alimente BM25 et la requête HyDE alimente FAISS. Les
résultats sont fusionnés avant la génération de la réponse, sans appel direct à
FAISS ou Groq dans l'interface.
