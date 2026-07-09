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

La mise a jour est controlee par un hash SHA256 de `data/processed/chunks_code_du_travail.json`. Si les chunks n'ont pas change et que les fichiers FAISS existent deja, la base vectorielle n'est pas reconstruite. Le mode `--force` permet de reconstruire volontairement la base, par exemple apres un changement de modele d'embedding.

```bash
python src/retrieval/vector_store.py
python src/retrieval/vector_store.py --force
python src/retrieval/vector_store.py --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

### Suite prevue

- Finaliser l'extraction des articles.
- Filtrer les articles par themes.
- Chunker par article sans casser les articles.
- Enrichir les chunks avec les metadonnees.
- Generer les embeddings.
- Indexer dans une base vectorielle persistante.
- Ajouter le retrieval vectoriel puis BM25/hybride.
- Automatiser la mise a jour du corpus.
- Evaluer la qualite du retrieval sur des questions connues.
