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
