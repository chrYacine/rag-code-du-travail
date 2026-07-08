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
