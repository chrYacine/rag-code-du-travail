# Roadmap initiale

## Objectif

Construire un assistant RAG sur le Code du travail avec une architecture Python modulaire, maintenable et conforme au workflow Git `main -> dev -> feature -> PR`.

## Contraintes projet retenues

- Ne pas utiliser LangChain ni LlamaIndex.
- Separer strictement l'indexation et l'interrogation.
- Persister la base vectorielle et ne pas la reconstruire au lancement.
- Utiliser le meme modele d'embedding pour l'indexation et la recherche.
- Citer les articles utilises dans les reponses.
- Afficher un avertissement juridique dans toutes les reponses.
- Integrer des agents de moderation modulaires.
- Fournir une interface utilisateur separee du moteur RAG.

## Repartition

### Adrien

- Architecture globale.
- Orchestration RAG.
- Prompts systeme.
- Integration Groq.
- Interface utilisateur.
- README et documentation.
- Integration finale.

### Yacine

- Preparation du corpus.
- Parsing et nettoyage.
- Chunking.
- Embeddings.
- Base vectorielle persistante.
- Retrieval.
- Tests du retrieval.

### Ensemble

- Validation de l'architecture.
- Revue des Pull Requests.
- Tests finaux.
- Reponses aux questions du sujet.
- Release finale.

## Branches recommandees

- `feature/init-project-structure`
- `feature/corpus-parser`
- `feature/chunking-pipeline`
- `feature/vector-store`
- `feature/retrieval-engine`
- `feature/rag-orchestrator`
- `feature/groq-generation`
- `feature/moderation-agents`
- `feature/streamlit-ui`
- `feature/documentation-and-final-report`

## Jalons

1. Initialiser l'arborescence commune.
2. Documenter les choix d'architecture.
3. Implementer le pipeline d'indexation.
4. Implementer le moteur de retrieval.
5. Implementer l'orchestrateur RAG.
6. Ajouter les moderateurs.
7. Integrer Groq.
8. Construire l'interface utilisateur.
9. Ajouter les tests et controles qualite.
10. Finaliser la documentation et taguer la release.
