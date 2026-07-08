# Architecture generale

## Principe

Le projet est organise autour de deux phases independantes.

## Phase 1: indexation

Documents source -> nettoyage -> chunking -> embeddings -> base vectorielle persistante.

Cette phase est executee uniquement lors de la creation ou de la mise a jour du corpus.

## Phase 2: interrogation

Question utilisateur -> moderation -> embedding -> recherche top-k -> contexte -> LLM -> reponse avec sources et avertissement juridique.

Cette phase recharge la base vectorielle existante.

## Modules prevus

- `src/indexing`: preparation du corpus, parsing, nettoyage, chunking.
- `src/retrieval`: embeddings de requete, recherche top-k, scores, metadonnees.
- `src/rag`: orchestration globale du pipeline d'interrogation.
- `src/moderation`: controle du corpus et controle des entrees utilisateur.
- `src/llm`: clients LLM, prompts, generation de reponse.
- `src/ui`: interface utilisateur.
- `src/config`: configuration partagee.

## Regles de conception

- Garder les classes petites et specialisees.
- Preferer la composition lorsque plusieurs composants collaborent.
- Utiliser l'heritage seulement pour des contrats communs clairs.
- Eviter les dependances entre l'interface et les details internes du RAG.
- Ne jamais exposer de secret ou de prompt systeme dans les reponses.
