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

## Contrats initiaux

La premiere brique de l'orchestration definit des contrats Python simples :

- `RetrievalEngine`: recherche les chunks pertinents pour une question.
- `RetrievedChunk`: represente un chunk, son score et ses metadonnees.
- `UserInputModerator`: valide une question avant retrieval et appel LLM.
- `LLMClient`: encapsule la generation de texte.
- `PromptBuilder`: construit les messages envoyes au modele.
- `RAGOrchestrator`: coordonne moderation, retrieval, prompt et generation.

Ces contrats permettent a Adrien et Yacine de travailler en parallele. Le moteur
de retrieval pourra etre implemente plus tard tant qu'il respecte la methode
`search(question, top_k)`.

## Optimisation retenue: recherche hybride

L'optimisation bonus recommandee est la recherche hybride, car elle combine :

- la recherche vectorielle pour les questions en langage naturel ;
- BM25 pour les mots-cles exacts et les references d'articles ;
- un score final du type `alpha * vector_score + beta * bm25_score`.

Le calcul hybride doit rester dans `src/retrieval`. L'orchestrateur RAG ne doit
pas connaitre les details de calcul : il consomme seulement des `RetrievedChunk`
tries par score final. Les details facultatifs des scores peuvent etre exposes
dans `score_details` pour l'interface et les tests.
