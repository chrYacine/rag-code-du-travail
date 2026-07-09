# Compte rendu — Assistant RAG Code du travail

## Objectif et résultat

Le projet réalise un assistant d'information juridique fondé sur un corpus ciblé du Code du
travail. Nous avons implémenté manuellement les briques du pipeline, sans LangChain ni LlamaIndex :
téléchargement et contrôle du corpus, nettoyage, chunking par article, embeddings
`sentence-transformers`, index FAISS persistant, recherche BM25, fusion hybride, construction du
prompt et appel à Groq. Une interface Streamlit affiche la réponse, l'avertissement juridique et
les articles effectivement retrouvés.

## Difficultés rencontrées

La première difficulté a été l'accès au corpus. L'API officielle Légifrance/PISTE implique une
application, OAuth et une gestion de jetons disproportionnée pour le temps du projet. Nous avons
retenu le JSON structuré de SocialGouv/legi-data comme source technique, tout en documentant
Légifrance comme source juridique primaire. La structure récursive du JSON, les états des articles
et les numéros à suffixes (`L1221-10-1`) ont nécessité des contrôles et des tests spécifiques.

La deuxième difficulté concernait la cohérence entre les vecteurs et leurs métadonnées. FAISS ne
stocke que les vecteurs : nous persistons donc une liste parallèle ordonnée, le modèle d'embedding,
la dimension et le hash des chunks. L'index n'est reconstruit que si le corpus ou le modèle change.

Enfin, les scores FAISS et BM25 ne sont pas directement comparables. Le moteur hybride récupère un
ensemble élargi de candidats, normalise les scores de chaque moteur, déduplique par article et
calcule une combinaison pondérée. Les recherches par numéro d'article reçoivent un traitement
lexical explicite afin qu'une similarité sémantique générale ne masque pas l'article demandé.

## Décisions de conception

Nous avons choisi un chunk par article complet. Cette granularité simplifie la citation et respecte
l'unité juridique, au prix d'un contexte parfois dispersé entre plusieurs dispositions. Le fil
d'Ariane, le thème, l'identifiant lisible, l'identifiant Légifrance et la date du corpus sont
conservés dans les métadonnées.

L'interrogation combine la question originale pour BM25 et une représentation HyDE pour FAISS. Les
questions comparatives peuvent être décomposées en deux à quatre sous-questions, puis les résultats
sont agrégés. Ces améliorations augmentent le coût en appels LLM ; elles disposent donc de
fallbacks vers la question originale. Une modération déterministe bloque les injections explicites
avant tout appel LLM.

Le prompt limite le modèle au contexte retrouvé, interdit l'invention d'articles et impose une
réponse prudente pour les cas conditionnels ou individuels. L'avertissement juridique est ajouté
par l'orchestrateur, indépendamment du texte généré, afin qu'il soit toujours présent.

## Organisation et qualité

Le travail a été réparti entre corpus/retrieval et orchestration/LLM/interface. Les fonctionnalités
ont été développées sur des branches dédiées et intégrées dans `dev` par Pull Requests. Les
contrats `RetrievedChunk`, `RetrievalEngine`, `LLMClient` et `UserInputModerator` ont permis de
travailler en parallèle. Les tests couvrent les erreurs réseau, le chunking, la persistance, FAISS,
BM25, la fusion hybride, Groq mocké, la modération, HyDE, la décomposition et l'orchestration.

## Avec davantage de temps

Nous migrerions vers l'API officielle PISTE, ajouterions les dates d'entrée en vigueur et un lien
Légifrance vérifiable pour chaque source. Nous construirions un jeu d'évaluation annoté plus large
avec des mesures `Recall@k`, MRR et des tests de fidélité des citations. Nous testerions aussi une
expansion des articles voisins et des renvois juridiques, un reranker, une meilleure détection des
demandes de conseil individuel et une évaluation comparative du coût et du gain réel de HyDE.
Enfin, nous ajouterions une surveillance automatisée de la qualité après chaque mise à jour du
corpus.
