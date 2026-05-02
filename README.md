# 🤖 Startup Agent — Agent RAG autonome avec LangGraph

Agent IA capable d'analyser une startup de façon entièrement autonome : il cherche lui-même les sources, les scrape, les indexe et génère une fiche d'analyse sourcée — sans qu'on lui fournisse une seule URL.

---

## 🧠 Architecture

```
Nom startup
     ↓
Search API (Serper) — trouve les URLs automatiquement
     ↓
Scraping + Chunking + Embeddings → ChromaDB
     ↓
LLM évalue : assez d'infos ?
     ↓ Non → nouvelle recherche
     ↓ Oui
Fiche d'analyse sourcée
```

## ⚙️ Stack technique

| Composant | Technologie |
|---|---|
| Orchestration agent | `LangGraph` |
| Search API | `Serper.dev` |
| Scraping | `requests` + `BeautifulSoup` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorielle | `ChromaDB` |
| LLM | `Llama 3.1 8B` via Groq API |

---

## 🚀 Installation

```bash
git clone https://github.com/ton-username/startup-agent
cd startup-agent

conda create -n startup-agent python=3.11
conda activate startup-agent

pip install langgraph langchain langchain-groq langchain-core requests beautifulsoup4 chromadb sentence-transformers groq python-dotenv google-search-results
```

Crée un fichier `.env` :

```
GROQ_API_KEY=ta_clé_groq
SERPER_API_KEY=ta_clé_serper
```

---

## 💻 Utilisation

```bash
python agent_v2.py
```

Pour analyser une autre startup, modifie le bas de `agent_v2.py` :

```python
rapport = analyser_startup(
    nom="Mistral AI",
    questions=[
        "Qui sont les fondateurs ?",
        "Quel est leur modèle économique ?",
        "Quels sont leurs produits ?",
    ]
)
```

---

## 🔬 Concepts implémentés

**Agent autonome avec LangGraph**
Graphe d'états avec boucle de décision — l'agent évalue lui-même si il a assez d'informations et relance une recherche si nécessaire, sans intervention humaine.

**Search API automatique**
L'agent utilise Serper.dev pour découvrir les sources pertinentes à partir du nom de la startup — aucune URL à fournir manuellement.

**Évaluation de la qualité par LLM**
Avant de générer le rapport, le LLM score la qualité des données disponibles (0-10) et décide s'il faut chercher plus d'informations.

**RAG multilingue sourcé**
Embeddings multilingues + ChromaDB pour un retrieval précis en français et en anglais. Chaque affirmation est liée à sa source.

---

## 📈 Limitations et pistes d'amélioration

- Certains sites bloquent le scraping (LinkedIn, Pappers) — Playwright permettrait de contourner ces protections
- Scraping séquentiel — `asyncio` permettrait de paralléliser et d'aller 5x plus vite
- Le scoring de qualité pourrait intégrer des métriques RAGAS pour être plus robuste

---

## 👤 Auteur

Projet réalisé en autodidacte dans le cadre d'une candidature de stage.
Inspiré de l'architecture de [Histia](https://histia.net) — startup IA spécialisée en intelligence de marché.