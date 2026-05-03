# 🤖 Startup Analyzer — Agent RAG autonome

🚀 **[Tester la démo en ligne](https://simonaaaa-startup-analyzer.hf.space)**

---

Agent IA capable d'analyser n'importe quelle startup de façon autonome : il cherche ses propres sources, les scrape, les indexe et génère une fiche d'analyse sourcée — sans qu'on lui fournisse une seule URL.

Inspiré de l'architecture de [Histia](https://histia.net) — startup IA spécialisée en intelligence de marché.

---

## 🧠 Architecture multi-agents

```
Orchestrateur LangGraph
        ↓
┌─────────────────────────────────┐
│  🔍 Agent Chercheur             │ → trouve les URLs via Serper
│  🕷️  Agent Scraper              │ → scrape avec aiohttp + Playwright
│  📊 Agent Analyste              │ → évalue la qualité (score 0-10)
│  ✍️  Agent Rédacteur            │ → génère la fiche sourcée
└─────────────────────────────────┘
        ↓
   Boucle si score insuffisant
```

## ⚙️ Stack technique

| Composant | Technologie |
|---|---|
| Orchestration | `LangGraph` |
| Search API | `Serper.dev` |
| Scraping rapide | `aiohttp` + `BeautifulSoup` |
| Scraping protégé | `Playwright` (fallback automatique) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorielle | `ChromaDB` |
| LLM | `Llama 3.3 70B` via Groq API |
| Évaluation | `RAGAS` (faithfulness 0.75, relevancy 0.93, precision 0.65) |
| API | `FastAPI` |
| Interface | `Streamlit` |

---

## 🚀 Installation

```bash
git clone https://github.com/l95387639/startup-agent
cd startup-agent

conda create -n startup-agent python=3.11
conda activate startup-agent

pip install -r requirements.txt
playwright install chromium
```

Crée un fichier `.env` :

```
GROQ_API_KEY=ta_clé_groq
SERPER_API_KEY=ta_clé_serper
```

---

## 💻 Utilisation

**Terminal 1 — Lance l'API :**
```bash
python api.py
```

**Terminal 2 — Lance l'interface :**
```bash
streamlit run app.py
```

Ouvre **http://localhost:8501** dans ton navigateur, entre un nom de startup et clique sur Analyser.

---

## 🔬 Concepts implémentés

**Architecture multi-agents avec LangGraph**
5 agents spécialisés qui communiquent via un état partagé. L'agent analyste évalue la qualité des données (score 0-10) et décide si une nouvelle itération de recherche est nécessaire.

**Scraping intelligent**
aiohttp pour les sites standards (rapide), Playwright en fallback automatique pour les sites qui bloquent les requêtes simples.

**Filtre de pertinence des sources**
Avant de scraper, le LLM évalue si chaque URL trouvée par Serper parle vraiment de la startup analysée. Les URLs hors sujet sont automatiquement ignorées.

**Self-RAG — Agent Vérificateur**
Après génération du rapport, un agent vérifie chaque affirmation factuelle contre les sources scrapées. Les hallucinations et informations non vérifiables sont supprimées automatiquement.

**Évaluation RAGAS sur Histia**
Mesure objective de la qualité du pipeline : faithfulness 0.75, answer relevancy 0.93, context precision 0.65.

**RAG multilingue sourcé**
Embeddings multilingues + ChromaDB. Chaque affirmation est liée à sa source d'origine.

---

## 📈 Pistes d'amélioration
- **LLM payant**
- **Re-ranking** 
- **LangSmith** 

---

## 👤 Auteur

Projet réalisé en autodidacte dans le cadre d'une candidature de stage.
Inspiré de l'architecture de [Histia](https://histia.net) — startup IA spécialisée en intelligence de marché.
