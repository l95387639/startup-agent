import os
import json
import chromadb
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from scraper import scrape_startup
from indexer import indexer_startup

load_dotenv()

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# --- État partagé entre tous les agents ---
class AgentState(TypedDict):
    nom_startup: str
    urls_trouvees: list[str]
    iterations: int
    chunks_suffisants: bool
    rapport: str
    questions: list[str]
    fiabilite_sources: dict

# --- LLM partagé ---
def get_llm(temperature: float = 0):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODEL,
        temperature=temperature
    )

# ============================================================
# AGENT 1 : CHERCHEUR
# ============================================================
def agent_chercheur(state: AgentState) -> AgentState:
    import requests
    
    queries = [
        f"{state['nom_startup']} startup",
        f"{state['nom_startup']} fondateurs cofondateurs équipe",
        f"{state['nom_startup']} intelligence artificielle produit technologie"
    ]
    query = queries[min(state["iterations"], len(queries) - 1)]
    print(f"\n[Chercheur] Recherche : '{query}'")

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json"
        },
        json={"q": query, "num": 8, "hl": "fr"}
    )
    
    urls = [r["link"] for r in response.json().get("organic", [])]
    urls_filtrees = [u for u in urls if "linkedin.com" not in u]
    toutes_urls = list(set(state["urls_trouvees"] + urls_filtrees))
    print(f"[Chercheur] {len(urls_filtrees)} nouvelles URLs trouvées, {len(toutes_urls)} au total")
    
    return {**state, "urls_trouvees": toutes_urls}

# ============================================================
# AGENT 2 : SCRAPER
# ============================================================
def agent_scraper(state: AgentState) -> AgentState:
    print(f"\n[Scraper] Scraping de {len(state['urls_trouvees'])} URLs...")
    data = scrape_startup(state["nom_startup"], state["urls_trouvees"])
    result = indexer_startup(data)
    
    if result is None:
        print("[Scraper] ⚠️ Aucun contenu indexé")
    
    return state

# ============================================================
# AGENT 3 : ANALYSTE
# ============================================================
def agent_analyste(state: AgentState) -> AgentState:
    print(f"\n[Analyste] Évaluation de la qualité des données...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")
    total = collection.count()
    print(f"[Analyste] {total} chunks dans la base")

    llm = get_llm(temperature=0)
    model = SentenceTransformer(MODEL_NAME)

    questions_test = [
        f"Qui sont les fondateurs de {state['nom_startup']} ?",
        f"Quel problème {state['nom_startup']} résout-il ?",
        f"Quelle technologie utilise {state['nom_startup']} ?"
    ]

    scores = []
    fiabilite = {}

    for question in questions_test:
        embedding = model.encode(question).tolist()
        resultats = collection.query(query_embeddings=[embedding], n_results=3)
        contexte = "\n".join(resultats["documents"][0])

        messages = [
            SystemMessage(content=f"""Tu es un analyste expert en évaluation de données.
Tu évalues si un contexte permet de répondre à une question sur {state['nom_startup']}.
Réponds UNIQUEMENT en JSON : {{"score": 0-10, "raison": "..."}}"""),
            HumanMessage(content=f"Question : {question}\nContexte : {contexte[:800]}")
        ]

        response = llm.invoke(messages)
        try:
            clean = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            score = result.get("score", 0)
            raison = result.get("raison", "")
            print(f"[Analyste] Score '{question[:40]}' : {score}/10 — {raison[:60]}")
            scores.append(score)
            fiabilite[question] = {"score": score, "raison": raison}
        except:
            scores.append(3)

    score_moyen = sum(scores) / len(scores)
    suffisant = score_moyen >= 6 or state["iterations"] >= 2
    
    print(f"[Analyste] Score moyen : {score_moyen:.1f}/10 — {'✓ suffisant' if suffisant else '✗ insuffisant, nouvelle recherche'}")

    return {
        **state,
        "chunks_suffisants": suffisant,
        "iterations": state["iterations"] + 1,
        "fiabilite_sources": fiabilite
    }

# ============================================================
# AGENT 4 : RÉDACTEUR
# ============================================================
def agent_redacteur(state: AgentState) -> AgentState:
    print(f"\n[Rédacteur] Génération du rapport final...")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")
    model = SentenceTransformer(MODEL_NAME)
    llm = get_llm(temperature=0.2)

    rapport = f"\n{'='*62}\n FICHE STARTUP : {state['nom_startup'].upper()}\n{'='*62}\n"
    rapport += f"Sources analysées : {len(state['urls_trouvees'])} URLs\n"
    rapport += f"Itérations de recherche : {state['iterations']}\n"

    for question in state["questions"]:
        score_info = state["fiabilite_sources"].get(question, {})
        score = score_info.get("score", "N/A")

        embedding = model.encode(question).tolist()
        resultats = collection.query(query_embeddings=[embedding], n_results=8)
        chunks = resultats["documents"][0]
        urls = [m["url"] for m in resultats["metadatas"][0]]
        contexte = "\n".join([f"[{urls[i]}]\n{chunks[i]}" for i in range(len(chunks))])

        messages = [
            SystemMessage(content=f"""Tu es un analyste startup expert.
Tu analyses UNIQUEMENT la startup {state['nom_startup']}.
Ignore toute information sur d'autres entreprises.
Réponds directement et factuellement en citant les sources [URL].
Si une information est incertaine, donne ce que tu sais et indique brièvement le niveau de certitude."""),
            HumanMessage(content=f"Contexte :\n{contexte}\n\nQuestion : {question}")
        ]

        response = llm.invoke(messages)
        rapport += f"\n{'─'*62}\n📌 {question}"
        if score != "N/A":
            rapport += f" [fiabilité: {score}/10]"
        rapport += f"\n{'─'*62}\n{response.content}\n"

    return {**state, "rapport": rapport}

# ============================================================
# AGENT 5 : VÉRIFICATEUR (Self-RAG)
# ============================================================
def agent_verificateur(state: AgentState) -> AgentState:
    print(f"\n[Vérificateur] Vérification des affirmations...")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")
    llm = get_llm(temperature=0)

    tous_chunks = collection.get()
    contexte_global = " ".join(tous_chunks["documents"][:50])

    messages = [
        SystemMessage(content=f"""Tu es un vérificateur factuel strict.
Tu reçois un rapport sur {state['nom_startup']} et le contexte source.
Tu dois :
1. Garder EXACTEMENT les mêmes sections et questions du rapport original
2. Pour chaque section, supprimer les affirmations non vérifiables
3. Si une section est entièrement invérifiable, écrire "Information non trouvée dans les sources disponibles."
4. Ne JAMAIS ajouter de nouvelles sections ou questions
5. Tout nom propre absent des sources doit être supprimé

Retourne le rapport avec exactement les mêmes sections qu'à l'entrée."""),
        HumanMessage(content=f"""CONTEXTE SOURCE :
{contexte_global[:3000]}

RAPPORT À VÉRIFIER :
{state['rapport']}

RAPPORT CORRIGÉ :""")
    ]

    response = llm.invoke(messages)
    rapport_verifie = response.content

    print(f"[Vérificateur] ✓ Rapport vérifié et nettoyé")

    return {**state, "rapport": rapport_verifie}

# --- Décision ---
def decision(state: AgentState) -> str:
    return "generer" if state["chunks_suffisants"] else "chercher"

# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================
def construire_agent():
    graph = StateGraph(AgentState)

    graph.add_node("chercheur", agent_chercheur)
    graph.add_node("scraper", agent_scraper)
    graph.add_node("analyste", agent_analyste)
    graph.add_node("redacteur", agent_redacteur)
    graph.add_node("verificateur", agent_verificateur)

    graph.set_entry_point("chercheur")
    graph.add_edge("chercheur", "scraper")
    graph.add_edge("scraper", "analyste")
    graph.add_conditional_edges("analyste", decision, {
        "chercher": "chercheur",
        "generer": "redacteur"
    })
    graph.add_edge("redacteur", "verificateur")
    graph.add_edge("verificateur", END)

    return graph.compile()

def analyser_startup(nom: str, questions: list[str]) -> str:
    agent = construire_agent()
    state = {
        "nom_startup": nom,
        "urls_trouvees": [],
        "iterations": 0,
        "chunks_suffisants": False,
        "rapport": "",
        "questions": questions,
        "fiabilite_sources": {}
    }
    result = agent.invoke(state)
    return result["rapport"]

if __name__ == "__main__":
    rapport = analyser_startup(
        nom="Histia",
        questions=[
            "Qui sont les fondateurs d'Histia ?",
            "Quel problème Histia résout-il pour ses clients ?",
            "Quelle technologie Histia utilise-t-il ?",
            "Quels sont les clients cibles d'Histia ?"
        ]
    )
    print(rapport)
    with open("rapport_agent.txt", "w", encoding="utf-8") as f:
        f.write(rapport)
    print("\n✅ Rapport sauvegardé dans rapport_agent.txt")