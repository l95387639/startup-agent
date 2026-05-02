import os
import json
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch

from scraper_playwright import scrape_startup
from indexer import indexer_startup

load_dotenv()

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# --- État de l'agent ---
class AgentState(TypedDict):
    nom_startup: str
    urls_trouvees: list[str]
    iterations: int
    chunks_suffisants: bool
    rapport: str
    questions: list[str]

# --- Outil : chercher des URLs via Serper ---
def chercher_urls(nom_startup: str, iteration: int = 0) -> list[str]:
    import requests
    queries = [
        f"{nom_startup} startup",
        f"{nom_startup} fondateurs cofondateurs équipe",
        f"{nom_startup} intelligence artificielle produit"
    ]
    query = queries[min(iteration, len(queries) - 1)]
    print(f"  Recherche Serper : '{query}'")

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json"
        },
        json={"q": query, "num": 8, "hl": "fr"}
    )
    results = response.json()
    urls = [r["link"] for r in results.get("organic", [])]
    print(f"  {len(urls)} URLs trouvées")
    return urls

# --- Outil : retrieval dans ChromaDB ---
def retrieval(question: str, collection, top_k: int = 5) -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode(question).tolist()
    resultats = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    chunks = []
    for i in range(len(resultats["documents"][0])):
        chunks.append({
            "contenu": resultats["documents"][0][i],
            "url": resultats["metadatas"][0][i]["url"],
            "distance": resultats["distances"][0][i]
        })
    return chunks

# --- Noeud 1 : rechercher les URLs ---
def node_chercher_urls(state: AgentState) -> AgentState:
    print(f"\n[Agent] Recherche URLs (itération {state['iterations'] + 1})...")
    urls = chercher_urls(state["nom_startup"], state["iterations"])
    # Ajoute les nouvelles URLs sans doublons
    toutes_urls = list(set(state["urls_trouvees"] + urls))
    return {**state, "urls_trouvees": toutes_urls}

# --- Noeud 2 : scraper et indexer ---
def node_scraper_indexer(state: AgentState) -> AgentState:
    print(f"\n[Agent] Scraping et indexation de {len(state['urls_trouvees'])} URLs...")
    data = scrape_startup(state["nom_startup"], state["urls_trouvees"])
    indexer_startup(data)
    return state

# --- Noeud 3 : évaluer si on a assez d'infos ---
def node_evaluer(state: AgentState) -> AgentState:
    print("\n[Agent] Évaluation de la qualité des données...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")
    total = collection.count()
    print(f"  {total} chunks dans la base")

    # Teste si on peut répondre aux questions clés
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODEL,
        temperature=0
    )

    questions_test = [
        f"Qui sont les fondateurs de {state['nom_startup']} ?",
        f"Quel problème {state['nom_startup']} résout-il ?"
    ]

    scores = []
    for question in questions_test:
        chunks = retrieval(question, collection, top_k=3)
        contexte = "\n".join([c["contenu"] for c in chunks])
        
        prompt = f"""Tu analyses la qualité d'un contexte pour répondre à une question.
Réponds UNIQUEMENT par un JSON : {{"score": 0-10, "raison": "..."}}

Question : {question}
Contexte disponible : {contexte[:1000]}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            clean = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            score = result.get("score", 0)
            print(f"  Score '{question[:40]}...' : {score}/10")
            scores.append(score)
        except:
            scores.append(3)

    score_moyen = sum(scores) / len(scores)
    suffisant = score_moyen >= 7 or state["iterations"] >= 3
    print(f"  Score moyen : {score_moyen:.1f}/10 — {'suffisant ✓' if suffisant else 'insuffisant, nouvelle recherche...'}")

    return {**state, "chunks_suffisants": suffisant, "iterations": state["iterations"] + 1}

# --- Noeud 4 : générer le rapport ---
def node_generer_rapport(state: AgentState) -> AgentState:
    print("\n[Agent] Génération du rapport final...")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODEL,
        temperature=0.2
    )

    rapport = f"\n{'='*62}\n FICHE STARTUP : {state['nom_startup'].upper()}\n{'='*62}\n"
    rapport += f"Sources analysées : {len(state['urls_trouvees'])} URLs\n"
    rapport += f"Itérations de recherche : {state['iterations']}\n"

    for question in state["questions"]:
        chunks = retrieval(question, collection, top_k=8)
        contexte = "\n".join([f"[{c['url']}]\n{c['contenu']}" for c in chunks])

        prompt = f"""Tu es un analyste startup. Réponds directement et factuellement.
Cite les sources avec [URL].
Si l'info est absente, dis-le en une phrase.

CONTEXTE:
{contexte}

QUESTION: {question}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        rapport += f"\n{'─'*62}\n📌 {question}\n{'─'*62}\n{response.content}\n"

    return {**state, "rapport": rapport}

# --- Décision : continuer ou générer ---
def decision(state: AgentState) -> str:
    if state["chunks_suffisants"]:
        return "generer"
    return "chercher"

# --- Construction du graphe LangGraph ---
def construire_agent():
    graph = StateGraph(AgentState)

    graph.add_node("chercher_urls", node_chercher_urls)
    graph.add_node("scraper_indexer", node_scraper_indexer)
    graph.add_node("evaluer", node_evaluer)
    graph.add_node("generer_rapport", node_generer_rapport)

    graph.set_entry_point("chercher_urls")
    graph.add_edge("chercher_urls", "scraper_indexer")
    graph.add_edge("scraper_indexer", "evaluer")
    graph.add_conditional_edges("evaluer", decision, {
        "chercher": "chercher_urls",
        "generer": "generer_rapport"
    })
    graph.add_edge("generer_rapport", END)

    return graph.compile()

# --- Point d'entrée ---
def analyser_startup(nom: str, questions: list[str]) -> str:
    agent = construire_agent()
    state = {
        "nom_startup": nom,
        "urls_trouvees": [],
        "iterations": 0,
        "chunks_suffisants": False,
        "rapport": "",
        "questions": questions
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