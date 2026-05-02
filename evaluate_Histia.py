import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

def retrieval(question: str, collection, top_k: int = 5) -> list[dict]:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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

# Questions de test + réponses de référence
QUESTIONS = [
    "Qui sont les fondateurs d'Histia ?",
    "Quel problème Histia résout-il pour ses clients ?",
    "Quelle technologie Histia utilise-t-il ?",
]

REPONSES_REFERENCE = [
    "Les fondateurs d'Histia sont Hiyu Shintani et Léopold Servais.",
    "Histia résout le problème de la recherche d'informations fiables et de la rédaction d'études de marché.",
    "Histia utilise une technologie propriétaire de traitement de données basée sur l'IA pour fiabiliser les informations.",
]

def preparer_dataset():
    """Prépare le dataset pour RAGAS."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("startups")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for question, reference in zip(QUESTIONS, REPONSES_REFERENCE):
        print(f"Évaluation : {question[:50]}...")

        chunks = retrieval(question, collection, top_k=5)
        contexte = [c["contenu"] for c in chunks]

        prompt = f"""Réponds à cette question en te basant uniquement sur le contexte.
        
Contexte : {" ".join(contexte[:3])}
Question : {question}
Réponse :"""
        response = llm.invoke(prompt)

        questions.append(question)
        answers.append(response.content)
        contexts.append(contexte)
        ground_truths.append(reference)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

def evaluer():
    """Lance l'évaluation RAGAS."""
    print("Préparation du dataset...")
    dataset = preparer_dataset()

    print("\nLancement de l'évaluation RAGAS...")

    llm = LangchainLLMWrapper(ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
        request_timeout=120
    ))
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Configure les métriques avec notre LLM
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm

    resultats = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== RÉSULTATS RAGAS ===")
    print(resultats)

    df = resultats.to_pandas()
    df.to_csv("ragas_results_veesion.csv", index=False)
    print("\nRésultats sauvegardés dans ragas_results_veesion.csv")

if __name__ == "__main__":
    evaluer()