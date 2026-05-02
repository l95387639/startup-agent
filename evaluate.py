import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from agent_v2 import retrieval
import chromadb

load_dotenv()

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

        # Retrieval
        chunks = retrieval(question, collection, top_k=5)
        contexte = [c["contenu"] for c in chunks]

        # Génération de la réponse
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
        temperature=0
    ))
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    resultats = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== RÉSULTATS RAGAS ===")
    print(resultats)

    # Sauvegarde
    df = resultats.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    print("\nRésultats sauvegardés dans ragas_results.csv")

if __name__ == "__main__":
    evaluer()