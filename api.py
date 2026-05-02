import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agent_v2 import analyser_startup

app = FastAPI(
    title="Startup Analyzer API",
    description="Agent RAG autonome pour analyser des startups",
    version="1.0.0"
)

# --- Modèles de données ---
class AnalyseRequest(BaseModel):
    nom: str
    questions: list[str] = [
        "Qui sont les fondateurs ?",
        "Quel problème résolvent-ils ?",
        "Quelle technologie utilisent-ils ?",
        "Quels sont leurs clients cibles ?"
    ]

class AnalyseResponse(BaseModel):
    nom: str
    rapport: str
    statut: str

# --- Routes ---
@app.get("/")
def accueil():
    return {"message": "Startup Analyzer API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/analyser", response_model=AnalyseResponse)
def analyser(request: AnalyseRequest):
    try:
        rapport = analyser_startup(
            nom=request.nom,
            questions=request.questions
        )
        return AnalyseResponse(
            nom=request.nom,
            rapport=rapport,
            statut="succès"
        )
    except Exception as e:
        return AnalyseResponse(
            nom=request.nom,
            rapport=str(e),
            statut="erreur"
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)