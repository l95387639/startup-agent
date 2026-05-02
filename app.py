import streamlit as st
import requests

st.set_page_config(
    page_title="Startup Analyzer",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Startup Analyzer")
st.caption("Agent RAG autonome — inspiré d'Histia")

# --- Formulaire ---
nom_startup = st.text_input("Nom de la startup à analyser", placeholder="ex: Mistral AI")

questions_defaut = [
    "Qui sont les fondateurs ?",
    "Quel problème résolvent-ils ?",
    "Quelle technologie utilisent-ils ?",
    "Quels sont leurs clients cibles ?"
]

with st.expander("Personnaliser les questions"):
    questions_text = st.text_area(
        "Une question par ligne",
        value="\n".join(questions_defaut),
        height=150
    )
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

# --- Lancement ---
if st.button("Analyser", type="primary", disabled=not nom_startup):
    with st.spinner(f"Analyse de {nom_startup} en cours... (2-3 minutes)"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/analyser",
                json={"nom": nom_startup, "questions": questions},
                timeout=300
            )
            data = response.json()

            if data["statut"] == "succès":
                st.success("Analyse terminée !")
                st.markdown("---")
                st.markdown(data["rapport"])

                # Bouton téléchargement
                st.download_button(
                    label="Télécharger le rapport",
                    data=data["rapport"],
                    file_name=f"rapport_{nom_startup.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"Erreur : {data['rapport']}")

        except Exception as e:
            st.error(f"Impossible de contacter l'API : {e}")

st.markdown("---")
st.caption("Projet réalisé par Brian — autodidacte IA/data")