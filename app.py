import streamlit as st
from agent_v2 import analyser_startup

st.set_page_config(
    page_title="Startup Analyzer",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Startup Analyzer")
st.caption("Agent RAG autonome — inspiré d'Histia")

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

if st.button("Analyser", type="primary", disabled=not nom_startup):
    with st.spinner(f"Analyse de {nom_startup} en cours... (2-3 minutes)"):
        try:
            rapport = analyser_startup(
                nom=nom_startup,
                questions=questions
            )
            st.success("Analyse terminée !")
            st.markdown("---")
            st.markdown(rapport)
            st.download_button(
                label="Télécharger le rapport",
                data=rapport,
                file_name=f"rapport_{nom_startup.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("---")
st.caption("Projet réalisé par Brian — autodidacte IA/data")