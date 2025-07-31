import streamlit as st
from pathlib import Path
import shutil
import os

from src.main import run_pipeline

# --- Config Streamlit ---
st.set_page_config(page_title="🧠 Analyse Qualitative IA", layout="centered")
st.title("Social Analyzer")

# --- Upload & paramètres ---
uploaded_files = st.file_uploader("📎 Uploade un ou plusieurs fichiers PDF", type=["pdf"], accept_multiple_files=True)
max_themes = st.slider("🔢 Nombre maximal de thèmes à extraire", 3, 20, 10)
start_btn = st.button("🚀 Lancer l’analyse")
rapport_global_path = Path("data/repport/rapport_global.txt")


# --- Pipeline d'analyse ---
if start_btn:
    if not uploaded_files:
        st.warning("Merci de fournir au moins un fichier PDF.")
    else:
        temp_dir = Path("data/raw")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Nettoyage des anciens fichiers
        for old_file in temp_dir.glob("*.pdf"):
            old_file.unlink()

        # Sauvegarde locale des fichiers uploadés
        for file in uploaded_files:
            with open(temp_dir / file.name, "wb") as f:
                f.write(file.read())

        with st.spinner("🛠️ Lancement du pipeline..."):
            try:
                run_pipeline(raw_dir=str(temp_dir), max_themes=max_themes)
                st.success("✅ Analyse terminée avec succès.")
            except Exception as e:
                st.error(f"❌ Erreur pendant l’analyse : {e}")

# --- Affichage du rapport global ---
if rapport_global_path.exists():
    st.subheader("📄 Synthèse globale du corpus")
    with open(rapport_global_path, "r", encoding="utf-8") as f:
        synthese = f.read()
        st.text_area("📝 Rapport généré :", value=synthese, height=600)

    st.download_button(
        label="💾 Télécharger la synthèse",
        data=synthese,
        file_name="rapport_global.txt",
        mime="text/plain"
    )
