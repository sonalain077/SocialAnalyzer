import streamlit as st
from pathlib import Path
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from datetime import datetime
from src.main import run_pipeline

# --- Config Streamlit ---
st.set_page_config(page_title="🧠 Analyse Qualitative IA", layout="centered")
st.title("📘 Social Analyzer")

# --- Upload & paramètres ---
uploaded_files = st.file_uploader("📎 Uploade un ou plusieurs fichiers PDF", type=["pdf"], accept_multiple_files=True)
max_themes = st.slider("🔢 Nombre maximal de thèmes à extraire", 3, 20, 10)
start_btn = st.button("🚀 Lancer l’analyse")

# 🔄 Chemin vers la synthèse AFFINÉE
rapport_global_refined_path = Path("data/repport/rapport_global_affine.txt")


def generate_pdf_from_text(text: str, output_path: Path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontSize=11, leading=16))
    styles.add(ParagraphStyle(name='CenteredTitle', alignment=TA_CENTER, fontSize=16, spaceAfter=20))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    elements = []

    # Page de garde
    elements.append(Paragraph("Synthèse Sociologique", styles['CenteredTitle']))
    elements.append(Paragraph("Rapport généré automatiquement", styles['Normal']))
    elements.append(Paragraph(f"Date : {datetime.today().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(PageBreak())

    # Corps du rapport
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 12))
        elif line.startswith("###"):
            elements.append(Paragraph(f"<b>{line[4:]}</b>", styles['Heading3']))
        elif line.startswith("**") and line.endswith("**"):
            elements.append(Paragraph(f"<b>{line.strip('**')}</b>", styles['Heading2']))
        else:
            elements.append(Paragraph(line, styles['Justify']))

    doc.build(elements)


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


# --- Affichage du rapport global AFFINÉ ---
if rapport_global_refined_path.exists():
    st.subheader("📄 Synthèse sociologique du corpus")
    with open(rapport_global_refined_path, "r", encoding="utf-8") as f:
        synthese = f.read()

    # Affichage esthétique
    
    html_synthese = synthese.replace("\n", "<br><br>")
    st.markdown(f"<div style='text-align: justify;'>{html_synthese}</div>", unsafe_allow_html=True)

    # Téléchargement TXT
    st.download_button(
        label="💾 Télécharger la synthèse (TXT)",
        data=synthese,
        file_name="rapport_global_affine.txt",
        mime="text/plain"
    )

    # Téléchargement PDF professionnel
    pdf_output_path = Path("data/repport/rapport_sociologique_pro.pdf")
    generate_pdf_from_text(synthese, pdf_output_path)

    with open(pdf_output_path, "rb") as f:
        st.download_button(
            label="📥 Télécharger en PDF",
            data=f,
            file_name="rapport_sociologique_affine.pdf",
            mime="application/pdf"
        )