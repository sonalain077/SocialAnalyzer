"""
Point d’entrée : exécute le LangGraph complet (avec rapport inclus) sur un ou plusieurs PDFs.
Usage :
    python -m src.main --pdf data/raw/entretien1.pdf
"""

from __future__ import annotations
import argparse
from pathlib import Path
from src.graph import build_graph
from src.state import GraphState


def run_pipeline(raw_dir="data/raw", max_themes=10):
    raw_dir_path = Path(raw_dir)

    if raw_dir_path.is_dir():
        pdf_files = sorted(raw_dir_path.glob("*.pdf"))
    elif raw_dir_path.is_file() and raw_dir_path.suffix.lower() == ".pdf":
        pdf_files = [raw_dir_path]
    else:
        raise FileNotFoundError(f"Aucun fichier ou dossier PDF valide trouvé : {raw_dir}")

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {raw_dir_path}")

    num_files = len(pdf_files)


    print(f"\n=== Lancement de l'analyse qualitative sur {num_files} fichier(s) PDF ===")

    g = build_graph()

    for idx, pdf_path in enumerate(pdf_files):
        print(f"\n🔍 Traitement de : {pdf_path.name}")

        # Incrémente le compteur externe


        # Passe les variables au state
        state = GraphState(
            pdf_path=str(pdf_path),
            max_themes=max_themes,
            num_files=num_files,
            files_processed= idx
        )

        try:
            result_state = g.invoke(state)
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {pdf_path.name} : {e}")
            continue

        # Log du rapport généré
        output_files = result_state.get("output_files", {})
        rapport = output_files.get("rapport_txt", None)

        if rapport:
            print(f"✅ Rapport généré : {rapport}")
        else:
            print(f"⚠️ Aucun rapport généré pour {pdf_path.name} (vérifie les logs)")

        # Synthèse globale (se fera sur le dernier passage)
        global_synth = result_state.get("rapport_global", None)

        if global_synth:
            print(f"\n===== Synthèse globale du corpus =====\n")
            print(global_synth)

            output_path = Path("output") / "synthese_globale_corpus.txt"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(global_synth)

            print(f"\n📝 Synthèse globale sauvegardée : {output_path}")

    print("\n=== Pipeline terminé pour tous les fichiers ===")


def main():
    parser = argparse.ArgumentParser(description="Pipeline d'analyse qualitative de PDF avec LangGraph")
    parser.add_argument("--pdf", type=str, default="data/raw", help="Chemin vers un fichier PDF ou un dossier")
    parser.add_argument("--max_themes", type=int, default=10, help="Nombre maximal de thèmes")
    args = parser.parse_args()

    print("\n=== Lancement du pipeline ===")
    print(f"Source : {args.pdf}")
    print(f"Nombre max de thèmes : {args.max_themes}\n")

    run_pipeline(raw_dir=args.pdf, max_themes=args.max_themes)


if __name__ == "__main__":
    main()
