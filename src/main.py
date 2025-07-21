"""main.py
Point d’entrée simple : exécute le LangGraph sur un PDF fourni en argument.
Exemple :
    python -m src.main --pdf data/raw/entretien1.pdf
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from src.nodes import generate_full_report, load_api_key, create_groq_client
from src.graph import build_graph
from src.state import GraphState



def run_pipeline(pdf_path: str, max_themes: int = 10, out_dir: str = "data/outputs") -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    g = build_graph()
    state = GraphState(pdf_path=pdf_path, max_themes=max_themes)

    result_state = g.invoke(state)  # Retourne un dictionnaire

    # Enregistrer les résultats principaux
    codes_path = out_dir_path / f"{Path(pdf_path).stem}_codes.json"
    clusters_path = out_dir_path / f"{Path(pdf_path).stem}_clusters.json"
    labels_path = out_dir_path / f"{Path(pdf_path).stem}_labels.json"

    with open(codes_path, "w", encoding="utf-8") as f:
        json.dump(result_state["all_codes"], f, ensure_ascii=False, indent=2)
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(result_state["clusters"], f, ensure_ascii=False, indent=2)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(result_state["theme_labels"], f, ensure_ascii=False, indent=2)

    print(f"Pipeline terminé. Résultats enregistrés dans {out_dir_path}")
        # --- Génération du rapport supplémentaire ---s
    csv_path = "./output/final_results.csv"
    report_output_path = out_dir_path / f"{Path(pdf_path).stem}_rapport.txt"

    if os.path.exists(csv_path):
        api_key = load_api_key()
        client = create_groq_client(api_key)

        generate_full_report(client=client, csv_path=csv_path, output_txt_path=str(report_output_path))
        print(f"Rapport final généré : {report_output_path}")
    else:
        print(f"Fichier {csv_path} introuvable pour la génération du rapport.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline d'analyse qualitative LangGraph")
    parser.add_argument("--pdf", help="Chemin vers le PDF à analyser. Si omis, prend le premier PDF dans data/raw/")
    parser.add_argument("--max_themes", type=int, default=10, help="Nombre maximal de thèmes (clusters)")
    parser.add_argument("--out", default="data/outputs", help="Dossier de sortie")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path:
        # Cherche le premier PDF dans data/raw
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        pdf_files = sorted(raw_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("Aucun PDF trouvé dans data/raw et --pdf non fourni")
        pdf_path = str(pdf_files[0])
        print(f"--pdf, utilisation de {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    run_pipeline(pdf_path, args.max_themes, args.out)


if __name__ == "__main__":
    main()
