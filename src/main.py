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
from src.nodes import generate_full_report, load_api_key, create_groq_client, compile_results_with_meta
from src.graph import build_graph
from src.state import GraphState



def run_pipeline(raw_dir="data/raw", max_themes=10, out_dir="data/outputs"):
    raw_dir_path = Path(raw_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_dir_path.glob("*.pdf"))
    if raw_dir_path.is_dir():
        pdf_files = sorted(raw_dir_path.glob("*.pdf"))
    elif raw_dir_path.is_file() and raw_dir_path.suffix.lower() == ".pdf":
        pdf_files = [raw_dir_path]
    else:
        raise FileNotFoundError(f"Aucun fichier ou dossier PDF valide trouvé : {raw_dir}")

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {raw_dir_path}")

    # Initialise le client GROQ une fois pour tout le batch
    api_key = load_api_key() 
    client = create_groq_client(api_key)

    for pdf_path in pdf_files:
        print(f"\n=== Traitement de {pdf_path.name} ===")

        base_name = pdf_path.stem

        # === Étape 1 : LangGraph (extraction, codage, clustering) ===
        g = build_graph()
        state = GraphState(pdf_path=str(pdf_path), max_themes=max_themes)
        result_state = g.invoke(state)
        print("\n=== Contenu de result_state ===")
        print(result_state.keys())

        # === Sauvegarde des outputs intermédiaires ===
        codes_path = out_dir_path / f"{base_name}_codes.json"
        clusters_path = out_dir_path / f"{base_name}_clusters.json"
        labels_path = out_dir_path / f"{base_name}_labels.json"

        with open(codes_path, "w", encoding="utf-8") as f:
            json.dump(result_state["all_codes"], f, ensure_ascii=False, indent=2)
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump(result_state["clusters"], f, ensure_ascii=False, indent=2)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(result_state["theme_labels"], f, ensure_ascii=False, indent=2)

        # === Étape 2 : Compilation en CSV ===
        df = compile_results_with_meta(
            result_state["all_codes"],
            result_state["code_to_cluster"],
            result_state["theme_labels"],
            result_state.get("meta_theme_map", {k: v for k, v in result_state["theme_labels"].items()}),  # fallback
            result_state.get("theme_to_meta", {k: k for k in result_state["theme_labels"].keys()})         # fallback
        )
        csv_path = out_dir_path / f"{base_name}_clustering.csv"
        df.to_csv(csv_path, index=False)
        report_dir = Path("data/repport")
        
        # === Étape 3 : Rapport complet ===
        report_output_path = report_dir / f"{base_name}_rapport.txt"
        generate_full_report(
            client=client,
            csv_path=str(csv_path),
            output_txt_path=str(report_output_path)
        )

        print(f"✅ Rapport généré pour {pdf_path.name}: {report_output_path}")

    print("\n=== Traitement du corpus terminé ===")


import argparse


def main():
    parser = argparse.ArgumentParser(description="Pipeline d'analyse de corpus PDF")
    parser.add_argument("--pdf", type=str, default="data/raw", help="Chemin vers un fichier PDF ou un dossier contenant des PDFs")
    parser.add_argument("--max_themes", type=int, default=10, help="Nombre maximal de thèmes à extraire")
    parser.add_argument("--out", type=str, default="data/outputs", help="Dossier de sortie pour les résultats")
    args = parser.parse_args()

    # Log pratique
    print("\n=== Lancement du pipeline ===")
    print(f"Source : {args.pdf}")
    print(f"Nombre maximal de thèmes : {args.max_themes}")
    print(f"Dossier de sortie : {args.out}\n")

    # Lance le pipeline (multi-fichier géré dans run_pipeline)
    run_pipeline(raw_dir=args.pdf, max_themes=args.max_themes, out_dir=args.out)

if __name__ == "__main__":
    main()

