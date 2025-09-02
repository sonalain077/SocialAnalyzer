"""state.py
Définition des types d'état pour le graphe LangGraph.
"""

from __future__ import annotations
from groq import Groq
import os
from typing import Any, Dict, List, Annotated, TypedDict
from typing_extensions import TypedDict as TypedDictExt
from operator import add
from dotenv import load_dotenv

__all__ = [
    "InputState",
    "NodeState",
    "OutputState",
    "GraphState"
]

# Types pour les entrées du graphe
class InputState(TypedDict):
    pdf_path: str
    max_themes: int
    model_name: str
    temperature: float

# Types pour l'état interne des nœuds
class NodeState(TypedDict):
    raw_text: str
    clean_text: str
    segments: List[str]
    coded_segments: List[List[Dict[str, Any]]]
    validated_segments: List[List[Dict[str, Any]]]
    all_codes: List[Dict[str, Any]]
    clusters: Dict[int, List[Dict[str, Any]]]
    code_to_cluster: Dict[str, int]
    theme_labels: Dict[int, str]

# Types pour les sorties du graphe
class OutputState(TypedDict):
    final_report: str
    output_files: Dict[str, str]

# État global du graphe
class GraphState(TypedDictExt, total=False):
    pdf_path: str

    max_themes: int
    raw_text: str
    clean_text: str
    segments: Annotated[List[str], add]
    coded_segments: Annotated[List[List[Dict[str, Any]]], add]
    validated_segments: Annotated[List[List[Dict[str, Any]]], add]
    all_codes: Annotated[List[Dict[str, Any]], add]
    clusters: Dict[int, List[Dict[str, Any]]]
    code_to_cluster: Dict[str, int]
    theme_labels: Dict[int, str]
    final_report: str
    rapport_global: str
    rapport_global_refined: str
    output_files: Dict[str, str]
    model_name: str
    temperature: float
    api_key: str
    logs: List[str]
    meta_theme_labels: Dict[int, str]  
    theme_to_meta: Dict[int, int]  
    global_synthesis: str
    files_processed: int 
    num_files: int

def ensure_api_key(state: GraphState) -> str:
    if state.get('api_key'):
        return state['api_key']
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY manquante dans les variables d'environnement")
    state['api_key'] = key
    return key
