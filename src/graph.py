"""graph.py
Construit le LangGraph en reliant les nœuds et expose la fonction build_graph().
Le state est importé directement depuis `state.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from langgraph.graph import END, START, StateGraph

from src.state import GraphState

if TYPE_CHECKING:
    from langgraph import CompiledGraph  # type: ignore

def should_generate_corpus_synthese(state: GraphState) -> bool:
    return state.get("files_processed", 0) >= state.get("num_files", 1)


def build_graph() -> "CompiledGraph[GraphState]":
    """Construit et compile le graph LangGraph"""

    graph = StateGraph(GraphState)

    # --- Déclaration des nœuds ----------------------------------------------------
    from src.nodes import (
        extract_node,
        clean_node,
        segment_node,
        analyze_node,
        judge_node,
        cluster_node,
        label_node,  
        meta_cluster_node,
        compile_node,
        report_node,
        synthese_corpus_node,
        refine_synthese_node
    )
    

    graph.add_node("extract", extract_node)
    graph.add_node("clean", clean_node)
    graph.add_node("segment", segment_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("judge", judge_node)
    graph.add_node("cluster", cluster_node)
    graph.add_node("label", label_node)
    graph.add_node("meta_cluster", meta_cluster_node)
    graph.add_node("compile", compile_node)
    graph.add_node("rapport", report_node)
    graph.add_node("synthese", synthese_corpus_node)
    graph.add_node("refine", refine_synthese_node)

    # --- Connexions ---------------------------------------------------------------

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "clean")
    graph.add_edge("clean", "segment")
    graph.add_edge("segment", "analyze")
    graph.add_edge("analyze", "judge")
    graph.add_edge("judge", "cluster")
    graph.add_edge("cluster", "label")
    graph.add_edge("label", "meta_cluster")
    graph.add_edge("meta_cluster", "compile")
    graph.add_edge("compile", "rapport")
    graph.add_edge("synthese", "refine") 
    graph.add_edge("refine", END)

    # Ajout de la condition pour la synthèse du corpus
    graph.add_conditional_edges(
        "rapport",
        should_generate_corpus_synthese,
        {
            True: "synthese",  
            False: "__end__",
        },
    )


    # Compilation du graph
    return graph.compile()
