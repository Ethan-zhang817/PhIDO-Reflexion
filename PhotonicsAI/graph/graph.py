"""Graph wiring for the PhIDO Reflexion PoC.

Topology::

    START -> entity_extraction -> component_selection -> schematic_generation
        -> eda_executor -> evaluator
        evaluator --pass--> END
        evaluator --fail--> reflector
            reflector --retry<N--> designer
            designer -> eda_executor
            reflector --retry>=N or "escalate to human"--> human_escalation
                human_escalation -- override/hint --> designer
                human_escalation -- abort        --> END
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langgraph.graph import END, START, StateGraph

from PhotonicsAI.config import PATH
from PhotonicsAI.graph.nodes import (
    component_selection_node,
    designer_node,
    eda_executor_node,
    entity_extraction_node,
    evaluator_node,
    human_escalation_node,
    reflector_node,
    route_after_component_selection,
    route_after_entity_extraction,
    route_after_evaluator,
    route_after_reflector,
    route_after_schematic_generation,
    schematic_generation_node,
)
from PhotonicsAI.graph.state import PhIDOState
from PhotonicsAI.graph.tracing import maybe_enable_langsmith


_DEFAULT_CHECKPOINT_DB = PATH.build / "phido_checkpoints.db"


def build_graph() -> StateGraph:
    """Construct the (uncompiled) ``StateGraph``."""
    g: StateGraph = StateGraph(PhIDOState)

    g.add_node("entity_extraction", entity_extraction_node)
    g.add_node("component_selection", component_selection_node)
    g.add_node("schematic_generation", schematic_generation_node)
    g.add_node("designer", designer_node)
    g.add_node("eda_executor", eda_executor_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("reflector", reflector_node)
    g.add_node("human_escalation", human_escalation_node)

    g.add_edge(START, "entity_extraction")
    g.add_conditional_edges(
        "entity_extraction",
        route_after_entity_extraction,
        {"ready": "component_selection", "fail": "human_escalation"},
    )
    g.add_conditional_edges(
        "component_selection",
        route_after_component_selection,
        {"ready": "schematic_generation", "fail": "human_escalation"},
    )
    g.add_conditional_edges(
        "schematic_generation",
        route_after_schematic_generation,
        {"ready": "eda_executor", "fail": "human_escalation"},
    )
    g.add_edge("eda_executor", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {"pass": END, "fail": "reflector"},
    )
    g.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {"retry": "designer", "escalate": "human_escalation"},
    )
    g.add_edge("designer", "eda_executor")
    g.add_conditional_edges(
        "human_escalation",
        _route_after_escalation,
        {"resume": "designer", "abort": END},
    )

    return g


def _route_after_escalation(state: PhIDOState) -> str:
    """Route the human escalation node back to designer or to END.

    If the human supplied an override/hint, the escalation node sets
    ``escalated=False``; otherwise we abort the run.
    """
    return "abort" if state.get("escalated") else "resume"


def _build_checkpointer(db_path: Optional[Path]):
    """Try to construct a SqliteSaver; fall back to in-memory on failure."""
    if db_path is None:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path.parent.mkdir(parents=True, exist_ok=True)
        # SqliteSaver in modern LangGraph exposes a context manager via
        # ``from_conn_string`` *and* a sync constructor accepting an
        # already-opened connection. We use the connection-based path so
        # the saver outlives the function scope.
        import sqlite3

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def get_compiled_graph(
    *,
    checkpoint_db: Optional[Path] = _DEFAULT_CHECKPOINT_DB,
    enable_tracing: bool = True,
):
    """Return a compiled graph with the requested checkpointer.

    The result is suitable for caching with ``@st.cache_resource`` in the
    Streamlit entry point.
    """
    if enable_tracing:
        maybe_enable_langsmith()
    checkpointer = _build_checkpointer(checkpoint_db)
    return build_graph().compile(checkpointer=checkpointer)
