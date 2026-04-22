"""LangGraph + Reflexion PoC for PhIDO.

This subpackage contains the experimental Reflexion-loop agent
(Designer -> EDA Executor -> Evaluator -> Reflector). It coexists with
the legacy ``PhotonicsAI.Photon`` Streamlit workflow and is intentionally
kept side-by-side so that it can be removed by deleting this directory.

Top-level imports are lazy: the adapters and state schema can be used
without ``langgraph`` installed (handy for unit-testing the structured
EDA reports). Importing ``build_graph`` / ``get_compiled_graph`` does
require ``langgraph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "build_graph",
    "get_compiled_graph",
    "PhIDOState",
]

if TYPE_CHECKING:  # pragma: no cover
    from PhotonicsAI.graph.graph import build_graph, get_compiled_graph
    from PhotonicsAI.graph.state import PhIDOState


def __getattr__(name: str):
    if name in {"build_graph", "get_compiled_graph"}:
        from PhotonicsAI.graph import graph as _graph_module

        return getattr(_graph_module, name)
    if name == "PhIDOState":
        from PhotonicsAI.graph import state as _state_module

        return _state_module.PhIDOState
    raise AttributeError(f"module 'PhotonicsAI.graph' has no attribute {name!r}")
