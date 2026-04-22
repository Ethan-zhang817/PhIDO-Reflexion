"""LangGraph nodes for the Reflexion PoC.

Lazy re-exports so that importing ``PhotonicsAI.graph.adapters`` does
not require ``langgraph`` to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "designer_node",
    "eda_executor_node",
    "evaluator_node",
    "human_escalation_node",
    "reflector_node",
    "route_after_evaluator",
    "route_after_reflector",
]

if TYPE_CHECKING:  # pragma: no cover
    from PhotonicsAI.graph.nodes.designer import designer_node
    from PhotonicsAI.graph.nodes.eda_executor import eda_executor_node
    from PhotonicsAI.graph.nodes.evaluator import evaluator_node, route_after_evaluator
    from PhotonicsAI.graph.nodes.human_escalation import human_escalation_node
    from PhotonicsAI.graph.nodes.reflector import reflector_node, route_after_reflector


def __getattr__(name: str):
    if name == "designer_node":
        from PhotonicsAI.graph.nodes.designer import designer_node

        return designer_node
    if name == "eda_executor_node":
        from PhotonicsAI.graph.nodes.eda_executor import eda_executor_node

        return eda_executor_node
    if name == "evaluator_node":
        from PhotonicsAI.graph.nodes.evaluator import evaluator_node

        return evaluator_node
    if name == "route_after_evaluator":
        from PhotonicsAI.graph.nodes.evaluator import route_after_evaluator

        return route_after_evaluator
    if name == "human_escalation_node":
        from PhotonicsAI.graph.nodes.human_escalation import human_escalation_node

        return human_escalation_node
    if name == "reflector_node":
        from PhotonicsAI.graph.nodes.reflector import reflector_node

        return reflector_node
    if name == "route_after_reflector":
        from PhotonicsAI.graph.nodes.reflector import route_after_reflector

        return route_after_reflector
    raise AttributeError(f"module 'PhotonicsAI.graph.nodes' has no attribute {name!r}")
