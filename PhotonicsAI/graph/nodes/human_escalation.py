"""Human-in-the-loop escalation node.

Uses LangGraph's ``interrupt`` primitive when available so a Streamlit
UI can pause the run, collect a human override (custom DSL, a hint
string, or simply "give up"), and resume with ``Command(resume=...)``.
"""

from __future__ import annotations

from typing import Any

try:  # langgraph >= 0.2.30 exposes interrupt at the top level
    from langgraph.types import interrupt
except ImportError:  # pragma: no cover - older langgraph
    try:
        from langgraph.graph import interrupt  # type: ignore[attr-defined]
    except ImportError:
        interrupt = None  # type: ignore[assignment]

from PhotonicsAI.graph.state import PhIDOState


def human_escalation_node(state: PhIDOState) -> dict:
    """Pause the graph, returning whatever the human supplies on resume.

    The human payload is expected to be a dict like::

        {"action": "override_dsl", "dsl": {...}}
        # or
        {"action": "hint", "text": "use mzi_2x2_..."}
        # or
        {"action": "abort"}

    For the abort case, the node simply marks ``escalated=True`` and the
    Designer next round will see no fresh reflection — the conditional
    edge in :mod:`PhotonicsAI.graph.graph` is responsible for routing
    such a state to END.
    """
    payload: Any = None
    if interrupt is not None:
        try:
            payload = interrupt(
                {
                    "reason": "max_retries_reached",
                    "retry_count": state.get("retry_count", 0),
                    "last_reflection": (state.get("reflections") or ["(none)"])[-1],
                    "eda_summary": (
                        state["eda_report"].summary_text
                        if state.get("eda_report") is not None
                        else ""
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            payload = None

    update: dict = {"escalated": True, "human_input": payload}

    if isinstance(payload, dict):
        action = payload.get("action")
        if action == "override_dsl" and isinstance(payload.get("dsl"), dict):
            update["circuit_dsl"] = payload["dsl"]
            update["retry_count"] = 0
            update["escalated"] = False
        elif action == "hint" and payload.get("text"):
            update["reflections"] = [f"[human] {payload['text']}"]
            update["retry_count"] = 0
            update["escalated"] = False

    return update
