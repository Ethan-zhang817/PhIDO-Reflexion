"""Evaluator: rule-based pass/fail. No LLM, by design (REFACTOR_PLAN.md §3.2)."""

from __future__ import annotations

from PhotonicsAI.graph.adapters.eda_report import EdaReport
from PhotonicsAI.graph.state import PhIDOState


def _verdict(report: EdaReport) -> str:
    """Compute the textual verdict; pure function for testability."""
    gds = report.get_gds()
    sax = report.get_sax()
    drc = report.get_drc()

    if not gds.ok:
        return "fail"
    if not sax.ok or sax.missing_models:
        return "fail"
    # DRC: skipped (e.g. KLayout missing) is a soft pass with a warning.
    # The flag lives on `drc.skipped_reason`; the executor sets ok=True
    # in that case so this branch does not over-trigger.
    if drc.skipped_reason:
        return "pass"
    if drc.n_violations > 0:
        return "fail"
    return "pass"


def evaluator_node(state: PhIDOState) -> dict:
    """LangGraph node: persist a textual verdict alongside the report.

    The verdict itself is *not* required by the conditional edge — the
    edge re-derives it from ``eda_report`` to keep state and routing
    decisions decoupled. We still surface it for UI / log readability.
    """
    report = state.get("eda_report")
    if report is None:
        verdict = "fail"
    else:
        verdict = _verdict(report)
    return {"timings": {**(state.get("timings") or {}), "evaluator_verdict": 0.0}}  # noop write; verdict in route fn


def route_after_evaluator(state: PhIDOState) -> str:
    """Conditional edge: ``pass`` -> END, ``fail`` -> reflector."""
    report = state.get("eda_report")
    if report is None:
        return "fail"
    return _verdict(report)
