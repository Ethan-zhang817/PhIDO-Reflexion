"""Evaluator: rule-based pass/fail. No LLM, by design (REFACTOR_PLAN.md §3.2)."""

from __future__ import annotations

from PhotonicsAI.graph.adapters.eda_report import EdaReport
from PhotonicsAI.graph.state import PhIDOState


def evaluator_verdict(report: EdaReport, *, require_drc_pass: bool = True) -> str:
    """``pass`` / ``fail`` for routing; ``require_drc_pass=False`` ignores DRC
    violation count (GDS + SAX 仍须成功)."""
    gds = report.get_gds()
    sax = report.get_sax()
    drc = report.get_drc()

    if not gds.ok:
        return "fail"
    if not sax.ok or sax.missing_models:
        return "fail"
    if not require_drc_pass:
        return "pass"
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
    return {"timings": {**(state.get("timings") or {}), "evaluator_verdict": 0.0}}


def route_after_evaluator(state: PhIDOState) -> str:
    """Conditional edge: ``pass`` -> END, ``fail`` -> reflector."""
    report = state.get("eda_report")
    if report is None:
        return "fail"
    if isinstance(report, dict):
        report = EdaReport.model_validate(report)
    require = state.get("require_drc_pass", True)
    return evaluator_verdict(report, require_drc_pass=require)
