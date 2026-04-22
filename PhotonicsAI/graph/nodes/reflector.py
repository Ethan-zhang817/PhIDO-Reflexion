"""Reflector node: produce a verbal critique, NEVER a DSL.

Per Shinn et al. 2023 (Reflexion), the Reflector must be strictly
separated from the Actor (Designer): no shared chat history, and the
Reflector cannot directly modify the artifact. Its sole output is a
short natural-language critique that is appended to ``state.reflections``
and surfaced at the top of the next Designer prompt.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from PhotonicsAI.graph.adapters.eda_report import summarize_report
from PhotonicsAI.graph.llm_runnables import LLMCallError, invoke_llm
from PhotonicsAI.graph.nodes._pdk_context import cells_short
from PhotonicsAI.graph.state import PhIDOState

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reflector.txt"

_ESCALATE_TOKEN = "escalate to human"


def _format_past_reflections(state: PhIDOState, limit: int = 3) -> str:
    reflections = state.get("reflections") or []
    if not reflections:
        return "(none)"
    recent = reflections[-limit:][::-1]
    return "\n".join(f"{i + 1}. {r.strip()}" for i, r in enumerate(recent))


def _format_dsl(state: PhIDOState) -> str:
    dsl = state.get("circuit_dsl")
    if not dsl:
        return "(no DSL produced this round)"
    return yaml.dump(dsl, sort_keys=False, default_flow_style=False)


def _format_summary(state: PhIDOState) -> str:
    report = state.get("eda_report")
    if report is None:
        return "(no EDA report this round)"
    return summarize_report(report)


def reflector_node(state: PhIDOState) -> dict:
    """Append a new reflection and bump the retry counter.

    The conditional edge (:func:`route_after_reflector`) decides whether
    the loop should continue or escalate.
    """
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        user_prompt=state.get("user_prompt", ""),
        current_circuit_dsl=_format_dsl(state),
        eda_summary=_format_summary(state),
        available_cells_short=cells_short(),
        past_reflections=_format_past_reflections(state),
    )

    model = state.get("reflector_model") or "claude-3-7-sonnet-20250219"
    try:
        critique = invoke_llm(prompt, system_prompt="", model=model).strip()
    except LLMCallError as exc:
        critique = (
            f"[reflector] LLM call failed: {exc}. Falling back to a generic "
            "hint: tighten device parameters and re-route to satisfy minimum "
            "spacing. If this repeats, escalate to human."
        )

    if not critique:
        critique = "(reflector returned empty output; escalate to human)"

    return {
        "reflections": [critique],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_reflector(state: PhIDOState) -> str:
    """``retry`` while under the cap and not escalating; ``escalate`` otherwise."""
    retries = state.get("retry_count", 0)
    cap = state.get("max_retries", 3)
    reflections = state.get("reflections") or []
    last = reflections[-1].lower() if reflections else ""

    if _ESCALATE_TOKEN in last:
        return "escalate"
    if retries >= cap:
        return "escalate"
    return "retry"
