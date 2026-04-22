"""Designer node: produce / revise the circuit DSL.

The node never sees the Reflector's chat history (Reflexion paper §10.2:
keep Actor and Reflector contexts separated). It only sees:

* the original user prompt,
* the most recent reflection (if any),
* the previous DSL (if any), as a starting point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from PhotonicsAI.graph.llm_runnables import LLMCallError, invoke_llm
from PhotonicsAI.graph.nodes._pdk_context import cells_block
from PhotonicsAI.graph.state import PhIDOState

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "designer.txt"


def _load_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _seed_section(state: PhIDOState) -> str:
    seed = state.get("circuit_dsl") or state.get("seed_circuit_dsl")
    if not seed:
        return ""
    return (
        "== Previous DSL (revise this rather than restart) ==\n"
        + yaml.dump(seed, sort_keys=False, default_flow_style=False)
    )


def _reflection_section(state: PhIDOState) -> str:
    reflections = state.get("reflections") or []
    if not reflections:
        return ""
    latest = reflections[-1].strip()
    return (
        "== Latest reflection (MUST follow) ==\n"
        f"{latest}\n"
    )


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_dsl(raw: str) -> dict | None:
    try:
        loaded: Any = yaml.safe_load(_strip_code_fences(raw))
    except yaml.YAMLError:
        return None
    return loaded if isinstance(loaded, dict) else None


def designer_node(state: PhIDOState) -> dict:
    """LangGraph node entry point.

    Returns a partial state update with the new ``circuit_dsl`` (and
    optionally an extra reflection if the LLM call fails outright, so
    the next iteration has *something* to react to).
    """
    template = _load_template()
    prompt = template.format(
        user_prompt=state.get("user_prompt", ""),
        seed_section=_seed_section(state),
        available_cells=cells_block(),
        reflection_section=_reflection_section(state),
    )

    model = state.get("designer_model") or "o1"
    try:
        raw = invoke_llm(prompt, system_prompt="", model=model)
    except LLMCallError as exc:
        return {
            "reflections": [
                f"[designer] LLM call failed ({exc}); please retry with smaller prompt."
            ],
        }

    dsl = _parse_dsl(raw)
    if dsl is None:
        return {
            "reflections": [
                "[designer] Output was not valid YAML or was not a mapping; "
                "next round must produce a single YAML document with `nodes` and `edges`."
            ],
        }
    return {"circuit_dsl": dsl}
