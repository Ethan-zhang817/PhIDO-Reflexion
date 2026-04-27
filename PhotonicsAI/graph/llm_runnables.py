"""Path B: wrap the existing ``llm_api.call_llm`` in ``RunnableLambda``.

Rationale (REFACTOR_PLAN.md §7): keeping the legacy multi-provider
dispatcher avoids re-implementing the Anthropic ``thinking`` block,
DeepSeek truncation, and tiktoken-based token counting. We only need
LangGraph-friendly callables, not full LangChain ``ChatModel`` parity.

A node calls ``invoke_llm(prompt, system_prompt, model)`` and gets back
a string. Errors are caught and converted to a sentinel ``LLMCallError``
string so the graph keeps making forward progress (the next Reflector
turn can then react to it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda

from PhotonicsAI.Photon import llm_api


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str = ""
    model: str = "gpt-4o-mini"


class LLMCallError(RuntimeError):
    """Raised when even the resilient legacy ``call_llm`` fails."""


def _invoke(req: LLMRequest) -> str:
    try:
        result = llm_api.call_llm(
            req.prompt, req.system_prompt, req.model
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMCallError(f"call_llm({req.model}) failed: {exc}") from exc
    if result is None:
        raise LLMCallError(
            f"call_llm({req.model}) returned None (unsupported provider?)"
        )
    return str(result)


def make_llm_runnable(model: str) -> RunnableLambda:
    """Bind a model name and return a ``RunnableLambda`` taking ``LLMRequest``.

    The bound model can still be overridden per-call by passing a
    different ``LLMRequest.model``.
    """

    def _call(req: LLMRequest | dict) -> str:
        if isinstance(req, dict):
            req = LLMRequest(**req)
        if not req.model:
            req.model = model
        return _invoke(req)

    return RunnableLambda(_call).with_config({"run_name": f"llm:{model}"})


def invoke_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-4o-mini",
    **_: Any,
) -> str:
    """Convenience helper used directly by node modules."""
    return _invoke(LLMRequest(prompt=prompt, system_prompt=system_prompt, model=model))
