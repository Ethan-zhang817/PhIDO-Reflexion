"""LangGraph-oriented text LLM invocations (LangChain :class:`Runnable` wrappers).

Nodes call :func:`invoke_text` (or the legacy name :func:`invoke_llm`) for
string completions, routed through the existing :func:`PhotonicsAI.Photon.llm_api.call_llm`
so multi-provider behavior (OpenAI, Anthropic, o-series, etc.) stays centralized.

Async variants use :func:`asyncio.to_thread` so they work with async LangGraph
nodes without blocking the event loop for CPU-bound work; the actual HTTP calls
remain synchronous inside ``call_llm``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda

from PhotonicsAI.Photon import llm_api

_DEFAULT_TAGS: tuple[str, ...] = ("graph", "llm", "text")


@dataclass
class TextLLMRequest:
    """Input to the graph text-completion Runnable (chat-style, string out)."""

    prompt: str
    system_prompt: str = ""
    model: str = "gpt-4o-mini"


class TextLLMCallError(RuntimeError):
    """Raised when :func:`PhotonicsAI.Photon.llm_api.call_llm` fails."""


# Backward-compatible names used across nodes and tests
LLMRequest = TextLLMRequest
LLMCallError = TextLLMCallError


def _invoke_text(req: TextLLMRequest) -> str:
    try:
        result = llm_api.call_llm(
            req.prompt, req.system_prompt, req.model
        )
    except Exception as exc:  # noqa: BLE001
        raise TextLLMCallError(
            f"call_llm({req.model}) failed: {exc}"
        ) from exc
    if result is None:
        raise TextLLMCallError(
            f"call_llm({req.model}) returned None (unsupported provider?)"
        )
    return str(result)


def invoke_text(
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-4o-mini",
    **_: Any,
) -> str:
    """Synchronous string completion. Prefer this from graph nodes."""
    return _invoke_text(
        TextLLMRequest(
            prompt=prompt, system_prompt=system_prompt, model=model
        )
    )


# Legacy alias (nodes, tests, REFACTOR_PLAN)
invoke_llm = invoke_text


async def ainvoke_text(
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-4o-mini",
) -> str:
    """Async string completion (wraps the sync :func:`call_llm` in a thread)."""
    return await asyncio.to_thread(
        invoke_text, prompt, system_prompt, model
    )


# Legacy
ainvoke_llm = ainvoke_text


def make_text_runnable(
    default_model: str, *, run_name: str | None = None
) -> RunnableLambda:
    """Return a :class:`~langchain_core.runnables.Runnable` bound to a default model.

    Accepts a :class:`TextLLMRequest` or a dict with the same fields. The bound
    ``default_model`` applies when ``model`` is empty.
    """
    if run_name is None:
        run_name = f"graph:text:{default_model}"

    def _call(req: TextLLMRequest | dict) -> str:
        if isinstance(req, dict):
            r = TextLLMRequest(**req)
        else:
            r = req
        if not r.model:
            r = replace(r, model=default_model)
        return _invoke_text(r)

    async def _acall(req: TextLLMRequest | dict) -> str:
        if isinstance(req, dict):
            r = TextLLMRequest(**req)
        else:
            r = req
        if not r.model:
            r = replace(r, model=default_model)
        return await asyncio.to_thread(_invoke_text, r)

    cfg = RunnableConfig(
        run_name=run_name,
        tags=[*_DEFAULT_TAGS, "runnable"],
    )
    return RunnableLambda(_call, afunc=_acall).with_config(cfg)


# Legacy name
make_llm_runnable = make_text_runnable
