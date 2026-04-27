"""OpenAI *structured output* (Pydantic) via :class:`Runnable` for LangGraph.

Uses the OpenAI SDK ``chat.completions.parse`` with a Pydantic model — the same
mechanism as the classic PhIDO ``llm_api.callgpt_pydantic`` but exposed as an
invocation request type and a :class:`langchain_core.runnables.RunnableLambda`
for composition, tracing, and future graph-level batching.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import TypeVar

from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from PhotonicsAI.Photon import llm_api

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_DEFAULT_TAGS: tuple[str, ...] = ("graph", "llm", "structured", "openai")


@dataclass
class StructuredOpenAIRequest:
    """Single structured parse call: system + user message → Pydantic model."""

    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]
    model: str
    run_name: str = "graph:openai:structured"


class StructuredOutputError(RuntimeError):
    """Raised when the OpenAI parse call fails or returns an unusable result."""


def invoke_structured_openai(req: StructuredOpenAIRequest) -> BaseModel:
    """Synchronous structured completion; returns the parsed Pydantic instance."""
    client = llm_api._openai_client()
    try:
        completion = client.beta.chat.completions.parse(
            model=req.model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            response_format=req.response_model,
        )
    except Exception as exc:  # noqa: BLE001
        raise StructuredOutputError(
            f"openai parse ({req.model}, {req.response_model.__name__}): {exc}"
        ) from exc

    message = completion.choices[0].message
    if message.parsed is not None:
        return message.parsed
    logger.warning("Structured parse refusal: %s", getattr(message, "refusal", ""))
    raise StructuredOutputError(
        f"openai returned no parsed object for {req.response_model.__name__}"
    )


async def ainvoke_structured_openai(
    req: StructuredOpenAIRequest,
) -> BaseModel:
    """Async structured completion (thread offload)."""
    return await asyncio.to_thread(invoke_structured_openai, req)


def make_structured_openai_runnable(
    default_model: str, *, run_name: str | None = None
) -> RunnableLambda:
    """Return a :class:`Runnable` that accepts :class:`StructuredOpenAIRequest`.

    When ``model`` is empty on the request, ``default_model`` is used.
    """
    if run_name is None:
        run_name = f"graph:structured:{default_model}"

    def _call(req: StructuredOpenAIRequest | dict) -> BaseModel:
        r = _coerce_request(req, default_model)
        return invoke_structured_openai(r)

    async def _acall(req: StructuredOpenAIRequest | dict) -> BaseModel:
        r = _coerce_request(req, default_model)
        return await ainvoke_structured_openai(r)

    cfg = RunnableConfig(
        run_name=run_name,
        tags=[*_DEFAULT_TAGS, "runnable"],
    )
    return RunnableLambda(_call, afunc=_acall).with_config(cfg)


def _coerce_request(
    req: StructuredOpenAIRequest | dict, default_model: str
) -> StructuredOpenAIRequest:
    if isinstance(req, dict):
        r = StructuredOpenAIRequest(**req)
    else:
        r = req
    if not r.model:
        r = replace(r, model=default_model)
    return r
