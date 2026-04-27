"""LangGraph-style LLM API for the PhIDO graph package.

* **Text** — :class:`TextLLMRequest` + :func:`invoke_text` + :func:`make_text_runnable`
  (wraps :func:`PhotonicsAI.Photon.llm_api.call_llm` for multi-provider chat).

* **Structured (OpenAI parse)** — :class:`StructuredOpenAIRequest` +
  :func:`invoke_structured_openai` + :func:`make_structured_openai_runnable` for
  Pydantic-typed responses (legacy EE/CS, etc.).

All callables are :class:`langchain_core.runnables.Runnable`-compatible
(``invoke`` / ``ainvoke``, ``config``/tags for LangSmith). Prefer importing from
this package in new code; :mod:`PhotonicsAI.graph.llm_runnables` re-exports the
legacy names.
"""

from __future__ import annotations

from PhotonicsAI.graph.llm.structured import (
    StructuredOpenAIRequest,
    StructuredOutputError,
    ainvoke_structured_openai,
    invoke_structured_openai,
    make_structured_openai_runnable,
)
from PhotonicsAI.graph.llm.text import (
    LLMCallError,
    LLMRequest,
    TextLLMCallError,
    TextLLMRequest,
    ainvoke_llm,
    ainvoke_text,
    make_llm_runnable,
    make_text_runnable,
    invoke_llm,
    invoke_text,
)

__all__ = [
    "LLMCallError",
    "LLMRequest",
    "StructuredOpenAIRequest",
    "StructuredOutputError",
    "TextLLMCallError",
    "TextLLMRequest",
    "ainvoke_llm",
    "ainvoke_structured_openai",
    "ainvoke_text",
    "invoke_llm",
    "invoke_structured_openai",
    "invoke_text",
    "make_llm_runnable",
    "make_structured_openai_runnable",
    "make_text_runnable",
]
