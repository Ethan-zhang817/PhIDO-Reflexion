"""Backward-compatible re-exports for :mod:`PhotonicsAI.graph.llm`.

New code should import from :mod:`PhotonicsAI.graph.llm` directly.
"""

from __future__ import annotations

from PhotonicsAI.graph.llm import (
    LLMCallError,
    LLMRequest,
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
    "TextLLMRequest",
    "ainvoke_llm",
    "ainvoke_text",
    "make_llm_runnable",
    "make_text_runnable",
    "invoke_llm",
    "invoke_text",
]
