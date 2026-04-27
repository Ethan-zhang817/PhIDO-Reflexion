"""Model id presets for the graph Streamlit UI (sidebar selectboxes).

These strings are passed verbatim to :func:`PhotonicsAI.Photon.llm_api.call_llm`.
When using an OpenAI-compatible proxy (e.g. set ``OPENAI_BASE_URL`` in ``.env`` to
a vendor such as ``https://api.gptsapi.net/v1``), you must pick a model name that
the upstream exposes for your key; the UI cannot enumerate remote catalogs.
"""

from __future__ import annotations

# Chat Completions (``call_openai``) — gpt-*, chatgpt-*, and similar.
_OPENAI_CHAT: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4",
    "gpt-3.5-turbo",
    "chatgpt-4o-latest",
)

# Reasoning (``call_openai_reasoning``) — o1 / o3 / o4 family.
_OPENAI_REASONING: tuple[str, ...] = (
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "o1-2024-12-17",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
    "o4",
)

# Non-OpenAI providers (unchanged from previous UI).
_DESIGNER_OTHER: tuple[str, ...] = (
    "claude-3-7-sonnet-20250219",
    "deepseek-reasoner",
    "gemini-2.5-pro",
)
_REFLECTOR_OTHER: tuple[str, ...] = (
    "claude-3-7-sonnet-20250219",
    "deepseek-reasoner",
)

DESIGNER_MODEL_CHOICES: list[str] = list(
    dict.fromkeys(  # stable dedupe, preserve order
        (*_OPENAI_CHAT, *_OPENAI_REASONING, *_DESIGNER_OTHER)
    )
)
REFLECTOR_MODEL_CHOICES: list[str] = list(
    dict.fromkeys((*_OPENAI_CHAT, *_OPENAI_REASONING, *_REFLECTOR_OTHER))
)
