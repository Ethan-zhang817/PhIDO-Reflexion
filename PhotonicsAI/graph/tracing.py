"""Optional LangSmith tracing wiring.

Tracing is **only** enabled when ``LANGSMITH_API_KEY`` is set in the
environment (or .env). Otherwise this module is a no-op so the graph
still runs in dev / CI without external dependencies.
"""

from __future__ import annotations

import os

_DEFAULT_PROJECT = "phido-reflexion"


def maybe_enable_langsmith(project: str | None = None) -> bool:
    """Enable LangSmith tracing if an API key is present.

    Returns True when tracing was activated, False otherwise. We set the
    environment variables LangSmith looks for at call time rather than
    importing the SDK, because the SDK auto-configures itself based on
    those vars.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
        "LANGCHAIN_API_KEY"
    )
    if not api_key:
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project or _DEFAULT_PROJECT)
    return True
