"""Lightweight PDK context shared by Designer and Reflector prompts.

Loading the design library is moderately expensive (it walks
``KnowledgeBase/DesignLibrary`` and parses every file's docstring), so
we cache the result at module level. Importing this module also avoids
forcing every test file to know how the legacy utility is invoked.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def list_available_cells() -> list[str]:
    """Return the sorted list of cell names exposed by ``DesignLibrary``."""
    from PhotonicsAI.Photon import utils as photon_utils

    docs = photon_utils.search_directory_for_docstrings()
    return sorted({d["module_name"] for d in docs})


@lru_cache(maxsize=1)
def cell_one_liners() -> dict[str, str]:
    """Return ``{cell_name: first_line_of_docstring}`` for prompt context."""
    from PhotonicsAI.Photon import utils as photon_utils

    summary: dict[str, str] = {}
    for d in photon_utils.search_directory_for_docstrings():
        name = d["module_name"]
        doc = (d.get("docstring") or "").strip().splitlines()
        summary[name] = doc[0].strip() if doc else ""
    return summary


def cells_block(max_cells: int = 60) -> str:
    """Format the cell list as a single multi-line string for the prompt.

    Capped at ``max_cells`` entries to avoid blowing the context window
    when the library grows. The Reflector gets a name-only short list;
    the Designer gets one-liner descriptions.
    """
    summary = cell_one_liners()
    items = list(summary.items())[:max_cells]
    return "\n".join(f"- {name}: {one_liner}" for name, one_liner in items)


def cells_short(max_cells: int = 60) -> str:
    """Return a comma-separated cell name list for the Reflector prompt."""
    names = list_available_cells()[:max_cells]
    return ", ".join(names)
