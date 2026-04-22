"""Global ``PhIDOState`` for the LangGraph Reflexion PoC.

The state schema is intentionally flat. The ``reflections`` field uses
LangGraph's ``Annotated[..., add]`` reducer so that each Reflector node
can simply ``return {"reflections": [new_reflection]}`` and rely on
LangGraph to append rather than overwrite.

Other fields use the default "last write wins" reducer.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Optional, TypedDict

from PhotonicsAI.graph.adapters.eda_report import EdaReport


DEFAULT_DESIGNER_MODEL = "o1"
DEFAULT_REFLECTOR_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_MAX_RETRIES = 3


class PhIDOState(TypedDict, total=False):
    """LangGraph state for the Layout-DRC Reflexion subgraph."""

    # ---- Input -------------------------------------------------------
    user_prompt: str
    """Original natural-language design intent."""

    seed_circuit_dsl: Optional[dict]
    """Optional starting DSL — if provided, the Designer treats it as a
    base to iterate on rather than synthesizing from scratch."""

    # ---- Designer / current artifact ---------------------------------
    circuit_dsl: Optional[dict]
    """DSL the Designer most recently produced (single-source-of-truth)."""

    gf_netlist: Optional[dict]
    """Cached gdsfactory netlist (derived from ``circuit_dsl``)."""

    gds_path: Optional[str]
    """Path to the most recently written GDS file."""

    sax_result: Optional[Any]
    """Latest SAX simulation output (wavelength sweep dict)."""

    # ---- EDA-Reflexion -----------------------------------------------
    eda_report: Optional[EdaReport]
    """Structured EDA report from the most recent executor run."""

    reflections: Annotated[list[str], add]
    """Verbal critiques accumulated across rounds. The Designer prompt
    only uses the latest entry; the full list is kept for transparency
    and possible future few-shot use."""

    retry_count: int
    """Number of Reflector -> Designer retries already performed."""

    max_retries: int
    """Hard cap before escalating to a human."""

    escalated: bool
    """Set by the human-escalation node when a human takes over."""

    human_input: Optional[dict]
    """Payload provided by the user when resuming after escalation."""

    # ---- Configuration ----------------------------------------------
    designer_model: str
    reflector_model: str

    # ---- Observability ----------------------------------------------
    thread_id: str
    timings: dict[str, float]
    token_usage: dict[str, int]


def initial_state(
    user_prompt: str,
    *,
    seed_circuit_dsl: Optional[dict] = None,
    designer_model: str = DEFAULT_DESIGNER_MODEL,
    reflector_model: str = DEFAULT_REFLECTOR_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    thread_id: str = "",
) -> PhIDOState:
    """Return a fully-populated ``PhIDOState`` ready to feed into the graph."""
    return PhIDOState(
        user_prompt=user_prompt,
        seed_circuit_dsl=seed_circuit_dsl,
        circuit_dsl=None,
        gf_netlist=None,
        gds_path=None,
        sax_result=None,
        eda_report=None,
        reflections=[],
        retry_count=0,
        max_retries=max_retries,
        escalated=False,
        human_input=None,
        designer_model=designer_model,
        reflector_model=reflector_model,
        thread_id=thread_id,
        timings={},
        token_usage={},
    )
