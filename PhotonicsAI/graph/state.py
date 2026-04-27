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


DEFAULT_DESIGNER_MODEL = "gpt-4o-mini"
DEFAULT_REFLECTOR_MODEL = "gpt-4o-mini"
DEFAULT_LEGACY_STAGE_MODEL = "gpt-4o"
DEFAULT_MAX_RETRIES = 3


class PhIDOState(TypedDict, total=False):
    """LangGraph state for the Layout-DRC Reflexion subgraph."""

    # ---- Input -------------------------------------------------------
    user_prompt: str
    """Original natural-language design intent."""

    seed_circuit_dsl: Optional[dict]
    """Optional starting DSL — if provided, the Designer treats it as a
    base to iterate on rather than synthesizing from scratch."""

    ee_result: Optional[dict]
    """Legacy entity-extraction/pretemplate artifact."""

    selected_components: list[str]
    """Ordered DesignLibrary component names chosen before schematic generation."""

    selected_pretemplate: Optional[dict]
    """Entity-extraction artifact after component names are mapped to PDK cells."""

    draft_dsl: Optional[dict]
    """Legacy draft DSL before schematic edges / placement are finalized."""

    schematic_dsl: Optional[dict]
    """Legacy schematic-generation output, close to GETTING_STARTED 4_SG."""

    legacy_debug: dict[str, str]
    """Optional raw DOT/preschematic artifacts from the original pipeline."""

    # ---- Designer / current artifact ---------------------------------
    circuit_dsl: Optional[dict]
    """DSL the Designer most recently produced (single-source-of-truth)."""

    gf_netlist: Optional[dict]
    """Cached gdsfactory netlist (derived from ``circuit_dsl``)."""

    gds_path: Optional[str]
    """Path to the most recently written GDS file."""

    gds_png_path: Optional[str]
    """Path to the most recently saved layout PNG (from ``gf.Component.plot``)."""

    sax_result: Optional[Any]
    """Reserved; **always** ``None`` in checkpointed state. The SAX curve is
    only exposed via ``sax_png_path`` (and the legacy ``build/plot_sax.png``) —
    persisting the sweep pytree broke LangGraph msgpack (``ArrayImpl``)."""

    sax_png_path: Optional[str]
    """Path to the saved S-parameter sweep PNG (from ``utils.plot_dict_arrays``)."""

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
    legacy_stage_model: str
    """Model for the original multi-stage path (entity → selection → preschematic, etc.)."""
    require_drc_pass: bool
    """If True (default), evaluator *pass* requires DRC zero violations. If
    False, *pass* when GDS and SAX are ok; DRC is still run and shown."""

    legacy_mode: str
    """``compat`` when the original multi-stage scaffold seeded the run."""

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
    legacy_stage_model: str = DEFAULT_LEGACY_STAGE_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    require_drc_pass: bool = True,
    thread_id: str = "",
) -> PhIDOState:
    """Return a fully-populated ``PhIDOState`` ready to feed into the graph."""
    return PhIDOState(
        user_prompt=user_prompt,
        seed_circuit_dsl=seed_circuit_dsl,
        ee_result=None,
        selected_components=[],
        selected_pretemplate=None,
        draft_dsl=None,
        schematic_dsl=None,
        legacy_debug={},
        circuit_dsl=None,
        gf_netlist=None,
        gds_path=None,
        gds_png_path=None,
        sax_result=None,
        sax_png_path=None,
        eda_report=None,
        reflections=[],
        retry_count=0,
        max_retries=max_retries,
        escalated=False,
        human_input=None,
        designer_model=designer_model,
        reflector_model=reflector_model,
        legacy_stage_model=legacy_stage_model,
        require_drc_pass=require_drc_pass,
        legacy_mode="",
        thread_id=thread_id,
        timings={},
        token_usage={},
    )
