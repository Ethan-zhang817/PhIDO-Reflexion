"""Legacy PhIDO pipeline node.

Runs the original multi-stage PhIDO schematic pipeline before layout. If that
pipeline fails, the graph must stop/escalate instead of falling back to direct
prompt-to-DSL synthesis.
"""

from __future__ import annotations

import copy

from PhotonicsAI.graph.adapters.legacy_pipeline import (
    apply_component_selection,
    build_legacy_seed,
    build_schematic_from_pretemplate,
    extract_entities,
    map_pretemplate_to_draft,
    select_components,
)
from PhotonicsAI.graph.state import DEFAULT_LEGACY_STAGE_MODEL, PhIDOState


def entity_extraction_node(state: PhIDOState) -> dict:
    """Run the original PhIDO entity-extraction stage."""
    if state.get("ee_result"):
        return {}
    try:
        model = state.get("legacy_stage_model") or DEFAULT_LEGACY_STAGE_MODEL
        extracted = extract_entities(state.get("user_prompt", ""), model=model)
    except Exception as exc:  # noqa: BLE001
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": [
                f"[entity_extraction] Original PhIDO entity extraction failed: {exc}"
            ],
        }
    return {"ee_result": extracted, "legacy_mode": "entity_extracted"}


def route_after_entity_extraction(state: PhIDOState) -> str:
    """Continue only when entity extraction produced a usable artifact."""
    return "ready" if state.get("ee_result") else "fail"


def component_selection_node(state: PhIDOState) -> dict:
    """Run the original PhIDO component-search / selection stage."""
    pretemplate = state.get("ee_result")
    if not pretemplate:
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": ["[component_selection] Missing entity-extraction result."],
        }
    try:
        model = state.get("legacy_stage_model") or DEFAULT_LEGACY_STAGE_MODEL
        selected_components = select_components(pretemplate, model=model)
        selected_pretemplate = apply_component_selection(
            pretemplate,
            selected_components,
        )
        draft = map_pretemplate_to_draft(selected_pretemplate)
    except Exception as exc:  # noqa: BLE001
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": [
                f"[component_selection] Original PhIDO component selection failed: {exc}"
            ],
        }
    return {
        "selected_components": selected_components,
        "selected_pretemplate": selected_pretemplate,
        "draft_dsl": draft,
        "legacy_mode": "components_selected",
    }


def route_after_component_selection(state: PhIDOState) -> str:
    """Continue only when components are mapped to PDK cells."""
    return "ready" if state.get("selected_pretemplate") else "fail"


def schematic_generation_node(state: PhIDOState) -> dict:
    """Run the original PhIDO schematic-generation / placement stage."""
    if state.get("circuit_dsl"):
        return {}
    seed = state.get("seed_circuit_dsl")
    if seed:
        return {
            "draft_dsl": copy.deepcopy(seed),
            "schematic_dsl": copy.deepcopy(seed),
            "circuit_dsl": copy.deepcopy(seed),
            "legacy_mode": "seed",
        }
    selected_pretemplate = state.get("selected_pretemplate")
    if not selected_pretemplate:
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": ["[schematic_generation] Missing selected pretemplate."],
        }
    try:
        schematic, session = build_schematic_from_pretemplate(
            selected_pretemplate,
            model=state.get("legacy_stage_model") or DEFAULT_LEGACY_STAGE_MODEL,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": [
                f"[schematic_generation] Original PhIDO schematic generation failed: {exc}"
            ],
        }
    return {
        "schematic_dsl": schematic,
        "circuit_dsl": copy.deepcopy(schematic),
        "legacy_debug": {
            "preschematic": session.get("p200_preschematic", ""),
            "dot_string": session.get("p300_dot_string", ""),
        },
        "legacy_mode": "schematic_generated",
    }


def route_after_schematic_generation(state: PhIDOState) -> str:
    """Continue to layout only when schematic generation produced a DSL."""
    return "ready" if state.get("circuit_dsl") else "fail"


def legacy_pipeline_node(state: PhIDOState) -> dict:
    """Populate legacy intermediate artifacts and a schematic DSL when known."""
    if state.get("circuit_dsl"):
        return {}
    seed = state.get("seed_circuit_dsl")
    if seed:
        return {
            "draft_dsl": copy.deepcopy(seed),
            "schematic_dsl": copy.deepcopy(seed),
            "circuit_dsl": copy.deepcopy(seed),
            "legacy_mode": "seed",
        }
    built = build_legacy_seed(
        state.get("user_prompt", ""),
        model=state.get("legacy_stage_model") or DEFAULT_LEGACY_STAGE_MODEL,
    )
    if built is None:
        return {
            "legacy_mode": "failed",
            "escalated": True,
            "reflections": [
                "[legacy_pipeline] Original multi-stage PhIDO flow failed; "
                "direct prompt-to-DSL generation is disabled."
            ],
        }
    return {
        **built,
        "circuit_dsl": copy.deepcopy(built["schematic_dsl"]),
        "legacy_mode": "compat",
    }


def route_after_legacy_pipeline(state: PhIDOState) -> str:
    """Only continue when legacy flow produced a concrete DSL."""
    return "ready" if state.get("circuit_dsl") else "fail"
