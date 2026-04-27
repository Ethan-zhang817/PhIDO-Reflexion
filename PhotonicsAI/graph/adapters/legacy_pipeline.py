"""Adapter for the original multi-stage PhIDO schematic pipeline.

This is **not** a tutorial replay layer. It wraps the legacy stages the Streamlit
app used before layout: entity extraction, component search, draft DSL creation,
ports/params enrichment, settings absorption, DOT edge generation, Graphviz
placement, and top-level port derivation.
"""

from __future__ import annotations

import re
from typing import Any

from PhotonicsAI.graph.state import DEFAULT_LEGACY_STAGE_MODEL


class LegacyPipelineError(RuntimeError):
    """Raised when the legacy schematic pipeline cannot produce a DSL."""


class AttrDict(dict):
    """Dict with attribute access, matching Streamlit ``session_state`` usage."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def map_pretemplate_to_draft(pretemplate: dict[str, Any]) -> dict[str, Any]:
    """Pure equivalent of ``webapp.map_pretemplate_to_template``."""
    return {
        "doc": {
            "title": pretemplate.get("title", ""),
            "description": pretemplate.get("brief_summary", ""),
            "reference": "(link)",
            "labels": [""],
        },
        "nodes": {
            f"N{i}": {"component": component}
            for i, component in enumerate(pretemplate.get("components_list", []), start=1)
        },
        "edges": pretemplate.get("circuit_instructions", ""),
        "properties": {},
    }


def extract_entities(user_prompt: str, *, model: str = DEFAULT_LEGACY_STAGE_MODEL) -> dict[str, Any]:
    """Run the original pydantic entity-extraction step (graph uses configurable model)."""
    from PhotonicsAI.graph.legacy_openai_stages import entity_extraction_with_model

    result = entity_extraction_with_model(user_prompt, model=model)
    if not isinstance(result, dict):
        raise LegacyPipelineError("entity_extraction did not return a mapping")
    if not result.get("components_list"):
        raise LegacyPipelineError("entity_extraction returned no components")
    return result


def select_components(
    pretemplate: dict[str, Any], *, model: str = DEFAULT_LEGACY_STAGE_MODEL
) -> list[str]:
    """Run the original component-search step for each extracted component."""
    from PhotonicsAI.Photon import DemoPDK
    from PhotonicsAI.graph.legacy_openai_stages import llm_search_with_model

    selected: list[str] = []
    docs = list(DemoPDK.list_of_docs)
    names = list(DemoPDK.list_of_cnames)

    for component_text in pretemplate.get("components_list", []):
        search = llm_search_with_model(str(component_text), docs, model=model)
        matches = getattr(search, "match_list", []) or []
        if not matches:
            raise LegacyPipelineError(f"no component match for {component_text!r}")
        idx = int(matches[0])
        if idx < 0 or idx >= len(names):
            raise LegacyPipelineError(f"component match index out of range: {idx}")
        selected.append(names[idx])
    return selected


def apply_component_selection(
    pretemplate: dict[str, Any],
    selected_components: list[str],
) -> dict[str, Any]:
    """Return a pretemplate whose components are DesignLibrary cell names."""
    if len(selected_components) != len(pretemplate.get("components_list", [])):
        raise LegacyPipelineError("selected component count does not match pretemplate")
    out = dict(pretemplate)
    out["components_list"] = list(selected_components)
    return out


def _strip_dot_fences(dot: str) -> str:
    text = (dot or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.replace("```dot", "").replace("```", "").strip()


_DOT_EDGE_RE = re.compile(r"\b([A-Za-z_]\w*)(?::\w+)?\s*--\s*([A-Za-z_]\w*)(?::\w+)?")
_DOT_NODE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s+\[", re.MULTILINE)


def _dot_edges(dot: str) -> list[tuple[str, str]]:
    """Extract undirected DOT edges, ignoring optional port suffixes."""
    return [(a, b) for a, b in _DOT_EDGE_RE.findall(dot or "")]


def _preschematic_node_order(dot: str) -> list[str]:
    """Return node order from declarations plus first-seen edge endpoints."""
    ordered: list[str] = []
    seen: set[str] = set()
    for node in _DOT_NODE_RE.findall(dot or ""):
        if node not in seen and node not in {"graph", "node", "edge"}:
            ordered.append(node)
            seen.add(node)
    for a, b in _dot_edges(dot):
        for node in (a, b):
            if node not in seen:
                ordered.append(node)
                seen.add(node)
    return ordered


def _all_nodes_are_one_by_two(circuit_dsl: dict[str, Any]) -> bool:
    for node in (circuit_dsl.get("nodes") or {}).values():
        props = (node or {}).get("properties") or {}
        if str(props.get("ports", "")).lower() != "1x2":
            return False
    return bool(circuit_dsl.get("nodes"))


def _edge_set(edges: list[tuple[str, str]]) -> set[frozenset[str]]:
    return {frozenset((a, b)) for a, b in edges}


def _deterministic_one_by_two_tree_dot(session: AttrDict) -> str | None:
    """Build a 1x2 tree DOT from the preschematic instead of trusting LLM edits.

    The legacy LLM ``dot_add_edges`` can occasionally add extra lateral links
    while trying to avoid crossings. For a pure 1x2 splitter tree the physical
    port assignment is deterministic: each parent uses o2/o3, each child uses o1.
    """
    circuit = session.p300_circuit_dsl
    if not _all_nodes_are_one_by_two(circuit):
        return None

    draft_nodes = list((circuit.get("nodes") or {}).keys())
    pre_nodes = _preschematic_node_order(session.p200_preschematic)
    if len(pre_nodes) != len(draft_nodes):
        return None
    name_map = dict(zip(pre_nodes, draft_nodes))
    preschematic_edges = _dot_edges(session.p200_preschematic)
    if not preschematic_edges:
        return None

    out_port_index = {node: 2 for node in draft_nodes}
    dot_lines = session.p300_dot_string_draft.strip().splitlines()
    try:
        closing = len(dot_lines) - 1 - dot_lines[::-1].index("}")
    except ValueError:
        return None

    edge_lines: list[str] = []
    for left, right in preschematic_edges:
        src = name_map.get(left)
        dst = name_map.get(right)
        if src is None or dst is None:
            return None
        out_port = out_port_index[src]
        if out_port > 3:
            return None
        out_port_index[src] += 1
        edge_lines.append(f"  {src}:o{out_port} -- {dst}:o1; ")

    return "\n".join(dot_lines[:closing] + edge_lines + dot_lines[closing:])


def _repair_dot_if_preschematic_mismatch(session: AttrDict) -> None:
    """Repair DOT when LLM changed the preschematic topology."""
    preschematic_edges = _dot_edges(session.p200_preschematic)
    candidate_edges = _dot_edges(session.p300_dot_string)
    pre_nodes = _preschematic_node_order(session.p200_preschematic)
    draft_nodes = list((session.p300_circuit_dsl.get("nodes") or {}).keys())
    name_map = dict(zip(pre_nodes, draft_nodes))
    mapped_edges = [
        (name_map.get(a, ""), name_map.get(b, ""))
        for a, b in preschematic_edges
        if name_map.get(a) and name_map.get(b)
    ]
    if len(candidate_edges) == len(mapped_edges) and _edge_set(candidate_edges) == _edge_set(mapped_edges):
        return

    repaired = _deterministic_one_by_two_tree_dot(session)
    if repaired is not None:
        session.p300_dot_string = repaired
        session.legacy_dot_repair = (
            "Rebuilt 1x2 tree DOT from preschematic because LLM DOT "
            f"topology differed: expected {len(mapped_edges)} edges, "
            f"got {len(candidate_edges)}."
        )


def build_schematic_from_pretemplate(
    pretemplate: dict[str, Any],
    *,
    model: str = DEFAULT_LEGACY_STAGE_MODEL,
    max_planarity_attempts: int = 4,
) -> tuple[dict[str, Any], AttrDict]:
    """Run the original schematic-generation path using a session-like object."""
    from PhotonicsAI.Photon import DemoPDK, llm_api, utils

    session = AttrDict()
    session.p100_llm_api_selection = model
    session.p200_pretemplate = pretemplate
    session.p200_pretemplate_copy = {"components_list": list(pretemplate["components_list"])}
    session.p200_preschematic = llm_api.preschematic(pretemplate, model)
    session.p300_circuit_dsl = map_pretemplate_to_draft(pretemplate)
    session.template_selected = False

    session.p300_circuit_dsl = DemoPDK.get_ports_info(session.p300_circuit_dsl)
    session.p300_circuit_dsl = DemoPDK.get_params(session.p300_circuit_dsl)
    session.p300_circuit_dsl = llm_api.apply_settings(session, model)

    session.p300_dot_string_draft = utils.circuit_to_dot(session.p300_circuit_dsl)
    if len(session.p300_circuit_dsl.get("nodes", {})) > 0:
        session.p300_dot_string = _strip_dot_fences(llm_api.dot_add_edges(session))
        session.p300_dot_string = _strip_dot_fences(llm_api.dot_verify(session))
        for _ in range(max_planarity_attempts):
            if utils.dot_planarity(session.p300_dot_string):
                break
            session.p300_dot_string = _strip_dot_fences(
                llm_api.dot_add_edges_errorfunc(session)
            )
            session.p300_dot_string = _strip_dot_fences(llm_api.dot_verify(session))
    else:
        session.p300_dot_string = utils.circuit_to_dot(session.p300_circuit_dsl)
        session.p300_dot_string = _strip_dot_fences(llm_api.dot_add_edges_templates(session))
    session.p300_dot_string = _strip_dot_fences(llm_api.dot_verify(session))
    _repair_dot_if_preschematic_mismatch(session)

    session.p300_circuit_dsl = utils.edges_dot_to_yaml(session)
    session.p300_footprints_dict, session.p300_circuit_dsl = DemoPDK.footprint_netlist(
        session.p300_circuit_dsl
    )
    session.p300_dot_string_scaled = utils.dot_add_node_sizes(
        session.p300_dot_string,
        utils.multiply_node_dimensions(session.p300_footprints_dict, 0.01),
    )
    session.p300_graphviz_node_coordinates = utils.get_graphviz_placements(
        session.p300_dot_string_scaled
    )
    session.p300_graphviz_node_coordinates = utils.multiply_node_dimensions(
        session.p300_graphviz_node_coordinates, 100 / 72
    )
    session.p300_circuit_dsl = utils.add_placements_to_dsl(session)
    session.p300_circuit_dsl = utils.add_final_ports(session)
    return session.p300_circuit_dsl, session


def build_legacy_seed(
    user_prompt: str, *, model: str = DEFAULT_LEGACY_STAGE_MODEL
) -> dict[str, Any] | None:
    """Run the original-style PhIDO pipeline and return graph seed artifacts.

    Returns ``None`` when the old pipeline fails so the graph can transparently
    fall back to the normal Designer node.
    """
    try:
        extracted = extract_entities(user_prompt, model=model)
        selected_components = select_components(extracted, model=model)
        selected_pretemplate = apply_component_selection(extracted, selected_components)
        draft = map_pretemplate_to_draft(selected_pretemplate)
        schematic, session = build_schematic_from_pretemplate(
            selected_pretemplate,
            model=model,
        )
    except Exception:  # noqa: BLE001
        return None
    return {
        "ee_result": extracted,
        "selected_components": selected_components,
        "draft_dsl": draft,
        "schematic_dsl": schematic,
        "legacy_debug": {
            "preschematic": session.get("p200_preschematic", ""),
            "dot_string": session.get("p300_dot_string", ""),
            "dot_repair": session.get("legacy_dot_repair", ""),
        },
    }
