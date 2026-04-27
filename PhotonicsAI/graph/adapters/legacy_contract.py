"""Contracts for the legacy PhIDO multi-stage workflow.

The LangGraph implementation should not treat the old Streamlit app as a
single black-box LLM call.  These contracts document the intermediate artifacts
that made the original flow stable enough for the getting-started examples.
They are intentionally small and import-light so tests can assert the migration
surface without loading Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyStageContract:
    """One observable stage in the original PhIDO workflow."""

    name: str
    source: str
    graph_state_field: str
    output_shape: str
    required_for: str


LEGACY_STAGE_CONTRACTS: tuple[LegacyStageContract, ...] = (
    LegacyStageContract(
        name="entity_extraction",
        source="PhotonicsAI.Photon.llm_api.entity_extraction",
        graph_state_field="ee_result",
        output_shape="pretemplate: title, brief_summary, circuit_instructions, components_list",
        required_for="component selection and template/draft DSL creation",
    ),
    LegacyStageContract(
        name="component_selection",
        source="PhotonicsAI.Photon.llm_api.llm_search / llm_retrieve",
        graph_state_field="selected_components",
        output_shape="ordered DesignLibrary component names",
        required_for="using model-backed PDK cells instead of ad-hoc component names",
    ),
    LegacyStageContract(
        name="draft_dsl",
        source="PhotonicsAI.Photon.webapp.map_pretemplate_to_template",
        graph_state_field="draft_dsl",
        output_shape="doc/nodes/edges/properties with nodes N1..Nn",
        required_for="schematic generation and settings absorption",
    ),
    LegacyStageContract(
        name="ports_params_settings",
        source="DemoPDK.get_ports_info / get_params + llm_api.apply_settings",
        graph_state_field="schematic_dsl",
        output_shape="nodes enriched with properties.ports, params, placements, ports",
        required_for="gdsfactory netlist generation and SAX model resolution",
    ),
    LegacyStageContract(
        name="schematic_placement",
        source="utils.circuit_to_dot / dot_add_edges / Graphviz placement",
        graph_state_field="schematic_dsl",
        output_shape="structured edges plus placement and top-level ports",
        required_for="routing-stable GDS generation",
    ),
)


def legacy_contract_summary() -> list[dict[str, str]]:
    """Return a serializable summary for UI/debug panels and tests."""
    return [
        {
            "name": c.name,
            "source": c.source,
            "graph_state_field": c.graph_state_field,
            "output_shape": c.output_shape,
            "required_for": c.required_for,
        }
        for c in LEGACY_STAGE_CONTRACTS
    ]
