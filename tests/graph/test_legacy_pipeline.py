"""Legacy PhIDO compatibility scaffolding tests."""

from __future__ import annotations

import yaml

from PhotonicsAI.graph.adapters.legacy_contract import legacy_contract_summary
from PhotonicsAI.graph.adapters.legacy_pipeline import (
    AttrDict,
    build_schematic_from_pretemplate,
    build_legacy_seed,
    map_pretemplate_to_draft,
)
from PhotonicsAI.graph.nodes.legacy_pipeline import legacy_pipeline_node


def test_legacy_contract_maps_original_stages_to_graph_state():
    summary = legacy_contract_summary()
    fields = {item["graph_state_field"] for item in summary}
    assert {
        "ee_result",
        "selected_components",
        "draft_dsl",
        "schematic_dsl",
    }.issubset(fields)
    assert any("apply_settings" in item["source"] for item in summary)


def test_map_pretemplate_to_draft_matches_original_shape():
    pretemplate = {
        "title": "Example",
        "brief_summary": "summary",
        "circuit_instructions": "connect two blocks",
        "components_list": ["a", "b"],
    }
    draft = map_pretemplate_to_draft(pretemplate)
    assert list(draft) == ["doc", "nodes", "edges", "properties"]
    assert draft["nodes"]["N1"]["component"] == "a"
    assert draft["nodes"]["N2"]["component"] == "b"
    assert draft["edges"] == "connect two blocks"


def _patch_real_legacy_steps(monkeypatch):
    from PhotonicsAI.Photon import DemoPDK, llm_api, utils

    monkeypatch.setattr(DemoPDK, "list_of_docs", ["mzi doc"], raising=False)
    monkeypatch.setattr(DemoPDK, "list_of_cnames", ["mzi_2x2_pn_diode"], raising=False)
    monkeypatch.setattr(
        llm_api,
        "entity_extraction",
        lambda _prompt: {
            "title": "x",
            "brief_summary": "summary",
            "circuit_instructions": "cascade two blocks",
            "components_list": ["2x2 MZI", "2x2 MZI"],
        },
    )

    class Match:
        match_list = [0]

    monkeypatch.setattr(llm_api, "llm_search", lambda _query, _contexts: Match())
    monkeypatch.setattr(llm_api, "preschematic", lambda _pretemplate, _model: "graph g { N1 -- N2; }")
    monkeypatch.setattr(
        DemoPDK,
        "get_ports_info",
        lambda dsl: {
            **dsl,
            "nodes": {
                k: {**v, "properties": {"ports": "2x2"}}
                for k, v in dsl["nodes"].items()
            },
        },
    )
    monkeypatch.setattr(
        DemoPDK,
        "get_params",
        lambda dsl: {
            **dsl,
            "nodes": {k: {**v, "params": {"length": 10}} for k, v in dsl["nodes"].items()},
        },
    )
    monkeypatch.setattr(llm_api, "apply_settings", lambda session, _model: session.p300_circuit_dsl)
    monkeypatch.setattr(llm_api, "dot_add_edges", lambda _session: 'graph g {\n  N1 [label="{{<o2> o2|<o1> o1} | N1: mzi | {<o3> o3|<o4> o4}}"];\n  N2 [label="{{<o2> o2|<o1> o1} | N2: mzi | {<o3> o3|<o4> o4}}"];\n  N1:o3 -- N2:o2;\n}')
    monkeypatch.setattr(llm_api, "dot_verify", lambda session: session.p300_dot_string)
    monkeypatch.setattr(llm_api, "dot_add_edges_errorfunc", lambda session: session.p300_dot_string)
    monkeypatch.setattr(utils, "dot_planarity", lambda _dot: True)
    monkeypatch.setattr(
        DemoPDK,
        "footprint_netlist",
        lambda dsl: (
            {"N1": (10.0, 4.0), "N2": (10.0, 4.0)},
            {
                **dsl,
                "nodes": {
                    k: {**v, "properties": {**v["properties"], "dx": 10.0, "dy": 4.0}}
                    for k, v in dsl["nodes"].items()
                },
            },
        ),
    )
    monkeypatch.setattr(utils, "get_graphviz_placements", lambda _dot: {"N1": (10.0, 20.0), "N2": (30.0, 20.0)})


def test_build_legacy_seed_runs_real_stage_sequence(monkeypatch):
    _patch_real_legacy_steps(monkeypatch)
    seed = build_legacy_seed("any new design")
    assert seed is not None
    assert seed["selected_components"] == ["mzi_2x2_pn_diode", "mzi_2x2_pn_diode"]
    schematic = seed["schematic_dsl"]
    assert schematic["edges"]["E1"]["link"] == "N1,o3: N2,o2"
    assert schematic["nodes"]["N1"]["placement"]["rotation"] == 0
    assert schematic["ports"]["o1"] == "N1,o2"
    assert "dot_string" in seed["legacy_debug"]


def test_schematic_generation_repairs_bad_llm_tree_edges(monkeypatch):
    from PhotonicsAI.Photon import DemoPDK, llm_api, utils

    pretemplate = {
        "title": "1x8 splitter",
        "brief_summary": "tree",
        "circuit_instructions": "three-stage 1x8 tree",
        "components_list": ["mzi_1x2_pindiode_cband"] * 7,
    }
    preschematic = """graph G {
    rankdir=LR;
    splitter [label="1x2 MZI 1"];
    mzi1 [label="MZI 1"];
    mzi2 [label="MZI 2"];
    mzi3 [label="MZI 3"];
    mzi4 [label="MZI 4"];
    mzi5 [label="MZI 5"];
    mzi6 [label="MZI 6"];
    splitter -- mzi1;
    splitter -- mzi2;
    mzi1 -- mzi3;
    mzi1 -- mzi4;
    mzi2 -- mzi5;
    mzi2 -- mzi6;
}"""
    bad_dot = """graph graph_name_placeholder {
  rankdir=LR;
  node [shape=record];
  N1 [label="{{<o1> o1} | N1: mzi | {<o2> o2|<o3> o3}}"];
  N2 [label="{{<o1> o1} | N2: mzi | {<o2> o2|<o3> o3}}"];
  N3 [label="{{<o1> o1} | N3: mzi | {<o2> o2|<o3> o3}}"];
  N4 [label="{{<o1> o1} | N4: mzi | {<o2> o2|<o3> o3}}"];
  N5 [label="{{<o1> o1} | N5: mzi | {<o2> o2|<o3> o3}}"];
  N6 [label="{{<o1> o1} | N6: mzi | {<o2> o2|<o3> o3}}"];
  N7 [label="{{<o1> o1} | N7: mzi | {<o2> o2|<o3> o3}}"];
  N1:o2 -- N2:o1;
  N2:o2 -- N3:o1;
  N2:o3 -- N4:o1;
  N3:o2 -- N5:o1;
  N3:o3 -- N6:o1;
  N4:o3 -- N7:o1;
  N5:o2 -- N6:o3;
  N6:o2 -- N7:o3;
}"""

    monkeypatch.setattr(llm_api, "preschematic", lambda _pretemplate, _model: preschematic)
    monkeypatch.setattr(
        DemoPDK,
        "get_ports_info",
        lambda dsl: {
            **dsl,
            "nodes": {
                k: {**v, "properties": {"ports": "1x2"}}
                for k, v in dsl["nodes"].items()
            },
        },
    )
    monkeypatch.setattr(
        DemoPDK,
        "get_params",
        lambda dsl: {
            **dsl,
            "nodes": {
                k: {**v, "params": {"length": 320, "delta_length": 100}}
                for k, v in dsl["nodes"].items()
            },
        },
    )
    monkeypatch.setattr(llm_api, "apply_settings", lambda session, _model: session.p300_circuit_dsl)
    monkeypatch.setattr(llm_api, "dot_add_edges", lambda _session: bad_dot)
    monkeypatch.setattr(llm_api, "dot_verify", lambda session: session.p300_dot_string)
    monkeypatch.setattr(llm_api, "dot_add_edges_errorfunc", lambda session: session.p300_dot_string)
    monkeypatch.setattr(utils, "dot_planarity", lambda _dot: True)
    monkeypatch.setattr(
        DemoPDK,
        "footprint_netlist",
        lambda dsl: (
            {f"N{i}": (428.3, 512.2) for i in range(1, 8)},
            {
                **dsl,
                "nodes": {
                    k: {**v, "properties": {**v["properties"], "dx": 428.3, "dy": 512.2}}
                    for k, v in dsl["nodes"].items()
                },
            },
        ),
    )
    monkeypatch.setattr(
        utils,
        "get_graphviz_placements",
        lambda _dot: {f"N{i}": (float(i * 100), float(i * 50)) for i in range(1, 8)},
    )

    schematic, session = build_schematic_from_pretemplate(pretemplate)

    links = [edge["link"] for edge in schematic["edges"].values()]
    assert links == [
        "N1,o2: N2,o1",
        "N1,o3: N3,o1",
        "N2,o2: N4,o1",
        "N2,o3: N5,o1",
        "N3,o2: N6,o1",
        "N3,o3: N7,o1",
    ]
    assert len(schematic["ports"]) == 9
    assert "Rebuilt 1x2 tree DOT" in session.legacy_dot_repair


def test_legacy_pipeline_node_uses_legacy_flow(monkeypatch):
    _patch_real_legacy_steps(monkeypatch)
    state = {"user_prompt": "Two cascaded MZIs", "reflections": []}
    out = legacy_pipeline_node(state)
    assert out["legacy_mode"] == "compat"
    assert out["circuit_dsl"]["nodes"]["N1"]["component"] == "mzi_2x2_pn_diode"
    yaml.safe_dump(out["circuit_dsl"])


def test_attrdict_matches_streamlit_session_access():
    session = AttrDict()
    session.foo = "bar"
    assert session["foo"] == "bar"
    session["baz"] = 1
    assert session.baz == 1
