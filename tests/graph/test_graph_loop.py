"""End-to-end loop test for the Reflexion subgraph.

We avoid hitting any real LLM provider by monkeypatching
``PhotonicsAI.graph.nodes.designer.invoke_text``. Two scenarios are
covered:

1. **DRC fail then pass**: the executor returns a violating EdaReport
   on round 1, then a clean one on round 2; we assert that the graph
   visits ``reflector`` exactly once and ends with a passing verdict.
2. **Cap reached**: the executor always returns failures; we assert
   that the graph routes to ``human_escalation`` at the cap.

The executor itself is also monkeypatched so tests do not need
gdsfactory / sax / klayout in the environment.
"""

from __future__ import annotations

from typing import Any

import pytest

from PhotonicsAI.graph.adapters.eda_report import (
    DRCExample,
    DRCReport,
    GdsReport,
    SaxReport,
    build_eda_report,
)
from PhotonicsAI.graph.state import initial_state


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM dispatcher with a controllable fake."""
    calls: list[dict[str, Any]] = []
    responses: list[str] = []

    def fake_invoke(prompt, system_prompt="", model="o1", **_):
        calls.append({"prompt": prompt, "model": model, "sys": system_prompt})
        if not responses:
            return "doc:\n  name: stub\nnodes: {}\nedges: {}\nproperties: {}\n"
        return responses.pop(0)

    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.designer.invoke_text", fake_invoke
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.reflector.invoke_text", fake_invoke
    )
    return calls, responses


@pytest.fixture
def fake_pdk_context(monkeypatch):
    """Avoid touching DesignLibrary on disk during graph tests."""
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes._pdk_context.list_available_cells",
        lambda: ["mzi_1x2_pindiode_cband", "mzi_2x2_heater_tin_cband"],
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes._pdk_context.cell_one_liners",
        lambda: {
            "mzi_1x2_pindiode_cband": "1x2 MZI with pin diode (C-band).",
            "mzi_2x2_heater_tin_cband": "2x2 MZI with TiN heater (C-band).",
        },
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes._pdk_context.cells_block",
        lambda max_cells=60: "- mzi_1x2_pindiode_cband: 1x2 MZI with pin diode.",
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes._pdk_context.cells_short",
        lambda max_cells=60: "mzi_1x2_pindiode_cband, mzi_2x2_heater_tin_cband",
    )


@pytest.fixture(autouse=True)
def fake_legacy_pipeline(monkeypatch):
    """Avoid real legacy LLM stages unless a test explicitly opts in."""
    schematic = {
        "doc": {"name": "legacy"},
        "nodes": {"N1": {"component": "stub"}},
        "edges": {},
        "properties": {},
    }
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.extract_entities",
        lambda *_args, **_kwargs: {"components_list": ["stub"]},
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.select_components",
        lambda *_args, **_kwargs: ["stub"],
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.build_schematic_from_pretemplate",
        lambda *_args, **_kwargs: (schematic, {"p200_preschematic": "", "p300_dot_string": ""}),
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.build_legacy_seed",
        lambda *_args, **_kwargs: {
            "ee_result": {"components_list": ["stub"]},
            "selected_components": ["stub"],
            "draft_dsl": {},
            "schematic_dsl": schematic,
        },
    )


def _violating_report() -> Any:
    return build_eda_report(
        GdsReport(ok=True, gds_path="/tmp/fake.gds"),
        SaxReport(ok=True, required_models=["mzi_1x2_pindiode_cband"]),
        DRCReport(
            ok=False,
            n_violations=42,
            by_rule={"Si_space": 42, "Si_width": 0},
            examples=[
                DRCExample(rule="Si_space", coord=[12.0, 56.0], min=0.13),
            ],
        ),
    )


def _clean_report() -> Any:
    return build_eda_report(
        GdsReport(ok=True, gds_path="/tmp/fake.gds"),
        SaxReport(ok=True),
        DRCReport(ok=True, n_violations=0),
    )


@pytest.fixture
def stubbed_executor(monkeypatch):
    """Replace the EDA executor with a queue-driven stub."""
    queue: list[Any] = []

    def fake_executor(state):
        if not queue:
            report = _clean_report()
        else:
            report = queue.pop(0)
        return {
            "eda_report": report,
            "gds_path": "/tmp/fake.gds",
            "sax_result": None,
            "timings": {},
        }

    # Patch BOTH the implementation site AND the import alias used by
    # ``graph.build_graph()`` (it does `from PhotonicsAI.graph.nodes import
    # eda_executor_node`, so we have to patch the re-export).
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.eda_executor.eda_executor_node", fake_executor
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.eda_executor_node", fake_executor, raising=False
    )
    import PhotonicsAI.graph.nodes as nodes_pkg

    monkeypatch.setattr(nodes_pkg, "eda_executor_node", fake_executor)
    import PhotonicsAI.graph.graph as graph_module

    monkeypatch.setattr(graph_module, "eda_executor_node", fake_executor)
    return queue


def _build_graph_no_checkpoint():
    from PhotonicsAI.graph.graph import build_graph

    return build_graph().compile()


def test_loop_fail_then_pass_visits_reflector_once(
    fake_llm, fake_pdk_context, stubbed_executor
):
    calls, responses = fake_llm
    # Designer round 1, Reflector, Designer round 2
    responses.extend(
        [
            "Lower N1.delta_length to 100 to clear Si_space.",
            "doc:\n  name: round2\nnodes: {}\nedges: {}\nproperties: {}\n",
        ]
    )
    stubbed_executor.append(_violating_report())
    stubbed_executor.append(_clean_report())

    graph = _build_graph_no_checkpoint()
    state = initial_state(
        "Design a 1x4 cWDM at 1550 nm.",
        max_retries=3,
    )
    visited: list[str] = []
    final_state: dict = {}
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in update.items():
            visited.append(node_name)
            final_state.update(node_state)
            if "reflections" in node_state:
                final_state.setdefault("reflections", []).extend(
                    node_state["reflections"]
                ) if not isinstance(
                    final_state.get("reflections"), list
                ) else None

    assert visited.count("reflector") == 1
    assert "human_escalation" not in visited
    assert visited[-1] == "evaluator"


def test_legacy_pipeline_seeds_prompt_without_designer_llm(
    monkeypatch, fake_llm, fake_pdk_context, stubbed_executor
):
    calls, responses = fake_llm
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.extract_entities",
        lambda *_args, **_kwargs: {"components_list": ["2x2 MZI"]},
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.select_components",
        lambda *_args, **_kwargs: ["mzi_2x2_pn_diode"],
    )
    schematic = {
        "doc": {"name": "legacy"},
        "nodes": {"N1": {"component": "mzi_2x2_pn_diode"}},
        "edges": {"E1": {"link": "N1,o3: N2,o2"}},
        "properties": {},
    }
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.build_schematic_from_pretemplate",
        lambda *_args, **_kwargs: (schematic, {"p200_preschematic": "", "p300_dot_string": ""}),
    )
    stubbed_executor.append(_clean_report())

    graph = _build_graph_no_checkpoint()
    state = initial_state("Two cascaded MZIs with 10 GHz bandwidth", max_retries=3)
    visited: list[str] = []
    final_state: dict = dict(state)
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in update.items():
            visited.append(node_name)
            final_state.update(node_state)

    assert visited[:4] == [
        "entity_extraction",
        "component_selection",
        "schematic_generation",
        "eda_executor",
    ]
    assert "designer" not in visited
    assert calls == []
    assert final_state["legacy_mode"] == "schematic_generated"
    assert final_state["circuit_dsl"]["edges"]["E1"]["link"] == "N1,o3: N2,o2"


def test_legacy_pipeline_failure_does_not_call_designer(
    monkeypatch, fake_llm, fake_pdk_context, stubbed_executor
):
    calls, responses = fake_llm
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.legacy_pipeline.extract_entities",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    graph = _build_graph_no_checkpoint()
    state = initial_state("unhandled design", max_retries=3)
    visited: list[str] = []
    final_state: dict = dict(state)
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in update.items():
            visited.append(node_name)
            final_state.update(node_state)

    assert "designer" not in visited
    assert "eda_executor" not in visited
    assert visited == ["entity_extraction", "human_escalation"]
    assert calls == []
    assert final_state["legacy_mode"] == "failed"


def test_drc_ignored_ends_without_reflector(
    fake_llm, fake_pdk_context, stubbed_executor
):
    """With require_drc_pass=False, a DRC-violating report still routes pass -> END."""
    calls, responses = fake_llm
    stubbed_executor.append(_violating_report())

    graph = _build_graph_no_checkpoint()
    state = initial_state("x", require_drc_pass=False, max_retries=3)
    visited: list[str] = []
    for update in graph.stream(state, stream_mode="updates"):
        visited.extend(update.keys())
    assert "reflector" not in visited
    assert "evaluator" in visited
    assert visited[-1] == "evaluator"


def test_loop_caps_at_max_retries_and_escalates(
    fake_llm, fake_pdk_context, stubbed_executor
):
    calls, responses = fake_llm
    responses.extend(
        [
            "first reflection",
            "doc:\n  name: r2\nnodes: {}\nedges: {}\nproperties: {}\n",
        ]
    )
    stubbed_executor.extend([_violating_report(), _violating_report(), _violating_report()])

    graph = _build_graph_no_checkpoint()
    state = initial_state(
        "Design a doomed circuit",
        max_retries=1,  # one retry then escalate
    )
    visited: list[str] = []
    for update in graph.stream(state, stream_mode="updates"):
        for node_name in update.keys():
            visited.append(node_name)

    assert "human_escalation" in visited
    assert visited.count("reflector") == 1


def test_loop_aborts_when_designer_emits_invalid_yaml(
    fake_llm, fake_pdk_context, stubbed_executor
):
    """Designer producing junk should still loop (reflection appended) rather than crash."""
    calls, responses = fake_llm
    responses.extend(
        [
            "this is not yaml, escape: : :",
            "Please return valid YAML.",
            "doc:\n  name: ok\nnodes: {}\nedges: {}\nproperties: {}\n",
        ]
    )
    stubbed_executor.extend([_violating_report(), _clean_report()])

    graph = _build_graph_no_checkpoint()
    state = initial_state("design X", max_retries=3)

    visited: list[str] = []
    for update in graph.stream(state, stream_mode="updates"):
        visited.extend(update.keys())

    assert "designer" in visited
    assert "evaluator" in visited


def test_designer_preserves_topology_for_routing_only_reflection(
    monkeypatch, fake_pdk_context
):
    from PhotonicsAI.graph.nodes.designer import designer_node

    previous = {
        "doc": {"title": "1x8 splitter"},
        "nodes": {
            "N1": {"component": "mzi", "placement": {"x": 1, "y": 1}},
            "N2": {"component": "mzi", "placement": {"x": 2, "y": 2}},
            "N3": {"component": "mzi", "placement": {"x": 2, "y": 0}},
            "N4": {"component": "mzi", "placement": {"x": 3, "y": 3}},
        },
        "edges": {
            "E1": {"link": "N1,o2: N2,o1"},
            "E2": {"link": "N1,o3: N3,o1"},
            "E3": {"link": "N2,o2: N4,o1"},
        },
        "ports": {"o1": "N1,o1", "o2": "N3,o2", "o3": "N4,o2"},
        "properties": {},
    }
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.designer.invoke_text",
        lambda *_args, **_kwargs: (
            "doc:\n"
            "  title: 1x8 splitter\n"
            "nodes:\n"
            "  N1:\n"
            "    component: mzi\n"
            "    placement: {x: 1, y: 1}\n"
            "  N2:\n"
            "    component: mzi\n"
            "    placement: {x: 20, y: 20}\n"
            "  N3:\n"
            "    component: mzi\n"
            "    placement: {x: 2, y: 0}\n"
            "  N4:\n"
            "    component: mzi\n"
            "    placement: {x: 3, y: 3}\n"
            "edges:\n"
            "  E1: {link: 'N1,o2: N2,o1'}\n"
            "  E2: {link: 'N1,o3: N3,o1'}\n"
            "  E3: {link: 'N2,o2: N4,o1'}\n"
            "  E4: {link: 'N1,o1: N4,o2'}\n"
            "ports:\n"
            "  o1: N3,o2\n"
            "  o2: N4,o3\n"
            "properties: {}\n"
        ),
    )

    out = designer_node(
        {
            "user_prompt": "1x8 splitter",
            "circuit_dsl": previous,
            "reflections": [
                "To resolve the routing failure, scale placements by 1.5. DECISION: RETRY"
            ],
        }
    )

    assert out["circuit_dsl"]["edges"] == previous["edges"]
    assert out["circuit_dsl"]["ports"] == previous["ports"]
    assert out["circuit_dsl"]["nodes"]["N2"]["placement"]["x"] == 20
    assert "Preserved previous edges/ports" in out["reflections"][0]
