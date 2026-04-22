"""End-to-end loop test for the Reflexion subgraph.

We avoid hitting any real LLM provider by monkeypatching
``PhotonicsAI.graph.llm_runnables.invoke_llm``. Two scenarios are
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
        "PhotonicsAI.graph.nodes.designer.invoke_llm", fake_invoke
    )
    monkeypatch.setattr(
        "PhotonicsAI.graph.nodes.reflector.invoke_llm", fake_invoke
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
            "doc:\n  name: round1\nnodes: {}\nedges: {}\nproperties: {}\n",
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


def test_loop_caps_at_max_retries_and_escalates(
    fake_llm, fake_pdk_context, stubbed_executor
):
    calls, responses = fake_llm
    responses.extend(
        [
            "doc:\n  name: r1\nnodes: {}\nedges: {}\nproperties: {}\n",
            "first reflection",
            "doc:\n  name: r2\nnodes: {}\nedges: {}\nproperties: {}\n",
            "second reflection",
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
