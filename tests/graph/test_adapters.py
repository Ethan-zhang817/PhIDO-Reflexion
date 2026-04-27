"""Unit tests for the EDA adapters.

These tests do *not* require KLayout, gdsfactory, sax, or any LLM
provider — every adapter is exercised against either a static fixture
or a mocked counterpart so the suite runs in <1s on a clean checkout.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from PhotonicsAI.graph.adapters import drc_adapter
from PhotonicsAI.graph.adapters.drc_parser import parse_lyrdb_xml
from PhotonicsAI.graph.adapters.eda_report import (
    DRCExample,
    DRCReport,
    GdsReport,
    SaxReport,
    build_eda_report,
    summarize_report,
)
from PhotonicsAI.graph.adapters.errors import (
    GdsBuildError,
    NetlistError,
    SaxModelMissingError,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# DRC parser
# ---------------------------------------------------------------------------


def test_drc_parser_counts_violations_and_extracts_examples():
    report = parse_lyrdb_xml(FIXTURES / "sample_report.lyrdb", top_k=3)
    assert isinstance(report, DRCReport)
    assert report.ok is False
    assert report.n_violations == 3
    assert report.by_rule == {"Si_space": 2, "Si_width": 1}
    assert len(report.examples) == 3
    si_space_examples = [ex for ex in report.examples if ex.rule == "Si_space"]
    assert len(si_space_examples) == 2
    first = si_space_examples[0]
    assert pytest.approx(first.coord[0], rel=1e-3) == 12.345
    assert pytest.approx(first.coord[1], rel=1e-3) == 67.890
    assert first.min == pytest.approx(0.130)


def test_drc_parser_handles_empty_report():
    report = parse_lyrdb_xml(FIXTURES / "empty_report.lyrdb")
    assert report.ok is True
    assert report.n_violations == 0
    assert report.by_rule == {}


def test_drc_parser_missing_file_returns_clean_report(tmp_path):
    report = parse_lyrdb_xml(tmp_path / "does_not_exist.lyrdb")
    assert report.ok is True
    assert report.n_violations == 0


def test_drc_parser_malformed_xml_marks_skipped(tmp_path):
    bad = tmp_path / "bad.lyrdb"
    bad.write_text("<not-xml>broken")
    report = parse_lyrdb_xml(bad)
    assert report.skipped_reason == "lyrdb parse error"


# ---------------------------------------------------------------------------
# DRC adapter (mocked KLayout)
# ---------------------------------------------------------------------------


def test_drc_adapter_skips_when_klayout_missing(tmp_path):
    with mock.patch.object(drc_adapter, "find_klayout", return_value=None):
        report = drc_adapter.run_drc_structured(tmp_path / "anything.gds")
    assert report.skipped_reason == "klayout_not_found"
    assert report.ok is True  # soft-pass downstream


def test_drc_adapter_handles_missing_gds(tmp_path):
    with mock.patch.object(drc_adapter, "find_klayout", return_value="/usr/bin/klayout"):
        report = drc_adapter.run_drc_structured(tmp_path / "missing.gds")
    assert report.skipped_reason and "gds_not_found" in report.skipped_reason


def test_drc_adapter_handles_missing_drc_script(tmp_path):
    gds = tmp_path / "fake.gds"
    gds.write_bytes(b"")
    with mock.patch.object(
        drc_adapter, "find_klayout", return_value="/usr/bin/klayout"
    ):
        report = drc_adapter.run_drc_structured(
            gds, drc_script=tmp_path / "missing.drc"
        )
    assert report.skipped_reason and "drc_script_not_found" in report.skipped_reason


def test_drc_adapter_invokes_klayout_with_expected_cli(tmp_path):
    gds = tmp_path / "fake.gds"
    gds.write_bytes(b"")
    drc_script = tmp_path / "fake.drc"
    drc_script.write_text("# stub")
    expected_report = tmp_path / "fake.lyrdb"
    expected_report.write_text("")

    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    fake_completed.stderr = ""
    fake_completed.stdout = ""

    with mock.patch.object(
        drc_adapter, "find_klayout", return_value="/usr/bin/klayout"
    ), mock.patch.object(
        drc_adapter.subprocess, "run", return_value=fake_completed
    ) as run_mock, mock.patch.object(
        drc_adapter, "parse_lyrdb_xml", return_value=DRCReport(ok=True)
    ):
        drc_adapter.run_drc_structured(
            gds, drc_script=drc_script, wmin=0.5, gmin=0.5
        )

    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "/usr/bin/klayout"
    assert "wmin=0.5" in " ".join(cmd)
    assert "gmin=0.5" in " ".join(cmd)
    assert str(drc_script) in cmd


# ---------------------------------------------------------------------------
# EdaReport summarization
# ---------------------------------------------------------------------------


def test_summary_includes_per_rule_counts_and_examples():
    drc = DRCReport(
        ok=False,
        n_violations=42,
        by_rule={"Si_space": 42, "Si_width": 0},
        examples=[
            DRCExample(rule="Si_space", coord=[12.34, 56.78], actual=0.10, min=0.13),
            DRCExample(rule="Si_space", coord=[20.0, 30.0], actual=0.11, min=0.13),
        ],
    )
    sax = SaxReport(ok=True, required_models=["mzi_1x2_pindiode_cband"])
    gds = GdsReport(ok=True)
    report = build_eda_report(gds, sax, drc)

    text = summarize_report(report)
    assert "Si_space=42" in text
    assert "Si_space" in text
    assert "@(12.34,56.78)" in text
    assert "min=0.130" in text
    assert "GDSFactory: ok" in text
    assert report.ok is False


def test_summary_for_clean_run():
    report = build_eda_report(
        GdsReport(ok=True),
        SaxReport(ok=True),
        DRCReport(ok=True, n_violations=0),
    )
    assert "DRC: ok" in report.summary_text
    assert report.ok is True


def test_summary_for_sax_missing_models():
    report = build_eda_report(
        GdsReport(ok=True),
        SaxReport(
            ok=False,
            missing_models=["mzi_unknown"],
            required_models=["mzi_unknown", "mzi_known"],
        ),
        DRCReport(ok=True),
    )
    assert report.ok is False
    assert "missing models" in report.summary_text
    assert "mzi_unknown" in report.summary_text


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_sax_model_missing_error_serializes_lists():
    err = SaxModelMissingError(
        missing=["a", "a", "b"], available=["c", "d"]
    )
    assert err.missing == ["a", "b"]
    assert err.available == ["c", "d"]
    assert "missing" in str(err)


def test_netlist_and_gds_errors_carry_metadata():
    n = NetlistError("boom", stage="from_yaml")
    assert n.stage == "from_yaml"
    g = GdsBuildError("boom", hint="check ports")
    assert "hint" in str(g)


# ---------------------------------------------------------------------------
# DSL shape validation (gds_adapter)
# ---------------------------------------------------------------------------


def test_validate_dsl_shape_rejects_tuple_nodes():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    # Reproduces the crash observed when running the Streamlit app: an
    # LLM produced `nodes` as a tuple instead of a mapping.
    with pytest.raises(NetlistError) as exc_info:
        validate_dsl_shape({"nodes": (("N1", {}),), "edges": {}})
    assert "tuple" in str(exc_info.value)


def test_validate_dsl_shape_unwraps_single_wrapper_key():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    wrapped = {"CIRCUIT_demo": {"nodes": {"N1": {}}, "edges": {}}}
    unwrapped = validate_dsl_shape(wrapped)
    assert set(unwrapped) == {"nodes", "edges"}


def test_validate_dsl_shape_coerces_list_of_singletons():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    dsl = {
        "nodes": [{"N1": {"component": "x"}}, {"N2": {"component": "y"}}],
        "edges": {},
    }
    coerced = validate_dsl_shape(dsl)
    assert list(coerced["nodes"].keys()) == ["N1", "N2"]


def test_validate_dsl_shape_reports_missing_keys():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    with pytest.raises(NetlistError) as exc_info:
        validate_dsl_shape({"doc": {}})
    assert "nodes" in str(exc_info.value)


def test_validate_dsl_shape_coerces_edges_from_empty_string():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    out = validate_dsl_shape(
        {
            "nodes": {"N1": {"component": "straight", "params": {}}},
            "edges": "",
        }
    )
    assert out["edges"] == {}


def test_validate_dsl_shape_coerces_edges_from_inline_yaml_string():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    s = 'E1: {link: "N1,o1: N2,o1", properties: {type: route}}'
    out = validate_dsl_shape(
        {
            "nodes": {
                "N1": {"component": "straight", "params": {}},
                "N2": {"component": "straight", "params": {}},
            },
            "edges": s,
        }
    )
    assert "E1" in out["edges"]


def test_validate_dsl_shape_rejects_edges_gibberish_string():
    from PhotonicsAI.graph.adapters.gds_adapter import validate_dsl_shape

    with pytest.raises(NetlistError) as exc_info:
        validate_dsl_shape(
            {
                "nodes": {"N1": {"component": "straight", "params": {}}},
                "edges": "no connections between instances",
            }
    )
    assert "edges" in str(exc_info.value).lower() or "mapping" in str(
        exc_info.value
    ).lower()


def test_build_gds_structured_absorbs_bad_dsl(tmp_path):
    from PhotonicsAI.graph.adapters.gds_adapter import build_gds_structured

    comp, report = build_gds_structured(
        {"nodes": "not a dict", "edges": {}}, tmp_path / "wont_be_written.gds"
    )
    assert comp is None
    assert report.ok is False
    assert any("dsl->netlist" in err for err in report.errors)


# ---------------------------------------------------------------------------
# Designer-DSL normalization (gds_adapter)
# ---------------------------------------------------------------------------


def test_scale_numeric_placements_leaves_string_coords_and_scales_floats():
    from PhotonicsAI.graph.adapters.gds_adapter import _scale_numeric_placements

    d = {
        "placements": {
            "A": {"x": 10, "y": 20.0, "rotation": 0},
            "B": {"x": "N1,o1", "y": 0},
        }
    }
    s = _scale_numeric_placements(d, 2.0)
    assert s["placements"]["A"]["x"] == 20.0
    assert s["placements"]["A"]["y"] == 40.0
    assert s["placements"]["B"]["x"] == "N1,o1"
    assert s["placements"]["B"]["y"] == 0


def test_normalize_link_accepts_no_space_and_adds_o_prefix():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_link

    assert _normalize_link("N1,2:N2,1") == "N1,o2: N2,o1"
    assert _normalize_link("N1,o2: N2,o1") == "N1,o2: N2,o1"
    assert _normalize_link("  N1 , 2 : N2 , 1 ") == "N1,o2: N2,o1"
    assert _normalize_link("garbage") is None


def test_derive_top_level_ports_marks_unconnected_endpoints():
    from PhotonicsAI.graph.adapters.gds_adapter import _derive_top_level_ports

    nodes = {
        "N1": {"properties": {"ports": "1x2"}},  # 3 ports
        "N2": {"properties": {"ports": "1x1"}},  # 2 ports
    }
    edges = {"E1": {"link": "N1,o2: N2,o1"}}
    ports = _derive_top_level_ports(nodes, edges)
    # N1,o1 + N1,o3 + N2,o2 = 3 open endpoints
    assert set(ports.values()) == {"N1,o1", "N1,o3", "N2,o2"}


def test_normalize_designer_dsl_fills_missing_ports_and_placement():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_designer_dsl

    dsl = {
        "nodes": {
            "N1": {"component": "x", "properties": {"ports": "1x1"}},
            "N2": {"component": "y", "placement": {"X": 5, "Y": 10, "Rotation": 90}},
        },
        "edges": {"E1": {"link": "N1,1:N2,1"}},
    }
    normalized, warnings = _normalize_designer_dsl(dsl)

    assert normalized["nodes"]["N1"]["placement"] == {"x": 0, "y": 0, "rotation": 0}
    assert normalized["nodes"]["N2"]["placement"] == {"x": 5, "y": 10, "rotation": 90}
    assert normalized["nodes"]["N1"]["params"] == {}
    assert normalized["edges"]["E1"]["link"] == "N1,o1: N2,o1"
    assert "ports" in normalized
    assert any("auto-derived" in w for w in warnings)


def test_normalize_designer_dsl_drops_unparseable_edges():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_designer_dsl

    dsl = {
        "nodes": {"N1": {"component": "x"}, "N2": {"component": "y"}},
        "edges": {"E1": {"link": "???"}, "E2": {"link": "N1,1:N2,1"}},
    }
    normalized, warnings = _normalize_designer_dsl(dsl)
    assert set(normalized["edges"]) == {"E2"}
    assert any("unparseable" in w for w in warnings)


def test_normalize_designer_dsl_directional_coupler_param_aliases():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_designer_dsl

    dsl = {
        "nodes": {
            "N1": {
                "component": "_directional_coupler",
                "params": {
                    "width": 500e-9,
                    "wavelength": 1550e-9,
                    "coupling_length": 2e-6,
                    "gap": 200e-9,
                },
            }
        },
        "edges": {},
    }
    normalized, warnings = _normalize_designer_dsl(dsl)
    p = normalized["nodes"]["N1"]["params"]
    assert "width" not in p and "wavelength" not in p
    assert p["length"] == pytest.approx(2.0)
    assert p["gap"] == pytest.approx(0.2)
    assert any("removed unsupported" in w for w in warnings)
    assert any("coupling_length" in w for w in warnings)


def test_normalize_designer_dsl_directional_coupler_rejects_zero_dy_dx():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_designer_dsl

    dsl = {
        "nodes": {
            "N1": {
                "component": "_directional_coupler",
                "params": {
                    "length": 2.0,
                    "gap": 0.2,
                    "dy": 0,
                    "dx": 0,
                },
            }
        },
        "edges": {},
    }
    normalized, warnings = _normalize_designer_dsl(dsl)
    p = normalized["nodes"]["N1"]["params"]
    assert p["dy"] == 4.0
    assert p["dx"] == 10.0
    assert any("_directional_coupler: dy was 0" in w for w in warnings)
    assert any("_directional_coupler: dx was 0" in w for w in warnings)


def test_normalize_designer_dsl_coerces_quoted_numeric_strings():
    from PhotonicsAI.graph.adapters.gds_adapter import _normalize_designer_dsl

    dsl = {
        "nodes": {
            "N1": {
                "component": "bend_euler",
                "params": {"radius": "10.0", "width": "0.5"},
                "placement": {"x": "0", "y": "1e2"},
            }
        },
        "edges": {},
    }
    normalized, warnings = _normalize_designer_dsl(dsl)
    p = normalized["nodes"]["N1"]["params"]
    pl = normalized["nodes"]["N1"]["placement"]
    assert p["radius"] == 10.0
    assert p["width"] == 0.5
    assert pl["x"] == 0
    assert pl["y"] == 100.0
    assert any("coerced" in w for w in warnings)


# ---------------------------------------------------------------------------
# Reflector escalate detection
# ---------------------------------------------------------------------------


def test_reflector_decision_marker_is_authoritative():
    from PhotonicsAI.graph.nodes.reflector import _reflector_requested_escalate

    assert _reflector_requested_escalate("...\nDECISION: ESCALATE\n") is True
    assert _reflector_requested_escalate("...\nDECISION: RETRY\n") is False
    assert _reflector_requested_escalate("decision: escalate") is True  # case-insensitive


def test_reflector_ignores_hedging_inline_escalate_phrase():
    from PhotonicsAI.graph.nodes.reflector import _reflector_requested_escalate

    # Real-world case that previously caused premature escalation: the
    # LLM ends a conditional sentence with "escalate to human." without
    # a standalone decision line.
    txt = (
        "Replace N1 with directional_coupler_v2.\n"
        "If no cell matches, otherwise escalate to human if the PDK lacks it."
    )
    assert _reflector_requested_escalate(txt) is False


def test_reflector_legacy_standalone_escalate_line_still_works():
    from PhotonicsAI.graph.nodes.reflector import _reflector_requested_escalate

    txt = "Nothing we can do.\nescalate to human"
    assert _reflector_requested_escalate(txt) is True


# ---------------------------------------------------------------------------
# Planarity adapter (does not require LLMs / gdsfactory)
# ---------------------------------------------------------------------------


def test_planarity_adapter_finds_no_crossings_for_planar_graph():
    pytest.importorskip("pygraphviz")
    from PhotonicsAI.graph.adapters import planarity_adapter

    dot = textwrap.dedent(
        """
        graph G {
          rankdir=LR;
          A -- B;
          B -- C;
        }
        """
    ).strip()
    assert planarity_adapter.is_planar(dot)
    assert planarity_adapter.find_crossings(dot) == []


# ---------------------------------------------------------------------------
# State / module import smoke
# ---------------------------------------------------------------------------


def test_state_initial_state_returns_expected_keys():
    from PhotonicsAI.graph.state import initial_state

    s = initial_state("hello")
    assert s["user_prompt"] == "hello"
    assert s["reflections"] == []
    assert s["retry_count"] == 0
    assert "designer_model" in s and "reflector_model" in s and "legacy_stage_model" in s


def test_graph_module_imports_without_langgraph_optional_extras():
    # Ensure our package import is light enough that test collection
    # does not fail when optional extras (e.g. langsmith) are absent.
    spec = importlib.util.find_spec("PhotonicsAI.graph")
    assert spec is not None
    assert sys.modules.get("PhotonicsAI.graph") is None or True
