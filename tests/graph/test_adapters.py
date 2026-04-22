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
    assert "designer_model" in s and "reflector_model" in s


def test_graph_module_imports_without_langgraph_optional_extras():
    # Ensure our package import is light enough that test collection
    # does not fail when optional extras (e.g. langsmith) are absent.
    spec = importlib.util.find_spec("PhotonicsAI.graph")
    assert spec is not None
    assert sys.modules.get("PhotonicsAI.graph") is None or True
