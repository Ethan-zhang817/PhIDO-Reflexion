"""EDA executor node.

Runs ``gds -> sax -> drc`` strictly in that order. Each stage records a
sub-report into ``EdaReport``; a failure in stage *N* does not skip
stage *N+1* unless the input artifact is unusable. This guarantees the
Reflector sees a comprehensive snapshot, not just the first error.

The DRC subprocess is offloaded to a worker thread so a Streamlit UI
calling ``graph.invoke`` does not freeze for the duration of KLayout's
~60 s timeout.
"""

from __future__ import annotations

import concurrent.futures
import time
import uuid
from pathlib import Path

import numpy as np

from PhotonicsAI.config import PATH
from PhotonicsAI.graph.adapters.drc_adapter import run_drc_structured
from PhotonicsAI.graph.adapters.eda_report import (
    DRCReport,
    GdsReport,
    SaxReport,
    build_eda_report,
)
from PhotonicsAI.graph.adapters.errors import SaxModelMissingError
from PhotonicsAI.graph.adapters.gds_adapter import build_gds_structured
from PhotonicsAI.graph.adapters.sax_adapter import run_sax_structured
from PhotonicsAI.graph.state import PhIDOState


_DRC_DIR = PATH.build / "drc_runs"
_DRC_DIR.mkdir(parents=True, exist_ok=True)


def _gds_path_for(thread_id: str) -> Path:
    name = f"{thread_id or 'anon'}_{uuid.uuid4().hex[:8]}.gds"
    return _DRC_DIR / name


def _run_drc_async(gds_path: Path, *, timeout: int = 60) -> DRCReport:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_drc_structured, gds_path, timeout=timeout)
        try:
            return future.result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError:
            return DRCReport(
                ok=False,
                skipped_reason=f"drc_thread_timeout_after_{timeout + 5}s",
                raw_report_path=str(gds_path.with_suffix(".lyrdb")),
            )


def eda_executor_node(state: PhIDOState) -> dict:
    """Run gds -> sax -> drc and persist a structured report into state."""
    timings: dict[str, float] = {}
    dsl = state.get("circuit_dsl")
    if not dsl:
        gds = GdsReport(ok=False, errors=["no circuit_dsl in state"])
        sax = SaxReport(ok=False, errors=["skipped: no DSL"])
        drc = DRCReport(ok=False, skipped_reason="no DSL")
        report = build_eda_report(gds, sax, drc)
        return {"eda_report": report, "timings": timings}

    # ---- Stage 1: GDS -------------------------------------------------
    gds_path = _gds_path_for(state.get("thread_id", ""))
    t0 = time.perf_counter()
    component, gds_report = build_gds_structured(dsl, gds_path)
    timings["gds"] = time.perf_counter() - t0

    # ---- Stage 2: SAX -------------------------------------------------
    t0 = time.perf_counter()
    sax_report = SaxReport(ok=False, errors=["skipped: no gdsfactory component"])
    sax_circuit = None
    sax_result = None
    if component is not None:
        try:
            sax_report, sax_circuit = run_sax_structured(component)
        except SaxModelMissingError as exc:
            sax_report = SaxReport(
                ok=False,
                missing_models=exc.missing,
                required_models=exc.missing,  # at least, full list when available
                errors=[str(exc)],
            )
            sax_circuit = None
        except Exception as exc:  # noqa: BLE001
            sax_report = SaxReport(ok=False, errors=[f"sax adapter crashed: {exc}"])
    timings["sax"] = time.perf_counter() - t0

    if sax_circuit is not None:
        try:
            wl = np.linspace(1.5, 1.6, 200)
            sax_result = sax_circuit(wl=wl)
        except Exception as exc:  # noqa: BLE001
            sax_report.errors.append(f"sax sweep failed: {exc}")

    # ---- Stage 3: DRC -------------------------------------------------
    t0 = time.perf_counter()
    if gds_report.gds_path:
        drc_report = _run_drc_async(Path(gds_report.gds_path))
    else:
        drc_report = DRCReport(
            ok=False,
            skipped_reason="no gds file written; cannot run DRC",
        )
    timings["drc"] = time.perf_counter() - t0

    # ---- Aggregate ----------------------------------------------------
    report = build_eda_report(gds_report, sax_report, drc_report)
    return {
        "eda_report": report,
        "gds_path": gds_report.gds_path,
        "sax_result": sax_result,
        "timings": timings,
    }
