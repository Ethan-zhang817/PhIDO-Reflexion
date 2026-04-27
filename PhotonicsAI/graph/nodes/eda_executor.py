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
import ormsgpack

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


def _sax_result_to_jsonable(result: object | None) -> object | None:
    """Convert SAX sweep output to plain-Python / msgpack-safe data.

    SAX / JAX may leave ``ArrayImpl`` leaves that are **not** ``isinstance(…, np.ndarray)``;
    LangGraph checkpointers then fail in ``ormsgpack.packb``. We normalize every
    tree leaf to built-ins and verify with the same pack step the runtime uses.
    S 参数常为复数，一并转为可序列化结构。
    """
    if result is None:
        return None
    try:
        import jax

        result = jax.device_get(result)
    except Exception:  # noqa: BLE001
        pass

    def _coerce_tolist_complexs(obj: object) -> object:
        if isinstance(obj, complex):
            return {"re": float(obj.real), "im": float(obj.imag)}
        if isinstance(obj, list):
            return [_coerce_tolist_complexs(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _coerce_tolist_complexs(v) for k, v in obj.items()}
        return obj

    def _to_plain_leaf(x: object) -> object:
        try:
            import jax

            if isinstance(x, jax.Array):
                return _to_plain_leaf(np.asarray(x))
        except Exception:  # noqa: BLE001
            pass
        if isinstance(x, np.ndarray):
            return _coerce_tolist_complexs(x.tolist())
        if isinstance(x, (np.floating, np.integer)):
            return float(x) if isinstance(x, np.floating) else int(x)
        if isinstance(x, np.complexfloating):
            return {"re": float(x.real), "im": float(x.imag)}
        if isinstance(x, (float, int, str, bool)) or x is None:
            return x
        if isinstance(x, complex):
            return {"re": float(x.real), "im": float(x.imag)}
        if hasattr(x, "shape") and hasattr(x, "__array__") and not isinstance(
            x, (dict, list, tuple, str, bytes)
        ):
            return _to_plain_leaf(np.asarray(x))
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)

    def _convert(x: object) -> object:
        if isinstance(x, dict):
            out: dict[object, object] = {}
            for k, v in x.items():
                if isinstance(k, tuple):
                    key = (
                        f"{k[0]},{k[1]}"
                        if len(k) == 2
                        else ",".join(str(p) for p in k)
                    )
                else:
                    key = k
                out[key] = _convert(v)
            return out
        if isinstance(x, (list, tuple)):
            return [_convert(i) for i in x]
        return _to_plain_leaf(x)

    try:
        out = _coerce_tolist_complexs(_convert(result))
        ormsgpack.packb(out)
    except Exception:  # noqa: BLE001
        return None
    return out


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
                warnings=getattr(exc, "diagnostics", []),
            )
            sax_circuit = None
        except Exception as exc:  # noqa: BLE001
            sax_report = SaxReport(ok=False, errors=[f"sax adapter crashed: {exc}"])
    timings["sax"] = time.perf_counter() - t0

    sax_png_path: str | None = None
    if sax_circuit is not None:
        try:
            wl = np.linspace(1.5, 1.6, 200)
            sax_result = sax_circuit(wl=wl)
        except Exception as exc:  # noqa: BLE001
            sax_report.errors.append(f"sax sweep failed: {exc}")
        else:
            try:
                from PhotonicsAI.Photon import utils as _legacy_utils

                _legacy_utils.plot_dict_arrays(wl, sax_result)
                _expected = PATH.build / "plot_sax.png"
                if _expected.exists():
                    sax_png_path = str(_expected)
            except Exception as exc:  # noqa: BLE001
                sax_report.errors.append(f"sax plot failed: {exc}")

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
    # Never persist the raw SAX pytree (JAX ``ArrayImpl``) in graph state: even
    # after tree→list conversion, LangGraph's serde (ormsgpack options + Pydantic
    # round-trips) can still throw before the stream yields, so the UI never
    # shows DRC. Curves: ``sax_png_path`` + build/plot_sax.png; structured DRC: ``eda_report``.
    return {
        "eda_report": report,
        "gds_path": gds_report.gds_path,
        "gds_png_path": gds_report.gds_png_path,
        "sax_result": None,
        "sax_png_path": sax_png_path,
        "timings": timings,
    }
