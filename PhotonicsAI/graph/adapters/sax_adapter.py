"""SAX simulation adapter.

Wraps :func:`sax.circuit` so that missing models raise
:class:`SaxModelMissingError` instead of being silently printed. The
adapter is read-only on the legacy ``DemoPDK`` registry — it does not
mutate ``DemoPDK.all_models``.
"""

from __future__ import annotations

from typing import Any

import sax

from PhotonicsAI.graph.adapters.eda_report import SaxReport
from PhotonicsAI.graph.adapters.errors import NetlistError, SaxModelMissingError


def _get_all_models() -> dict[str, Any]:
    """Return the ``all_models`` dict registered by ``DemoPDK``.

    Imported lazily to avoid pulling in heavy deps (gdsfactory, jax, ...)
    at module import time, which would slow down unit tests that mock
    this layer.
    """
    from PhotonicsAI.Photon.DemoPDK import all_models  # type: ignore[import-not-found]

    return all_models


def collect_required_models(component) -> tuple[list[str], dict]:
    """Compute SAX required models from a ``gf.Component``.

    Mirrors :func:`PhotonicsAI.Photon.DemoPDK.yaml_netlist_to_gds` but
    returns rather than printing. A second value is the recursive
    netlist dict, returned so callers can pass it back into
    :func:`sax.circuit` without recomputing.
    """
    try:
        netlist = component.get_netlist(recursive=True)
        required = sax.get_required_circuit_models(netlist)
    except Exception as exc:
        raise NetlistError(
            f"recursive get_netlist failed: {exc}", stage="get_netlist"
        ) from exc
    return list(required), netlist


def run_sax_structured(
    component,
    *,
    backend: str = "default",
) -> tuple[SaxReport, Any | None]:
    """Run SAX over ``component`` and return a structured report.

    Returns a tuple ``(report, circuit_callable_or_None)`` so that the
    executor can invoke the circuit on a wavelength sweep separately.

    Raises:
        SaxModelMissingError: when one or more required models are not
            registered. ``SaxReport.missing_models`` is populated and
            the second tuple element is ``None`` so callers can choose
            to propagate vs. degrade gracefully.
    """
    available = _get_all_models()
    try:
        required, netlist = collect_required_models(component)
    except NetlistError as exc:
        return (
            SaxReport(ok=False, errors=[str(exc)]),
            None,
        )

    missing = [name for name in required if name not in available]
    if missing:
        report = SaxReport(
            ok=False,
            missing_models=sorted(set(missing)),
            required_models=required,
        )
        # The error is *raised* (not just returned) so the executor can
        # decide whether to swallow it. We still surface a populated
        # report via the exception, mirroring the SaxModelMissingError
        # contract from REFACTOR_PLAN.md §4.2.
        raise SaxModelMissingError(missing=missing, available=list(available))

    try:
        circuit, _info = sax.circuit(netlist, available, backend=backend)
    except Exception as exc:
        return (
            SaxReport(
                ok=False,
                errors=[f"sax.circuit failed: {exc}"],
                required_models=required,
            ),
            None,
        )

    return (
        SaxReport(ok=True, required_models=required),
        circuit,
    )
