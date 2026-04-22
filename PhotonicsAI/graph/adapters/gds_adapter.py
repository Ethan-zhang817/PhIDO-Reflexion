"""GDSFactory build adapter.

Three responsibilities:

1. Convert a circuit DSL (the format used by ``CIRCUIT_*.yaml``) into a
   GDSFactory netlist via the existing :func:`PhotonicsAI.Photon.utils.dsl_to_gf`.
2. Build the ``gf.Component`` from that netlist with structured error
   handling (routing failure vs. layer-overflow vs. other).
3. Write a GDS file to disk for the DRC stage to consume.

Failures populate :class:`GdsReport` rather than crashing the agent.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gdsfactory as gf
import yaml

from PhotonicsAI.graph.adapters.eda_report import GdsReport


def dsl_to_netlist(circuit_dsl: dict) -> dict:
    """Convert a circuit DSL dict into a GDSFactory netlist dict.

    Wraps :func:`PhotonicsAI.Photon.utils.dsl_to_gf` so the rest of the
    graph package does not import the legacy module directly. We make a
    deep copy because ``dsl_to_gf`` mutates its input in places.
    """
    from PhotonicsAI.Photon import utils as photon_utils

    return photon_utils.dsl_to_gf(copy.deepcopy(circuit_dsl))


def _normalize_netlist(netlist: dict) -> dict:
    """Strip non-gdsfactory keys and ensure a minimal ``placements`` block.

    Mirrors the cleanup done in :func:`DemoPDK.yaml_netlist_to_gds`
    (drops ``reasoning`` / ``comments``, fills empty placements with a
    diagonal stagger).
    """
    data = copy.deepcopy(netlist)
    for key in ("reasoning", "comments"):
        data.pop(key, None)

    if not data.get("placements"):
        data["placements"] = {}
        x = y = 0
        for instance in data.get("instances", {}):
            data["placements"][instance] = {"x": x, "y": y}
            x += 10
            y += 10
    return data


def build_component(
    netlist: dict,
    *,
    ignore_links: bool = False,
) -> tuple[Any, GdsReport]:
    """Build a ``gf.Component`` from a GDSFactory netlist dict.

    Returns ``(component_or_None, report)``. The report is updated to
    reflect routing / layer issues; the component is ``None`` only when
    even the ``ignore_links`` fallback fails.
    """
    report = GdsReport(ok=True)
    data = _normalize_netlist(netlist)

    if ignore_links and "routes" in data:
        data = copy.deepcopy(data)
        data.pop("routes", None)

    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    try:
        component = gf.read.from_yaml(yaml_text)
        return component, report
    except Exception as exc:
        message = str(exc)
        report.ok = False
        report.errors.append(f"gf.read.from_yaml failed: {message}")
        if "layer numbers larger than 65535" in message:
            report.layer_overflow = True
        if "route" in message.lower() or "port" in message.lower():
            report.routing_failed = True

    if not ignore_links:
        try:
            data_no_routes = copy.deepcopy(data)
            data_no_routes.pop("routes", None)
            yaml_text2 = yaml.dump(
                data_no_routes, default_flow_style=False, sort_keys=False
            )
            component = gf.read.from_yaml(yaml_text2)
            report.routing_failed = True
            report.errors.append(
                "fell back to ignore_links=True; routing must be fixed"
            )
            return component, report
        except Exception as exc2:
            report.errors.append(f"ignore_links fallback failed: {exc2}")

    return None, report


def write_gds(component, gds_path: str | Path) -> tuple[bool, list[str]]:
    """Write ``component`` to ``gds_path`` with the legacy fallback chain.

    Returns ``(success, errors)``. Errors are accumulated rather than
    raised so the GDS report can absorb partial failures.
    """
    gds_path = Path(gds_path)
    gds_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        component.write_gds(str(gds_path))
        return True, errors
    except Exception as exc:
        errors.append(f"write_gds failed: {exc}")
        if "layer numbers larger than 65535" not in str(exc):
            return False, errors

    try:
        flattened = component.flatten()
        if flattened is not None:
            flattened.write_gds(str(gds_path))
            return True, errors
    except Exception as exc2:
        errors.append(f"flatten().write_gds failed: {exc2}")

    try:
        component.write_gds(str(gds_path), max_points=None)
        return True, errors
    except Exception as exc3:
        errors.append(f"write_gds(max_points=None) failed: {exc3}")

    return False, errors


def build_gds_structured(
    circuit_dsl: dict,
    gds_path: str | Path,
) -> tuple[Any | None, GdsReport]:
    """End-to-end DSL -> GDS conversion with a structured report.

    Returns ``(component_or_None, GdsReport)``. The report's
    ``gds_path`` is set when the GDS file is successfully written.
    """
    netlist = dsl_to_netlist(circuit_dsl)
    component, report = build_component(netlist, ignore_links=False)
    if component is None:
        return None, report

    success, write_errors = write_gds(component, gds_path)
    report.errors.extend(write_errors)
    if success:
        report.gds_path = str(gds_path)
    else:
        report.ok = False
    return component, report
