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
import importlib
import re
from pathlib import Path
from typing import Any

import yaml

from PhotonicsAI.graph.adapters.eda_report import GdsReport
from PhotonicsAI.graph.adapters.errors import NetlistError


_REQUIRED_TOP_KEYS = ("nodes", "edges")

# Matches a port reference like "N1,2", "N1,o2" or "N1, o2" inside an edge link.
_PORT_TOKEN_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*,\s*(o?\d+)\s*$")


def _unwrap_single_wrapper(circuit_dsl: dict) -> dict:
    """Unwrap ``{CIRCUIT_name: {...}}`` style wrappers emitted by LLMs.

    The legacy PhIDO YAML files use ``CIRCUIT_wdd0: {doc, nodes, ...}``
    as a top-level namespace, so an LLM trained on those will often
    emit the same shape. We transparently peel it off here.
    """
    if len(circuit_dsl) == 1:
        sole_value = next(iter(circuit_dsl.values()))
        if isinstance(sole_value, dict) and any(
            k in sole_value for k in _REQUIRED_TOP_KEYS
        ):
            return sole_value
    return circuit_dsl


def _coerce_nodes(value: Any) -> dict | None:
    """Accept ``dict`` directly or a list of ``{id: {...}}`` single-key dicts."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        merged: dict = {}
        for item in value:
            if not isinstance(item, dict) or len(item) != 1:
                return None
            merged.update(item)
        return merged
    return None


def _coerce_mapping_field(
    value: Any, *, field_name: str, empty_fallback: bool = True
) -> dict | None:
    """Best-effort: turn LLM drifts (``str``, ``None``) into a ``dict`` for top-level
    ``nodes`` / ``edges``.

    * ``None`` or ``""`` → ``{}`` when *empty_fallback* is set.
    * A *string* that is valid YAML (inline mapping or multiline) → parsed dict.
    """
    if isinstance(value, dict):
        return value
    if value is None and empty_fallback:
        return {}
    if isinstance(value, str):
        s = value.strip()
        if not s and empty_fallback:
            return {}
        if s in ("{}", "null", "None", "[]"):
            return {}
        try:
            loaded = yaml.safe_load(s)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(loaded, dict):
            return loaded
    if isinstance(value, (list, tuple)):
        return _coerce_nodes(value)
    return None


def validate_dsl_shape(circuit_dsl: Any) -> dict:
    """Return a shape-valid DSL or raise :class:`NetlistError`.

    Accepts common LLM drifts (single wrapper key, ``nodes`` as a list
    of singletons) but rejects anything that cannot be coerced into a
    ``{nodes: dict, edges: dict, ...}`` shape.
    """
    if not isinstance(circuit_dsl, dict):
        raise NetlistError(
            f"DSL root must be a mapping, got {type(circuit_dsl).__name__}",
            stage="shape",
        )

    dsl = _unwrap_single_wrapper(circuit_dsl)

    for required in _REQUIRED_TOP_KEYS:
        if required not in dsl:
            raise NetlistError(
                f"DSL is missing required top-level key '{required}'. "
                f"Got keys: {sorted(dsl.keys())}",
                stage="shape",
            )

    coerced_nodes = _coerce_nodes(dsl["nodes"])
    if coerced_nodes is None:
        coerced_nodes = _coerce_mapping_field(dsl["nodes"], field_name="nodes")
    if coerced_nodes is None:
        raise NetlistError(
            "DSL['nodes'] must be a mapping of id -> node (or a list "
            "of single-key dicts); got "
            f"{type(dsl['nodes']).__name__}",
            stage="shape",
        )
    dsl["nodes"] = coerced_nodes

    coerced_edges = _coerce_nodes(dsl["edges"])
    if coerced_edges is None:
        coerced_edges = _coerce_mapping_field(dsl["edges"], field_name="edges")
    if coerced_edges is None:
        raise NetlistError(
            "DSL['edges'] must be a mapping of id -> edge. "
            "For no inter-instance routes use a YAML mapping, e.g. `edges: {}` "
            "(not a string or prose). "
            f"Got {type(dsl['edges']).__name__!s}.",
            stage="shape",
        )
    dsl["edges"] = coerced_edges

    return dsl


def _parse_port_count(value: Any) -> int:
    """Parse a ``properties.ports`` spec like ``"1x2"`` into total port count.

    Accepts strings (``"1x2"``), ints, tuples/lists ``(1, 2)``. Falls back
    to 2 when the shape is unrecognizable.
    """
    if isinstance(value, int):
        return max(int(value), 1)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return max(int(value[0]) + int(value[1]), 1)
        except (TypeError, ValueError):
            return 2
    if isinstance(value, str) and "x" in value.lower():
        try:
            left, right = value.lower().split("x", 1)
            return max(int(left.strip()) + int(right.strip()), 1)
        except (TypeError, ValueError):
            return 2
    return 2


def _normalize_link_endpoint(endpoint: str) -> str | None:
    """Canonicalize one side of a ``link`` (``"N1,2"`` → ``"N1,o2"``)."""
    match = _PORT_TOKEN_RE.match(endpoint)
    if not match:
        return None
    node, port = match.group(1), match.group(2)
    if not port.startswith("o"):
        port = f"o{port}"
    return f"{node},{port}"


def _normalize_link(link: str) -> str | None:
    """Return a canonical ``"src,oN: dst,oM"`` or ``None`` when unparseable."""
    if not isinstance(link, str):
        return None
    # Legacy sample YAMLs use "N1,2:N2,1" (no space). ``dsl_to_gf``
    # however splits on ``": "``, so we always normalize to the
    # space-after-colon form. The endpoints themselves have exactly
    # one comma, so the separating ``:`` between them is the *last*
    # ``:`` in the string after the first comma of each endpoint.
    raw = link.strip()
    # Accept and canonicalize both "a:b" and "a: b".
    if ": " in raw:
        left, right = raw.split(": ", 1)
    elif ":" in raw:
        left, right = raw.split(":", 1)
    else:
        return None
    left_c = _normalize_link_endpoint(left)
    right_c = _normalize_link_endpoint(right)
    if not left_c or not right_c:
        return None
    return f"{left_c}: {right_c}"


_SI_METRE_CEILING = 1e-3
"""Values with ``0 < abs(v) <= _SI_METRE_CEILING`` are treated as SI metres and
converted to micrometres for layout params (typical ``2e-6`` m → ``2`` µm)."""


def _coerce_scalar_to_number(value: Any) -> Any:
    """Turn YAML string numerals (``'2.0'``, ``'2e-6'``) into ``int`` / ``float``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            v = float(s)
        except ValueError:
            return value
        if v == int(v) and "e" not in s.lower() and "E" not in s and "." not in s:
            return int(s)
        return v
    return value


def _coerce_mapping_numbers(
    d: dict[str, Any],
    *,
    label: str,
    warnings: list[str] | None = None,
) -> None:
    """In-place: numeric-looking string values become int/float (mutates *d*)."""
    for k, v in list(d.items()):
        if isinstance(v, dict):
            continue
        if isinstance(v, str):
            new_v = _coerce_scalar_to_number(v)
            if not isinstance(new_v, str):
                d[k] = new_v
                if warnings is not None:
                    warnings.append(
                        f"{label}: coerced {k!r} from string {v!r} to "
                        f"{type(new_v).__name__}"
                    )


def _maybe_si_metres_to_um(value: float) -> float:
    if value == 0.0:
        return 0.0
    a = abs(value)
    if 1e-12 < a <= _SI_METRE_CEILING:
        return float(value) * 1e6
    return float(value)


def _normalize_params_directional_coupler(
    params: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Align LLM drift with :func:`PhotonicsAI.KnowledgeBase.DesignLibrary._directional_coupler._directional_coupler`.

    That cell exposes ``length``, ``gap``, ``dy``, ``dx`` in **micrometres** (and
    forwards to :func:`gdsfactory.components.coupler.coupler`). LLMs often emit
    ``width``/``wavelength`` or ``coupling_length`` in SI metres.
    """
    out: dict[str, Any] = dict(params)
    drop = (
        "width",
        "waveguide_width",
        "wavelength",
        "wl",
    )
    for k in drop:
        if k in out:
            out.pop(k)
            warnings.append(
                "_directional_coupler: removed unsupported "
                f"param {k!r} (this cell uses length, gap, dy, dx in µm)"
            )
    if "coupling_length" in out and "length" not in out:
        out["length"] = out.pop("coupling_length")
        warnings.append("_directional_coupler: renamed coupling_length -> length")
    elif "coupling_length" in out:
        out.pop("coupling_length", None)
        warnings.append(
            "_directional_coupler: dropped coupling_length (length already set)"
        )
    for key in ("length", "gap"):
        if key in out and isinstance(out[key], (int, float)):
            old = float(out[key])
            new = _maybe_si_metres_to_um(old)
            if new != old:
                out[key] = new
                warnings.append(
                    f"_directional_coupler: {key} converted SI m → µm "
                    f"({old!r} -> {new!r})"
                )
    # ``dy`` / ``dx`` are vertical port pitch and bend extent (µm). LLMs often
    # set them to 0 "to minimize footprint"; :func:`gdsfactory.components.coupler.coupler`
    # then fails internally (``list index out of range``) and the layout is wrong.
    _dc_default_dy = 4.0
    _dc_default_dx = 10.0
    for key, dflt in (("dy", _dc_default_dy), ("dx", _dc_default_dx)):
        if key not in out:
            continue
        try:
            v = float(out[key])
        except (TypeError, ValueError):
            continue
        if abs(v) < 1e-9:
            out[key] = dflt
            warnings.append(
                f"_directional_coupler: {key} was 0 (invalid for coupler "
                f"geometry); replaced with default {dflt} µm"
            )
    return out


def _apply_cell_param_hygiene(
    component: str,
    params: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    if component == "_directional_coupler":
        return _normalize_params_directional_coupler(params, warnings)
    return params


def _normalize_placement(node: dict) -> dict:
    """Return a lowercase ``{x, y, rotation}`` mapping with safe defaults."""
    placement = node.get("placement") or {}
    if not isinstance(placement, dict):
        placement = {}
    # Accept legacy capitalization (X/Y/Rotation) as well as lowercase.
    def _pick(*keys):
        for k in keys:
            if k in placement and placement[k] is not None:
                return placement[k]
        return 0

    return {
        "x": _pick("x", "X"),
        "y": _pick("y", "Y"),
        "rotation": _pick("rotation", "Rotation"),
    }


def _derive_top_level_ports(nodes: dict, edges: dict) -> dict[str, str]:
    """Derive top-level ``ports`` from open ports on instances.

    Mirrors :func:`PhotonicsAI.Photon.utils.add_final_ports` but works
    directly on the structured DSL instead of a DOT string.
    """
    used: set[str] = set()
    for edge in edges.values():
        link = (edge or {}).get("link")
        normalized = _normalize_link(link) if isinstance(link, str) else None
        if not normalized:
            continue
        a, b = normalized.split(": ", 1)
        used.add(a.strip())
        used.add(b.strip())

    open_endpoints: list[str] = []
    for node_id, node in nodes.items():
        props = (node or {}).get("properties") or {}
        n_ports = _parse_port_count(props.get("ports"))
        for p in range(1, n_ports + 1):
            endpoint = f"{node_id},o{p}"
            if endpoint not in used:
                open_endpoints.append(endpoint)

    return {f"o{i + 1}": ep for i, ep in enumerate(open_endpoints)}


def _normalize_designer_dsl(dsl: dict) -> tuple[dict, list[str]]:
    """Make an LLM-authored DSL palatable to the legacy ``dsl_to_gf``.

    Returns ``(normalized_dsl, warnings)``. Applies the following
    transformations without mutating the input:

    * Ensure each node has ``params``, ``placement{x,y,rotation}``,
      and ``properties{ports}`` (filled with sensible defaults).
    * Canonicalize edge ``link`` strings to ``"src,oN: dst,oM"``.
    * Auto-derive a top-level ``ports`` mapping from open ports when
      missing, mirroring the legacy ``add_final_ports`` behavior.
    * Coerce quoted numeric strings in ``params`` / ``placement`` to real
      numbers so gdsfactory never subtracts a ``str`` from a ``float``.
    """
    warnings: list[str] = []
    dsl = copy.deepcopy(dsl)

    # ---- Nodes --------------------------------------------------------
    for node_id, node in list(dsl.get("nodes", {}).items()):
        if not isinstance(node, dict):
            warnings.append(f"node[{node_id}] is not a mapping; skipped")
            continue
        node.setdefault("params", {})
        if isinstance(node.get("params"), dict):
            _coerce_mapping_numbers(
                node["params"],
                label=f"node[{node_id}].params",
                warnings=warnings,
            )
        comp = node.get("component")
        if isinstance(comp, str) and isinstance(node.get("params"), dict):
            node["params"] = _apply_cell_param_hygiene(comp, node["params"], warnings)
        props = node.get("properties")
        if not isinstance(props, dict):
            props = {}
        props.setdefault("ports", "1x1")
        node["properties"] = props
        node["placement"] = _normalize_placement(node)
        if isinstance(node.get("placement"), dict):
            _coerce_mapping_numbers(
                node["placement"],
                label=f"node[{node_id}].placement",
                warnings=warnings,
            )
        dsl["nodes"][node_id] = node

    # ---- Edges --------------------------------------------------------
    bad_edges: list[str] = []
    for edge_id, edge in list(dsl.get("edges", {}).items()):
        if not isinstance(edge, dict):
            bad_edges.append(edge_id)
            continue
        link = edge.get("link")
        canon = _normalize_link(link) if isinstance(link, str) else None
        if canon is None:
            bad_edges.append(edge_id)
            warnings.append(
                f"edge[{edge_id}] has unparseable link={link!r}; dropped"
            )
            continue
        edge["link"] = canon
        dsl["edges"][edge_id] = edge
    for edge_id in bad_edges:
        dsl["edges"].pop(edge_id, None)

    # ---- Top-level ports ---------------------------------------------
    existing_ports = dsl.get("ports")
    if not isinstance(existing_ports, dict) or not existing_ports:
        derived = _derive_top_level_ports(dsl.get("nodes", {}), dsl.get("edges", {}))
        if derived:
            dsl["ports"] = derived
            warnings.append(
                f"auto-derived top-level ports from open endpoints: {list(derived)}"
            )
        else:
            warnings.append(
                "no top-level ports could be derived (no open endpoints)"
            )

    return dsl, warnings


def _enrich_circuit_dsl_from_legacy_pdk(
    circuit_dsl: dict,
    warnings: list[str],
) -> dict:
    """Replicate the legacy PhIDO pre-`dsl_to_gf` PDK pass (see `webapp` schematic
    step): :func:`PhotonicsAI.Photon.DemoPDK.get_ports_info` and per-node
    default **settings** the same way :func:`PhotonicsAI.Photon.DemoPDK.get_params`
    does—then **merge LLM/Designer ``params`` on top** so user intent (length, gap,
    …) still wins. Without this, a single LLM that omits or mis-types params
    never matches the old multi-step + `apply_settings` path.
    """
    out = copy.deepcopy(circuit_dsl)
    # Registers DemoPDK with gdsfactory; required for `gf.read.from_yaml` below.
    import PhotonicsAI.Photon.DemoPDK as demo_mod  # noqa: F401

    user_by_node: dict[str, dict[str, Any]] = {}
    for nid, n in (out.get("nodes") or {}).items():
        if isinstance(n, dict) and isinstance(n.get("params"), dict):
            user_by_node[nid] = copy.deepcopy(n["params"])
        else:
            user_by_node[nid] = {}

    try:
        out = demo_mod.get_ports_info(out)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pdk enrich: get_ports_info failed: {exc!s}")

    import gdsfactory as gf

    for key, value in (out.get("nodes") or {}).items():
        if not isinstance(value, dict):
            continue
        comp = value.get("component")
        if not isinstance(comp, str):
            continue
        gf_netlist = {"instances": {key: {"component": comp}}}
        try:
            c = gf.read.from_yaml(yaml.dump(gf_netlist, default_flow_style=False))
            updated = c.get_netlist(recursive=False)
            value["params"] = dict(
                (updated.get("instances") or {})
                .get(key, {})
                .get("settings", {})
            )
        except Exception as exc:  # noqa: BLE001
            value.setdefault("params", {})
            warnings.append(
                f"pdk enrich: default params for {key!r} ({comp}) failed: {exc!s}"
            )
        user = user_by_node.get(key) or {}
        if user:
            base = value.get("params")
            if not isinstance(base, dict):
                base = {}
            value["params"] = {**base, **user}
    return out


def dsl_to_netlist(circuit_dsl: dict) -> tuple[dict, list[str]]:
    """Convert a circuit DSL dict into a GDSFactory netlist dict.

    Wraps :func:`PhotonicsAI.Photon.utils.dsl_to_gf` so the rest of the
    graph package does not import the legacy module directly. We make a
    deep copy because ``dsl_to_gf`` mutates its input in places.

    Returns ``(gf_netlist, warnings)``; warnings are non-fatal hints
    surfaced from the normalizer (e.g. "auto-derived ports from open
    endpoints") so the Reflector can see what we silently patched.
    """
    from PhotonicsAI.Photon import utils as photon_utils

    warnings: list[str] = []
    validated = validate_dsl_shape(copy.deepcopy(circuit_dsl))
    validated = _enrich_circuit_dsl_from_legacy_pdk(validated, warnings)
    normalized, norm_warnings = _normalize_designer_dsl(validated)
    warnings.extend(norm_warnings)
    try:
        netlist = photon_utils.dsl_to_gf(normalized)
    except KeyError as exc:
        # Surface a precise shape error the Reflector can act on rather
        # than the raw KeyError from the legacy mutator.
        raise NetlistError(
            f"dsl_to_gf needs key {exc!s} that is still missing after "
            "normalization; upstream DSL is too malformed.",
            stage="dsl_to_gf",
        ) from exc
    return netlist, warnings


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
    for iid, inst in (data.get("instances") or {}).items():
        if isinstance(inst, dict) and isinstance(inst.get("settings"), dict):
            _coerce_mapping_numbers(
                inst["settings"],
                label=f"netlist.instance[{iid}].settings",
                warnings=None,
            )
    for pid, pl in (data.get("placements") or {}).items():
        if isinstance(pl, dict):
            _coerce_mapping_numbers(
                pl,
                label=f"netlist.placement[{pid}]",
                warnings=None,
            )
    return data


# ``DemoPDK`` registers cells in ``pdk.cells`` on first import. Long-lived
# processes (Streamlit) keep the old ``@gf.cell`` callables in memory; editing
# ``DesignLibrary/*.py`` on disk does not update them unless the module is
# reloaded.  We re-sync by ``mtime`` so hot edits (e.g. new kwargs on a cell)
# apply without a full process restart.
_demo_pdk_cell_mtimes: dict[str, float] = {}


def _refresh_demo_pdk_cells_from_disk() -> None:
    """``importlib.reload`` any DesignLibrary module whose ``.py`` file changed, then
    point ``Pdk.cells[name]`` at the new :func:``@gf.cell`` function.
    """
    try:
        import gdsfactory as gf
    except Exception:  # noqa: BLE001
        return
    pdk = gf.get_active_pdk()
    if pdk is None or getattr(pdk, "name", None) != "DemoPDK":
        return
    from PhotonicsAI.config import PATH
    from PhotonicsAI.Photon.DemoPDK import list_python_files

    lib = Path(PATH.pdk)
    for stem in list_python_files(PATH.pdk):
        fpath = lib / f"{stem}.py"
        if not fpath.is_file():
            continue
        mtime = fpath.stat().st_mtime
        if _demo_pdk_cell_mtimes.get(stem) == mtime:
            continue
        _demo_pdk_cell_mtimes[stem] = mtime
        mod_name = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{stem}"
        try:
            mod = importlib.import_module(mod_name)
            mod = importlib.reload(mod)
        except Exception:  # noqa: BLE001
            continue
        if stem not in pdk.cells:
            continue
        try:
            cell_fn = getattr(mod, stem)
        except Exception:  # noqa: BLE001
            continue
        pdk.cells[stem] = cell_fn


def _ensure_demo_pdk_active() -> None:
    """Ensure the legacy ``DemoPDK`` is the active gdsfactory PDK.

    Importing :mod:`PhotonicsAI.Photon.DemoPDK` has the side-effect of
    calling ``DemoPDK.activate()``; subsequent ``gf.read.from_yaml``
    calls then validate ``instance.component`` against DemoPDK's cells
    (``_directional_coupler``, ``mzi_1x2_pindiode_cband``, ...)
    rather than the built-in ``generic`` PDK.

    We also re-load DesignLibrary source files that changed on disk so the
    active PDK's ``cells`` map tracks edited kwargs (e.g. ``gap`` on
    ``_directional_coupler``) without restarting Streamlit.
    """
    # Importing the module runs ``DemoPDK.activate()`` at module level.
    import PhotonicsAI.Photon.DemoPDK  # noqa: F401
    _refresh_demo_pdk_cells_from_disk()


def _scale_numeric_placements(data: dict, factor: float) -> dict:
    """Deep-copy *data* and scale numeric ``x``/``y`` in ``placements`` (µm).

    Graphviz/legacy placements are only a *hint*; gdsfactory's optical router
    may still fail (e.g. *same angle* at a merge) when the schematic is
    correct for SAX. Spacing instances apart often unblocks the router
    before we drop ``routes`` entirely.
    """
    out = copy.deepcopy(data)
    for pl in (out.get("placements") or {}).values():
        if not isinstance(pl, dict):
            continue
        for k in ("x", "y"):
            v = pl.get(k)
            if isinstance(v, (int, float)):
                pl[k] = float(v) * factor
    return out


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

    _ensure_demo_pdk_active()
    import gdsfactory as gf

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
        for factor in (1.5, 2.0, 2.5, 3.0, 3.5):
            try:
                scaled = _scale_numeric_placements(data, factor)
                yaml_s = yaml.dump(
                    scaled, default_flow_style=False, sort_keys=False
                )
                component = gf.read.from_yaml(yaml_s)
                return component, GdsReport(
                    ok=True,
                    warnings=[
                        f"gdsfactory: recovered router after "
                        f"scaling placements by {factor}× (legacy Graphviz hint)"
                    ],
                )
            except Exception:  # noqa: BLE001
                continue
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


def save_layout_png(
    component, png_path: str | Path, *, show_labels: bool = True
) -> tuple[bool, list[str]]:
    """Render ``component`` to a PNG via ``gf.Component.plot``.

    Returns ``(success, errors)``. Uses the Agg backend to remain safe
    inside Streamlit / background threads (no GUI required).
    """
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        errors.append(f"matplotlib unavailable: {exc}")
        return False, errors

    try:
        fig = component.plot(return_fig=True, show_labels=show_labels)
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True, errors
    except Exception as exc:  # noqa: BLE001
        errors.append(f"plot/savefig failed: {exc}")
        return False, errors


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

    Any shape / translation error is absorbed into the report rather
    than raised, so the Reflector loop always receives structured
    feedback it can reason about.
    """
    try:
        netlist, normalize_warnings = dsl_to_netlist(circuit_dsl)
    except NetlistError as exc:
        return None, GdsReport(ok=False, errors=[f"dsl->netlist: {exc}"])
    except Exception as exc:  # noqa: BLE001
        return None, GdsReport(
            ok=False,
            errors=[f"dsl->netlist crashed unexpectedly: {type(exc).__name__}: {exc}"],
        )

    component, report = build_component(netlist, ignore_links=False)
    if normalize_warnings:
        report.warnings.extend(f"normalizer: {w}" for w in normalize_warnings)
    if component is None:
        return None, report

    success, write_errors = write_gds(component, gds_path)
    report.errors.extend(write_errors)
    if success:
        report.gds_path = str(gds_path)
    else:
        report.ok = False

    png_path = Path(gds_path).with_suffix(".png")
    png_ok, png_errors = save_layout_png(component, png_path)
    if png_ok:
        report.gds_png_path = str(png_path)
    else:
        report.errors.extend(png_errors)

    return component, report
