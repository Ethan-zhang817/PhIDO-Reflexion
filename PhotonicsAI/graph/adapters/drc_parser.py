"""Parser for KLayout XML report databases (``.lyrdb``).

KLayout writes report databases in an XML dialect. Each violation is an
``<item>`` linking back to a ``<category>`` (the rule name) and one or
more ``<values>`` describing the geometry. We do not need full fidelity
— the Reflector agent only needs:

* a histogram of violations by rule, and
* a small number of representative coordinates per rule.

This module parses the XML defensively: if KLayout changes the format
slightly we degrade to "n_violations unknown but not zero" rather than
crashing the agent loop.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PhotonicsAI.graph.adapters.eda_report import DRCExample, DRCReport


_COORD_RE = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)")
_MIN_RE = re.compile(r"min\s*[:=]?\s*([-\d.eE+]+)")


def _strip_quotes(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip().strip("'\"")


def _first_coord(value_text: str) -> list[float]:
    """Return the first ``(x, y)`` pair found in a KLayout value string.

    KLayout encodes geometries as ``edge: (x1,y1;x2,y2)``,
    ``box: (x1,y1;x2,y2)``, ``polygon: (x1,y1;x2,y2;...)``, etc. The first
    coordinate is enough as a "pointer" for the Reflector's hint.
    """
    m = _COORD_RE.search(value_text)
    if not m:
        return []
    try:
        return [float(m.group(1)), float(m.group(2))]
    except ValueError:
        return []


def _parse_min_from_description(description: str) -> float | None:
    m = _MIN_RE.search(description or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _iter_items(root: ET.Element) -> Iterable[ET.Element]:
    items_node = root.find("items")
    if items_node is None:
        return []
    return list(items_node.findall("item"))


def parse_lyrdb_xml(
    report_path: str | Path,
    *,
    top_k: int = 3,
) -> DRCReport:
    """Parse a KLayout XML report database into a :class:`DRCReport`.

    The function is tolerant of empty / malformed reports: it will return
    a ``DRCReport(ok=True, n_violations=0)`` for an empty file rather
    than raising.
    """
    p = Path(report_path)
    if not p.exists():
        return DRCReport(ok=True, n_violations=0, raw_report_path=str(p))

    raw = p.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return DRCReport(ok=True, n_violations=0, raw_report_path=str(p))

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return DRCReport(
            ok=False,
            n_violations=0,
            by_rule={"_parse_error": 1},
            raw_report_path=str(p),
            skipped_reason="lyrdb parse error",
        )

    category_descriptions: dict[str, str] = {}
    cats_node = root.find("categories")
    if cats_node is not None:
        for cat in cats_node.findall("category"):
            name = (cat.findtext("name") or "").strip()
            description = (cat.findtext("description") or "").strip()
            if name:
                category_descriptions[name] = description

    by_rule: dict[str, int] = defaultdict(int)
    examples_per_rule: dict[str, list[DRCExample]] = defaultdict(list)

    for item in _iter_items(root):
        rule = _strip_quotes(item.findtext("category"))
        cell = _strip_quotes(item.findtext("cell"))
        if not rule:
            rule = "_unknown"

        by_rule[rule] += 1
        if len(examples_per_rule[rule]) >= top_k:
            continue

        coord: list[float] = []
        values_node = item.find("values")
        if values_node is not None:
            for value in values_node.findall("value"):
                if value.text:
                    coord = _first_coord(value.text)
                    if coord:
                        break

        description = category_descriptions.get(rule, "")
        examples_per_rule[rule].append(
            DRCExample(
                rule=rule,
                cell=cell,
                coord=coord,
                description=description,
                min=_parse_min_from_description(description),
            )
        )

    n_violations = sum(by_rule.values())
    examples: list[DRCExample] = []
    for rule_examples in examples_per_rule.values():
        examples.extend(rule_examples)

    return DRCReport(
        ok=(n_violations == 0),
        n_violations=n_violations,
        by_rule=dict(by_rule),
        examples=examples,
        raw_report_path=str(p),
    )
