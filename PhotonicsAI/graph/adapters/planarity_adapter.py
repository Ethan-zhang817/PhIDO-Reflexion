"""Planarity check adapter that returns the actual crossing edge pairs.

The legacy :func:`PhotonicsAI.Photon.utils.dot_planarity` only returns a
boolean. For the Reflector to make a useful suggestion ("move N3 up by
~20 um") it needs to know *which* edges cross. This adapter recomputes
the geometry once via Graphviz and exposes the crossings as a list.
"""

from __future__ import annotations

from typing import NamedTuple

import pygraphviz as pgv


class CrossingEdgePair(NamedTuple):
    edge1: tuple[str, str]  # ("C1", "C2") -- only node-level for now
    edge2: tuple[str, str]
    edge1_coords: tuple[tuple[float, float], tuple[float, float]]
    edge2_coords: tuple[tuple[float, float], tuple[float, float]]


def _do_intersect(p1, q1, p2, q2) -> bool:
    """Standard 2D segment intersection (mirrors utils.dot_planarity)."""

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def find_crossings(dot_string: str) -> list[CrossingEdgePair]:
    """Return the list of crossing edge pairs in a Graphviz DOT graph.

    The list is empty when the graph is planar.
    """
    lines = dot_string.strip().splitlines()
    if lines and lines[0].strip() == "dot":
        dot_string = "\n".join(lines[1:])

    graph = pgv.AGraph(string=dot_string)
    graph.layout(prog="dot")

    edges: list[tuple[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]] = []
    for edge in graph.edges():
        pos = edge.attr["pos"]
        if not pos:
            continue
        points = pos.split()
        try:
            start = tuple(map(float, points[0].split(",")))
            end = tuple(map(float, points[-1].split(",")))
        except ValueError:
            continue
        edges.append(((str(edge[0]), str(edge[1])), (start, end)))

    crossings: list[CrossingEdgePair] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            (e1_nodes, (p1, q1)) = edges[i]
            (e2_nodes, (p2, q2)) = edges[j]
            if _do_intersect(p1, q1, p2, q2):
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                crossings.append(
                    CrossingEdgePair(
                        edge1=e1_nodes,
                        edge2=e2_nodes,
                        edge1_coords=(p1, q1),
                        edge2_coords=(p2, q2),
                    )
                )

    return crossings


def is_planar(dot_string: str) -> bool:
    """Convenience boolean wrapper, matching legacy :func:`dot_planarity`."""
    return len(find_crossings(dot_string)) == 0
