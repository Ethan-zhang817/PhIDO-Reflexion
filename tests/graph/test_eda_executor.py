"""Tests for EDA executor helpers (no graph / no KLayout run)."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import ormsgpack

from PhotonicsAI.graph.nodes.eda_executor import _sax_result_to_jsonable


def test_sax_result_to_jsonable_converts_jax_and_tuple_keys():
    raw = {("o1", "o2"): jnp.ones(2, dtype=jnp.float32), "k": np.array([3, 4])}
    out = _sax_result_to_jsonable(raw)
    json.dumps(out)
    ormsgpack.packb(out)
    assert out["o1,o2"] == [1.0, 1.0]
    assert out["k"] == [3, 4]


def test_sax_result_to_jsonable_jax_array_impl_and_complex():
    s = (1.0 + 0.1j) * jnp.linspace(0, 1, 4)
    raw = {("o1", "o2"): s}
    out = _sax_result_to_jsonable(raw)
    assert out is not None
    ormsgpack.packb(out)
    v = out["o1,o2"]
    assert all(isinstance(x, dict) and "re" in x and "im" in x for x in v)


def test_sax_result_to_jsonable_none():
    assert _sax_result_to_jsonable(None) is None
