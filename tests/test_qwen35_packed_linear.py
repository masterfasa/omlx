# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed 4-bit projections' skinny (9..48 row) kernel."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import qwen35_packed_linear as pl

pytestmark = pytest.mark.skipif(not is_nax_available(), reason="needs an M5 (NAX) GPU")


def _linear(K: int, N: int, seed: int) -> nn.QuantizedLinear:
    layer = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    layer.weight, scales, biases = mx.quantize(w, group_size=64, bits=4)
    layer.scales = scales.astype(mx.bfloat16)
    layer.biases = biases.astype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer


def _rows(M: int, K: int, seed: int) -> mx.array:
    return (mx.random.normal((M, K), key=mx.random.key(seed)) * 0.1).astype(mx.bfloat16)


def _max_rel(y: mx.array, ref: mx.array) -> float:
    return (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()


# N = 384 splits K across 8 simdgroups on every M5 GPU, N = 25600 across 2.
@pytest.mark.parametrize("K,N", [(1024, 384), (512, 25600)])
def test_skinny_matches_reference_at_every_row_count(K, N):
    layer = _linear(K, N, seed=K + N)
    w, s, b = layer.weight, layer.scales, layer.biases
    packed = pl._pack([layer])[0]
    x = _rows(pl._SKINNY_MAX_ROWS, K, seed=4)
    truth = (
        x.astype(mx.float32)
        @ mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32).T
    )
    stock = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    full = packed(x)
    assert full.shape == (pl._SKINNY_MAX_ROWS, N) and full.dtype == mx.bfloat16
    assert _max_rel(full, truth) <= 1.25 * _max_rel(stock, truth) + 1e-4
    # Rows keep their bits at every row count, across 16-row fragment edges.
    for M in (9, 16, 17, 33):
        assert mx.array_equal(packed(x[:M]), full[:M]).item(), M


def test_dispatch_keeps_small_and_wide_calls_off_the_skinny_kernel(monkeypatch):
    calls = []
    original = pl._skinny

    def spy(x2, *args):
        calls.append(int(x2.shape[0]))
        return original(x2, *args)

    monkeypatch.setattr(pl, "_skinny", spy)
    K, N = 512, 25600
    packed = pl._pack([_linear(K, N, seed=5)])[0]
    for M in (8, 9, 48, 49):
        mx.eval(packed(_rows(M, K, seed=M)))
    assert calls == [9, 48]


def test_core_count_aware_small_geometry_and_matvec_ksplit():
    # 80-core M5 Ultra keeps upstream _TARGET_TGS = 1400 behavior verbatim.
    for M in (2, 4, 5, 6, 8):
        assert pl._small_geometry(5120, 34816, M, cores=80) == (5, 1)
        assert pl._small_geometry(5120, 10240, M, cores=80) == (10, 2)
        assert pl._small_geometry(5120, 14336, M, cores=80) == (10, 2)

    # 20-core M5 Pro keeps upstream splits at M=2..4, eliminates _reduce
    # (ks=1, parts=2) on gdn_qkv/attn_qkv from M=5, and caps mlp_gate+up at
    # (2, 1) from M=6.
    for M in (2, 4):
        assert pl._small_geometry(5120, 34816, M, cores=20) == (5, 1)
        assert pl._small_geometry(5120, 10240, M, cores=20) == (10, 2)
        assert pl._small_geometry(5120, 14336, M, cores=20) == (10, 2)
    assert pl._small_geometry(5120, 34816, 5, cores=20) == (5, 1)
    assert pl._small_geometry(5120, 10240, 5, cores=20) == (1, 2)
    assert pl._small_geometry(5120, 14336, 5, cores=20) == (1, 2)
    for M in (6, 8):
        assert pl._small_geometry(5120, 34816, M, cores=20) == (2, 1)
        assert pl._small_geometry(5120, 10240, M, cores=20) == (1, 2)
        assert pl._small_geometry(5120, 14336, M, cores=20) == (1, 2)

    # 12-core M6 and 32-core M5 Max stay bounded without over-splitting.
    assert pl._small_geometry(5120, 34816, 6, cores=12) == (1, 1)
    assert pl._small_geometry(5120, 10240, 6, cores=12) == (1, 2)
    assert pl._small_geometry(5120, 10240, 6, cores=32) == (1, 2)
    assert pl._small_geometry(5120, 34816, 6, cores=32) == (2, 1)


@pytest.mark.parametrize("M", [1, 2, 4, 5, 6, 8])
@pytest.mark.parametrize("K,N", [(1024, 1024), (2048, 2560)])
def test_packed_small_and_matvec_match_reference(M, K, N):
    layer = _linear(K, N, seed=M * 17 + N)
    w, s, b = layer.weight, layer.scales, layer.biases
    packed = pl._pack([layer])[0]
    x = _rows(M, K, seed=M + 3)
    truth = (
        x.astype(mx.float32)
        @ mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32).T
    )
    stock = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    got = packed(x)
    assert got.shape == (M, N) and got.dtype == mx.bfloat16
    assert _max_rel(got, truth) <= 1.25 * _max_rel(stock, truth) + 1e-4
