# SPDX-License-Identifier: Apache-2.0
"""Tests for Track 4 small-row (4..8) 5-bit and 6-bit NAX skinny verify routing."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.custom_kernels import nax_skinny_qmm as nsq
from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import batched_decode_qmm as bdq
from omlx.patches import qwen35_verify_qmm as vq

needs_nax = pytest.mark.skipif(not is_nax_available(), reason="needs an M5 (NAX) GPU")


def _weights(K: int, N: int, bits: int, dtype, seed: int):
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.02).astype(dtype)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    return wq, scales.astype(dtype), biases.astype(dtype)


def _max_rel(y: mx.array, ref: mx.array) -> float:
    return (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()


def test_small_skinny_routing_boundaries(monkeypatch):
    monkeypatch.setattr(vq, "is_nax_available", lambda: True)
    bf16 = mx.bfloat16

    # 5-bit takes skinny for M=4..8 when N >= 4096
    for M in (4, 5, 6, 7, 8):
        assert vq._verify_route(M, 5120, 6144, 5, 64, bf16) == "skinny", M
        assert vq._verify_route(M, 17408, 5120, 5, 64, bf16) == "skinny", M
    # Below 4 rows or N < 4096 stays off small skinny
    assert vq._verify_route(3, 5120, 6144, 5, 64, bf16) is None
    assert vq._verify_route(5, 5120, 2048, 5, 64, bf16) != "skinny"

    # 6-bit takes skinny for M=6..8 (replacing sg8; M=4..5 keep stock) when N >= 4096
    assert vq._verify_route(4, 5120, 248320, 6, 64, bf16) is None
    assert vq._verify_route(5, 5120, 248320, 6, 64, bf16) is None
    for M in (6, 7, 8):
        assert vq._verify_route(M, 5120, 248320, 6, 64, bf16) == "skinny", M
        assert vq._verify_route(M, 5120, 6144, 6, 64, bf16) == "skinny", M

    # 4-bit and 8-bit remain off small skinny at M=4..8
    assert vq._verify_route(4, 5120, 17408, 4, 64, bf16) is None
    assert vq._verify_route(6, 5120, 17408, 4, 64, bf16) == "sg8"
    assert vq._verify_route(8, 6144, 5120, 8, 64, bf16) != "skinny"

    # Batched decode route preserves sg8/mma fallback and does not return None
    bdq._route.cache_clear()
    assert bdq._route(6, 5120, 6144, 5, 64, bf16) == "sg8"
    assert bdq._route(8, 5120, 6144, 6, 64, bf16) == "sg8"
    assert bdq._route(12, 5120, 6144, 5, 64, bf16) == "mma"

    # Non-NAX falls back cleanly
    monkeypatch.setattr(vq, "is_nax_available", lambda: False)
    assert not vq._small_skinny_eligible(5, 5120, 6144, 5, 64, bf16)
    assert not vq._small_skinny_eligible(6, 5120, 248320, 6, 64, bf16)


@needs_nax
@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("M", [4, 5, 6, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_small_row_skinny_matches_fp32_reference(bits, M, dtype):
    K, N = 1024, 4096
    wq, sc, bi = _weights(K, N, bits, dtype, seed=bits * 31 + M)
    x = (mx.random.normal((M, K), key=mx.random.key(M + 11)) * 0.1).astype(dtype)
    truth = x.astype(mx.float32) @ (
        mx.dequantize(wq, sc, bi, group_size=64, bits=bits).astype(mx.float32).T
    )
    stock = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=bits)
    got = nsq.skinny_qmm(x, wq, sc, bi, K, N, bits=bits)
    assert got.shape == (M, N) and got.dtype == dtype
    assert _max_rel(got, truth) <= 1.25 * _max_rel(stock, truth) + 1e-4


@needs_nax
def test_armed_quantized_linear_routes_5bit_verify_to_skinny(monkeypatch):
    assert vq.apply_verify_qmm_patch()
    K, N = 1024, 4096
    linear = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=5)
    linear.weight, linear.scales, linear.biases = _weights(K, N, 5, mx.bfloat16, seed=9)
    x = (mx.random.normal((1, 5, K), key=mx.random.key(19)) * 0.1).astype(mx.bfloat16)
    stock = linear(x)

    calls = []
    real = nsq.skinny_qmm

    def spy(x2, *args, **kwargs):
        calls.append(int(x2.shape[0]))
        return real(x2, *args, **kwargs)

    monkeypatch.setattr(nsq, "skinny_qmm", spy)
    vq.set_verify_qmm_armed(True)
    try:
        routed = linear(x)
    finally:
        vq.set_verify_qmm_armed(False)
    assert calls == [5]
    assert mx.allclose(routed, stock, atol=2e-2, rtol=2e-2).item()
