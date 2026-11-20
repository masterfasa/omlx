# SPDX-License-Identifier: Apache-2.0
"""The fused verifier norm must use the served RMS/FP32 SiLU arithmetic."""

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen3_5.language import Qwen3_5RMSNormGated

from omlx.patches import qwen35_gdn_verify_fused as patch
from omlx.patches.qwen35_gdn_verify_fused import _norm_gate_kernel

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _fused(norm, x, gate):
    rows = x.size // 128
    return _norm_gate_kernel(norm.eps)(
        inputs=[x, gate, norm.weight],
        template=[("InT", x.dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape, (rows * 2,)],
        output_dtypes=[x.dtype, mx.float32],
    )[0]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4])
def test_fused_norm_gate_matches_served_arithmetic(dtype, eps):
    norm = Qwen3_5RMSNormGated(128, eps=eps)
    for seed in range(40):
        mx.random.seed(seed)
        norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
        x = (mx.random.normal((1, 1, 32, 128)) * 10 ** (seed % 8 - 4)).astype(dtype)
        gate = (mx.random.normal(x.shape) * (seed / 2 + 0.125)).astype(dtype)
        expected, actual = norm(x, gate), _fused(norm, x, gate)
        mx.eval(expected, actual)
        assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_norm_gate_matches_every_gate_encoding(dtype):
    mx.random.seed(42)
    norm = Qwen3_5RMSNormGated(128)
    norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
    x = mx.ones((1, 1, 512, 128), dtype)
    gate = (
        mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(dtype).reshape(x.shape)
    )
    expected, actual = norm(x, gate), _fused(norm, x, gate)
    mx.eval(expected, actual)
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


def test_unsupported_sigmoid_declines_fusion(monkeypatch):
    from types import SimpleNamespace

    class Cache:
        _speculation = {"length": 2, "records": {}}

        def __getitem__(self, index):
            return None

    norm = Qwen3_5RMSNormGated(128)
    norm.weight = mx.ones((128,), mx.float16)
    layer = SimpleNamespace(
        norm=norm,
        head_k_dim=128,
        head_v_dim=128,
        num_v_heads=8,
        num_k_heads=4,
        A_log=mx.ones((8,), mx.float16),
        dt_bias=mx.ones((8,), mx.float16),
    )
    monkeypatch.setattr(patch, "_PATCHED", True)
    monkeypatch.setattr(patch, "_NORM_CLASS", Qwen3_5RMSNormGated)
    monkeypatch.setattr(patch, "_sigmoid_exp", lambda: None)
    assert not patch.fused_eligible(layer, mx.ones((1,), mx.float16), Cache(), 2)


def test_probe_declines_unknown_served_arithmetic(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    monkeypatch.setattr(
        language, "_precise_swiglu", lambda h, gate, x: mx.ones_like(gate)
    )
    patch._sigmoid_exp.cache_clear()
    try:
        assert patch._sigmoid_exp() is None
    finally:
        patch._sigmoid_exp.cache_clear()


@pytest.mark.parametrize("keep", [0, 2, 4])
def test_commit_speculation_all_custom_fast_path_and_mixed_parity(keep):
    """Fast path (all custom records) and mixed path must restore identical state & metadata."""
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    assert patch.apply_arrays_cache_replay_patch()
    T = 4
    prev = mx.arange(3 * 4, dtype=mx.float32).reshape(1, 3, 4)
    rows = (mx.arange(T * 4, dtype=mx.float32) + 100).reshape(1, T, 4)
    start_s = mx.zeros((1, 2, 2, 2), dtype=mx.float32)
    mid_s = mx.ones((1, 2, 2, 2), dtype=mx.float32) * 2.0
    final_s = mx.ones((1, 2, 2, 2), dtype=mx.float32) * 4.0

    # All-custom transaction (window_pair + deferred)
    c_fast = ArraysCache(size=2, left_padding=[2])
    c_fast[0] = prev
    c_fast[1] = start_s
    tx_fast = start_speculative_cache([c_fast], T)
    patch.record_window(c_fast, 0, prev, rows, 3)
    patch.record_deferred_states(c_fast, 1, start_s, final_s, T, lambda _k: mid_s)
    c_fast[0] = rows[:, -3:, :]
    c_fast[1] = final_s
    tx_fast.commit([keep])

    assert c_fast._speculation is None
    expected_conv = (
        prev
        if keep == 0
        else (rows[:, keep - 3 : keep] if keep >= 3 else mx.concatenate([prev[:, keep:], rows[:, :keep]], axis=1))
    )
    expected_rec = start_s if keep == 0 else (final_s if keep == T else mid_s)
    assert mx.array_equal(c_fast[0], expected_conv).item()
    assert mx.array_equal(c_fast[1], expected_rec).item()
    assert c_fast._left_padding_advance == keep

    # Mixed transaction (custom slot 0 + stock states slot 1)
    c_mix = ArraysCache(size=2, left_padding=[2])
    c_mix[0] = prev
    c_mix[1] = start_s
    tx_mix = start_speculative_cache([c_mix], T)
    patch.record_window(c_mix, 0, prev, rows, 3)
    per_tok = [start_s + (i + 1) for i in range(T)]
    c_mix.record_speculative_states(1, mx.stack(per_tok[:-1], axis=1), per_tok[-1])
    c_mix[0] = rows[:, -3:, :]
    c_mix[1] = per_tok[-1]
    tx_mix.commit([keep])

    assert c_mix._speculation is None
    assert mx.array_equal(c_mix[0], expected_conv).item()
    expected_mix_1 = start_s if keep == 0 else per_tok[keep - 1]
    assert mx.array_equal(c_mix[1], expected_mix_1).item()
    assert c_mix._left_padding_advance == keep


def test_keep_array_caches_small_tuples():
    a = patch._keep_array([3])
    b = patch._keep_array([3])
    assert a is b
    assert a.dtype == mx.int32
    assert a.tolist() == [3]


@pytest.mark.parametrize(
    "q_dtype,a_log_dtype,want",
    [
        (mx.bfloat16, mx.float32, True),
        (mx.bfloat16, mx.bfloat16, True),
        (mx.bfloat16, mx.float16, False),
        (mx.float16, mx.float32, True),
        (mx.float16, mx.bfloat16, False),
    ],
)
def test_replay_and_fused_eligibility_allows_fp32_a_log(q_dtype, a_log_dtype, want):
    from types import SimpleNamespace

    class Cache:
        _speculation = {"length": 4, "records": {}}

        def __getitem__(self, index):
            return None

    norm = Qwen3_5RMSNormGated(128)
    norm.weight = mx.ones((128,), q_dtype)
    layer = SimpleNamespace(
        norm=norm,
        head_k_dim=128,
        head_v_dim=128,
        num_v_heads=8,
        num_k_heads=4,
        A_log=mx.ones((8,), a_log_dtype),
        dt_bias=mx.ones((8,), q_dtype),
    )
    assert patch.apply_arrays_cache_replay_patch()
    q = mx.zeros((1, 4, 4, 128), q_dtype)
    assert patch.replay_eligible(layer, q, Cache(), 4) is want
    assert patch.fused_eligible(layer, q, Cache(), 4) is want
