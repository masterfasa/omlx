# SPDX-License-Identifier: Apache-2.0
"""Skinny tensor-unit (NAX) matmul for verify-width rows against affine weights.

``y[M, N] = x[M, K] @ dequant(W[N, K])^T`` for M in 9..48, group size 64, on M5 GPUs.
A threadgroup owns 32 output columns and all rows, so each weight is read and
dequantized once per call. Its S simdgroups split K by quant group and reduce in
threadgroup memory; each issues 16x32x16 ``matmul2d`` ops with x on the left and the
dequantized weight rows as the transposed right operand.

Two weight layouts:

- MLX's (``packed=False``): 4-, 5-, 6- and 8-bit rows, 8 codes in ``bits`` bytes.
- ``qwen35_packed_linear``'s tile layout (``packed=True``, 4-bit): codes
  [tile][group][column][32 B], scales/biases [tile][group / 4][column][4].

Inside a 64-wide group, K is permuted (identically for x and w) so a lane's 16 codes of a
weight row are one contiguous ``2 * bits``-byte load. Dequantization matches stock MLX:
fp32 ``scale * q + bias``, rounded once to the activation dtype; float32 accumulation.
The K split scales with GPU core count, row-fragment count (``ceil(M / 16)``) and
bit-width, so a row gets the same bits across every row count sharing the same
split.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx

from ..utils.hardware import get_gpu_core_count
from .nax_tiles import NAX_TILE_HEADER

logger = logging.getLogger(__name__)

_SOURCE = """
    using namespace omlx_nax;
    using Frag = BaseNAXFrag;
    constexpr int GROUPS = K / 64;
    constexpr int WORDS = (2 * BITS + 3) / 4;   // uint32 words per lane per group
    constexpr uint MASK = (1u << BITS) - 1u;

    const int M = x_shape[0];
    const int sg = int(simdgroup_index_in_threadgroup);
    const uint lane = thread_index_in_simdgroup;
    const short2 coord = Frag::get_coord();
    const int fm = coord.y;
    const int fn = coord.x;
    const int q4 = fn >> 2;
    const int n0 = int(threadgroup_position_in_grid.x) * 32;

    // The lane's weight rows: columns n0 + 16 f + fm + 8 i.
    const device uchar* wrow[2][2];
    const device T* srow[2][2];
    const device T* brow[2][2];
    OMLX_NAX_UNROLL
    for (int f = 0; f < 2; ++f) {
        OMLX_NAX_UNROLL
        for (int i = 0; i < 2; ++i) {
            const int n = n0 + 16 * f + fm + 8 * i;
            if (PACKED) {
                const ulong tile = ulong(n / TILE_N);
                const int col = n % TILE_N;
                wrow[f][i] = (const device uchar*)w + (tile * GROUPS * TILE_N + col) * 32 + 8 * q4;
                srow[f][i] = sc + tile * GROUPS * TILE_N + col * 4;
                brow[f][i] = bi + tile * GROUPS * TILE_N + col * 4;
            } else {
                wrow[f][i] = (const device uchar*)w + ulong(n) * (K / 8 * BITS) + 2 * BITS * q4;
                srow[f][i] = sc + ulong(n) * GROUPS;
                brow[f][i] = bi + ulong(n) * GROUPS;
            }
        }
    }
    const device T* xrow[RF][2];
    OMLX_NAX_UNROLL
    for (int t = 0; t < RF; ++t) {
        OMLX_NAX_UNROLL
        for (int i = 0; i < 2; ++i) {
            xrow[t][i] = x + ulong(min(16 * t + fm + 8 * i, M - 1)) * K + 16 * q4;
        }
    }

    Frag::dtype_frag_t<float> acc[RF][2];
    OMLX_NAX_UNROLL
    for (int t = 0; t < RF; ++t) {
        acc[t][0] = Frag::dtype_frag_t<float>(0);
        acc[t][1] = Frag::dtype_frag_t<float>(0);
    }

    for (int g = sg; g < GROUPS; g += S) {
        uint wv[2][2][WORDS];
        float s[2][2];
        float b[2][2];
        OMLX_NAX_UNROLL
        for (int f = 0; f < 2; ++f) {
            OMLX_NAX_UNROLL
            for (int i = 0; i < 2; ++i) {
                const device uchar* p = wrow[f][i] + (PACKED ? g * TILE_N * 32 : g * 8 * BITS);
                if (BITS == 4) {
                    const uint2 v = *(const device uint2*)p;
                    wv[f][i][0] = v.x;
                    wv[f][i][1] = v.y;
                } else if (BITS == 8) {
                    const uint4 v = *(const device uint4*)p;
                    wv[f][i][0] = v.x;
                    wv[f][i][1] = v.y;
                    wv[f][i][2] = v.z;
                    wv[f][i][3] = v.w;
                } else if (BITS == 6) {
                    OMLX_NAX_UNROLL
                    for (int h = 0; h < WORDS; ++h) wv[f][i][h] = ((const device uint*)p)[h];
                } else {
                    // 5-bit: 10 bytes at an even offset.
                    const device ushort* hp = (const device ushort*)p;
                    OMLX_NAX_UNROLL
                    for (int h = 0; h < WORDS; ++h) {
                        const uint lo = hp[2 * h];
                        const uint hi = (2 * h + 1 < BITS) ? uint(hp[2 * h + 1]) : 0u;
                        wv[f][i][h] = lo | (hi << 16);
                    }
                }
                const int at = PACKED ? (g >> 2) * TILE_N * 4 + (g & 3) : g;
                s[f][i] = float(srow[f][i][at]);
                b[f][i] = float(brow[f][i][at]);
            }
        }
        OMLX_NAX_UNROLL
        for (int j = 0; j < 4; ++j) {
            Frag::dtype_frag_t<T> A[RF];
            OMLX_NAX_UNROLL
            for (int t = 0; t < RF; ++t) {
                OMLX_NAX_UNROLL
                for (int i = 0; i < 2; ++i) {
                    const vec<T, 4> xv = *(const device vec<T, 4>*)(xrow[t][i] + 64 * g + 4 * j);
                    OMLX_NAX_UNROLL
                    for (int c = 0; c < 4; ++c) {
                        A[t][4 * i + c] = xv[c];
                    }
                }
            }
            Frag::dtype_frag_t<T> B[2];
            OMLX_NAX_UNROLL
            for (int f = 0; f < 2; ++f) {
                OMLX_NAX_UNROLL
                for (int i = 0; i < 2; ++i) {
                    OMLX_NAX_UNROLL
                    for (int c = 0; c < 4; ++c) {
                        // Code 4 j + c of the lane's 16 starts at bit BITS * (4 j + c).
                        const int bit = BITS * (4 * j + c);
                        const int word = bit >> 5;
                        const int sh = bit & 31;
                        uint v = wv[f][i][word] >> sh;
                        if (sh + BITS > 32) {
                            v |= wv[f][i][word + 1] << (32 - sh);
                        }
                        const float q = float(v & MASK);
                        B[f][4 * i + c] = static_cast<T>(fma(s[f][i], q, b[f][i]));
                    }
                }
            }
            OMLX_NAX_UNROLL
            for (int t = 0; t < RF; ++t) {
                Frag::mma(
                    acc[t][0], acc[t][1], A[t], metal::false_type{},
                    B[0], B[1], metal::true_type{});
            }
        }
    }

    // Sum the S partials one 16-row fragment at a time and store from simdgroup 0.
    threadgroup float red[(S > 1 ? S - 1 : 1) * 2 * 8 * 32];
    OMLX_NAX_UNROLL
    for (int t = 0; t < RF; ++t) {
        if (S > 1) {
            if (sg > 0) {
                OMLX_NAX_UNROLL
                for (int f = 0; f < 2; ++f) {
                    OMLX_NAX_UNROLL
                    for (int e = 0; e < 8; ++e) {
                        red[(((sg - 1) * 2 + f) * 8 + e) * 32 + lane] = acc[t][f][e];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                for (int o = 0; o < S - 1; ++o) {
                    OMLX_NAX_UNROLL
                    for (int f = 0; f < 2; ++f) {
                        OMLX_NAX_UNROLL
                        for (int e = 0; e < 8; ++e) {
                            acc[t][f][e] += red[((o * 2 + f) * 8 + e) * 32 + lane];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (sg == 0) {
            OMLX_NAX_UNROLL
            for (int i = 0; i < 2; ++i) {
                const int r = 16 * t + fm + 8 * i;
                if (r < M) {
                    device T* yp = y + ulong(r) * N + n0 + fn;
                    OMLX_NAX_UNROLL
                    for (int f = 0; f < 2; ++f) {
                        OMLX_NAX_UNROLL
                        for (int c = 0; c < 4; ++c) {
                            yp[16 * f + c] = static_cast<T>(acc[t][f][4 * i + c]);
                        }
                    }
                }
            }
        }
    }
"""

BITS = (4, 5, 6, 8)
MIN_ROWS = 9
MAX_ROWS = 48

# Simdgroups per GPU core the K split aims for on 32+ core GPUs and at 33..48
# rows (RF == 3), where register/barrier pressure dominates (see ``split``).
_SIMDGROUPS_PER_CORE = 16
# Below 32 cores (12-core M6, 20-core M5 Pro) and while register pressure is
# low (RF <= 2, i.e. <= 32 rows), each threadgroup needs a deeper K split to
# hide DRAM latency; budget ramps linearly from 32 cores down to 20 cores.
_LOW_CORE_REF = 32
_LOW_CORE_BUDGET = 2200
_WIDE_N = 65536


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_nax_skinny_qmm",
        input_names=["x", "w", "sc", "bi"],
        output_names=["y"],
        header=NAX_TILE_HEADER,
        source=_SOURCE,
    )


# Core count assumed when it cannot be read (M5 Max). It changes speed, not results.
FALLBACK_GPU_CORES = 32


@functools.cache
def gpu_cores() -> int:
    cores = get_gpu_core_count()
    if cores:
        return cores
    logger.warning(
        "GPU core count unavailable; sizing the skinny kernel's K split for %d cores",
        FALLBACK_GPU_CORES,
    )
    return FALLBACK_GPU_CORES


@functools.cache
def _split_for_cores(N: int, rf: int, bits: int, cores: int) -> int:
    base = _SIMDGROUPS_PER_CORE * cores
    max_s = 8
    budget = base
    if rf < 3 and cores < _LOW_CORE_REF:
        core_factor = min(1.0, (_LOW_CORE_REF - cores) / 12.0)
        bit_factor = 1.0 + max(0, bits - 4) * 0.25
        if rf == 1:
            boost = int(_LOW_CORE_BUDGET * core_factor * bit_factor)
            budget = max(base, boost)
        else:
            boost = int(_LOW_CORE_BUDGET * core_factor * min(bit_factor, 1.5))
            budget = max(base, boost)
            if bits < 8 and base < boost:
                s_base = 2
                while s_base < 8 and (N // 32) * s_base * 2 <= base:
                    s_base *= 2
                max_s = max(4, s_base)
    s = 2
    while s < max_s and (N // 32) * s * 2 <= budget:
        s *= 2
    return max(s, 4) if N >= _WIDE_N else s


def split(
    N: int,
    rows: int | None = None,
    *,
    bits: int = 4,
    cores: int | None = None,
) -> int:
    """Simdgroups splitting K: a power of two in ``{2, 4, 8}``.

    Uses ``_SIMDGROUPS_PER_CORE * cores`` on 32+ core GPUs and at 33..48 rows
    (``RF == 3``). On lower-core GPUs (12..20 cores) at 9..32 rows (``RF <= 2``),
    boosts the budget continuously with ``(32 - cores) / 12`` and bit-width so
    5120..17408-column projections take split 4 (or 8 at 8-bit / narrow 1-fragment
    shapes) and taper back to 2 at 33..48 rows. Vocabulary-size outputs
    (``N >= 65536``) stay at least 4.
    """
    c = gpu_cores() if cores is None else int(cores)
    rf = 3 if rows is None else max(1, -(-int(rows) // 16))
    return _split_for_cores(int(N), rf, int(bits), c)


def skinny_qmm(
    x2: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    K: int,
    N: int,
    *,
    bits: int,
    packed: bool = False,
    tile_n: int = 128,
) -> mx.array:
    """``x2`` (M, K) times affine group-64 weights (N, K) -> (M, N) in x2's dtype."""
    M = int(x2.shape[0])
    s = split(N, M, bits=bits)
    (y,) = _kernel()(
        inputs=[mx.contiguous(x2), w, scales, biases],
        template=[
            ("T", x2.dtype),
            ("K", K),
            ("N", N),
            ("BITS", bits),
            ("PACKED", int(packed)),
            ("TILE_N", tile_n),
            ("RF", -(-M // 16)),
            ("S", s),
        ],
        grid=(32 * s * (N // 32), 1, 1),
        threadgroup=(32 * s, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x2.dtype],
    )
    return y
