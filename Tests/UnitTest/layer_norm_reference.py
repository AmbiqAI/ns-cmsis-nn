# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#
"""Reference for arm_layer_norm_s8 / arm_layer_norm_s16 / arm_layer_norm_f32, and their export-time parameters.

The integer kernels are defined by this file: their scalar, DSP and MVE paths must match `layer_norm_int` byte for
byte. The definition uses only integer operations that map one to one onto MVE instructions (VMLAS, VQRDMULH,
VRSHL), so it does not depend on CMSIS_NN_USE_SINGLE_ROUNDING.

Layer norm over the innermost dimension of a [rows, N] tensor:

    y[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]

Integer form. The input scale and zero point cancel out of (x - mean) / std; only eps depends on them.

  Per row:
    S = sum(x)                  exact (int32)
    Q = sum(x * x)              exact (int64)
    D = N * Q - S * S + E       exact (int64): N^2 * (variance + eps), E = round(N^2 * eps / s_in^2) from export
    (m, L) = row_multiplier(D)  2^F / sqrt(D) ~= m * 2^L / 2^31, m in Q31, L >= 0 (integer Newton iterations)
  Per element:
    t   = (N << L) * x - (S << L)          int32 (wrapping arithmetic; the true value fits)
    z   = qrdmulh(t, m)                    (x - mean) / std with F fraction bits, |z| <= sqrt(N - 1) * 2^F
    acc = qrdmulh(z, G[c]) + B[c]
    y   = clamp(rshr(acc, T), act_min, act_max)

  where qrdmulh(a, b) = (a * b + 2^30) >> 31 (VQRDMULH) and rshr(v, T) = (v + 2^(T-1)) >> T (VRSHL by -T).
  G[c] = round(gamma[c] / s_out * 2^(31 - P)) shares one exponent P across the row, T = F - P, and
  B[c] = round(beta[c] / s_out * 2^T) + (zp_out << T) (`export_params`). Against float64 layer norm the result is
  within one output step, and exact for most values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FRAC_BITS = 20
NEWTON_STEPS = 3
# Largest supported row sizes: s8 keeps sum(x*x) in int32 (MVE VMLADAVA); s16 keeps N*x - S in int32.
MAX_ROW_SIZE = {"s8": 65536, "s16": 32767}
EPSILON_MAX = (1 << 62) - 1 - (1 << 60)  # keeps D = V + E below 2^62


def _seed_table() -> list[int]:
    """1/sqrt(X) at the middle of each [i/8, (i+1)/8) bucket of X in [1, 4), Q31."""
    return [int(round(2**31 / math.sqrt((i + 0.5) / 8))) for i in range(8, 32)]


SEED_TABLE = _seed_table()


# -- the per-row inverse square root ------------------------------------------------------------------------------


def inv_sqrt_multiplier(d: int) -> tuple[int, int]:
    """(multiplier, shift) with multiplier * 2^shift / 2^31 ~= 2^FRAC_BITS / sqrt(d), d in [1, 2^62).

    d is normalized by an even shift to dn = d * 4^a in [2^60, 2^62); x = dn >> 30 is X = dn / 2^60 in [1, 4)
    as Q30. Y ~= 1/sqrt(X) as Q31 starts from SEED_TABLE[x >> 27] and takes NEWTON_STEPS steps of
    Y <- Y * (3 - X * Y^2) / 2, all in truncating unsigned integer arithmetic. Then
    1/sqrt(d) = Y * 2^(a - 30) / 2^31, so the shift is FRAC_BITS + a - 30, in [-10, 20].
    """
    if not 0 < d < (1 << 62):
        raise ValueError(f"d out of range: {d}")
    p = d.bit_length() - 1  # highest set bit
    a = (61 - p) >> 1
    dn = d << (2 * a)
    x = dn >> 30  # Q30, [2^30, 2^32)
    y = SEED_TABLE[(x >> 27) - 8]
    for _ in range(NEWTON_STEPS):
        y2 = (y * y) >> 31  # Q31
        xy2 = (x * y2) >> 30  # Q31
        y = (y * ((3 << 31) - xy2)) >> 32  # Q31
    y = min(y, (1 << 31) - 1)
    return y, FRAC_BITS + a - 30


# -- the integer kernel --------------------------------------------------------------------------------------------


def qrdmulh(a: int, b: int) -> int:
    """VQRDMULH.S32 for operands that never both equal INT32_MIN: round-half-up (a * b) / 2^31."""
    return (a * b + (1 << 30)) >> 31


def rshr(v: int, shift: int) -> int:
    """VRSHL.S32 by -shift: round-half-up v / 2^shift (shift >= 0)."""
    return (v + (1 << (shift - 1))) >> shift if shift > 0 else v


def row_multiplier(d: int) -> tuple[int, int]:
    """(m, L): 2^FRAC_BITS / sqrt(d) ~= m * 2^L / 2^31 with L >= 0. A negative exponent is folded into m (which
    keeps at least 21 significant bits), so the per-element path needs no rounding shift. d == 0 -> (0, 0)."""
    if d == 0:
        return 0, 0
    m, shift = inv_sqrt_multiplier(d)
    if shift < 0:
        return rshr(m, -shift), 0
    return m, shift


@dataclass(frozen=True)
class IntParams:
    epsilon: int  # E
    output_shift: int  # T, [1, 31]
    gamma: np.ndarray  # int32 [N], G
    bias: np.ndarray  # int32 [N], B
    act_min: int
    act_max: int


def layer_norm_int(x: np.ndarray, p: IntParams) -> np.ndarray:
    """arm_layer_norm_s8 / _s16 on x of shape [rows, N] (int8 or int16); returns the same dtype."""
    rows, n = x.shape
    out = np.empty_like(x)
    for r in range(rows):
        row = [int(v) for v in x[r]]
        s = sum(row)
        q = sum(v * v for v in row)
        m, lsh = row_multiplier(n * q - s * s + p.epsilon)
        for c, v in enumerate(row):
            t = (n * v - s) << lsh
            assert -(1 << 31) <= t < (1 << 31), "t overflows int32: outside the supported range"
            z = qrdmulh(t, m)
            acc = qrdmulh(z, int(p.gamma[c])) + int(p.bias[c])
            out[r, c] = min(max(rshr(acc, p.output_shift), p.act_min), p.act_max)
    return out


# -- export: float layer-norm parameters -> kernel parameters ------------------------------------------------------


def export_params(gamma, beta, eps: float, in_scale: float, out_scale: float, out_zp: int, n: int,
                  dtype: str) -> IntParams:
    """Kernel parameters for layer norm with affine gamma/beta (None = identity) at the given quantization."""
    if dtype not in MAX_ROW_SIZE:
        raise ValueError(dtype)
    if not 1 <= n <= MAX_ROW_SIZE[dtype]:
        raise ValueError(f"row size {n} outside [1, {MAX_ROW_SIZE[dtype]}] for {dtype}")
    gamma = np.ones(n) if gamma is None else np.asarray(gamma, dtype=np.float64)
    beta = np.zeros(n) if beta is None else np.asarray(beta, dtype=np.float64)
    if gamma.shape != (n,) or beta.shape != (n,):
        raise ValueError("gamma and beta need one value per element of the row")
    epsilon = min(int(round(n * n * eps / (in_scale * in_scale))), EPSILON_MAX)
    ratio = gamma / out_scale
    amax = float(np.max(np.abs(ratio)))
    # The shared exponent P keeps every G in int32 (amax < 2^P) and every bias below 2^29 in magnitude; the output
    # shift T = F - P is at most 31.
    exponent = max(math.frexp(amax)[1] if amax > 0 else FRAC_BITS - 31, FRAC_BITS - 31)
    bmax = float(np.max(np.abs(beta / out_scale))) + abs(int(out_zp))
    if bmax > 0:
        exponent = max(exponent, FRAC_BITS - (29 - math.frexp(bmax)[1]))
    output_shift = FRAC_BITS - exponent
    if output_shift < 1:
        raise ValueError(f"|gamma| / output_scale up to {amax:.4g} is too large (needs < 2^{FRAC_BITS - 1}); "
                         "use a larger output scale")
    g = np.clip(np.round(ratio * 2.0 ** (31 - exponent)), -(2**31 - 1), 2**31 - 1).astype(np.int64)
    bias = np.round(beta / out_scale * 2.0**output_shift).astype(np.int64) + (int(out_zp) << output_shift)
    if np.any(np.abs(bias) >= 2**30):
        raise ValueError("beta / output_scale too large for the output shift; use a larger output scale")
    info = np.iinfo(np.int8 if dtype == "s8" else np.int16)
    return IntParams(epsilon=epsilon, output_shift=output_shift, gamma=g.astype(np.int32),
                     bias=bias.astype(np.int32), act_min=int(info.min), act_max=int(info.max))


# -- float -----------------------------------------------------------------------------------------------------------


def layer_norm_float(x: np.ndarray, gamma=None, beta=None, eps: float = 1e-5) -> np.ndarray:
    """Float64 layer norm over the last dimension (the accuracy reference for all three kernels)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        y = y * np.asarray(gamma, dtype=np.float64)
    if beta is not None:
        y = y + np.asarray(beta, dtype=np.float64)
    return y
