# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Generate the arm_nn_exp_f32 and arm_nn_exp_f16 goldens.

float32: inputs and their float64 exponential. float16: inputs and the binary16 nearest
their exponential. Both references are exp(x) evaluated in 60-digit decimal arithmetic
and rounded once, so the data depend on no libm and are the same on every host.

Each input also carries the exact bits of the kernel's scalar body, from port_exp_f32
below: the body's operations in exact rational arithmetic, each rounded once to float32
as the C operation rounds. Those pin the body bit for bit, which the float64 bound
cannot. --check regenerates in memory and exits 1 if a checked-in header differs.
"""
import argparse
import struct
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 60
HERE = Path(__file__).parent
LN2 = Decimal(2).ln()


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def exp_f64(bits):
    """float64 exp of the float32 with these bits; NaN inputs keep a NaN placeholder."""
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return 0x7FF8000000000000
    x = f32_value(bits)
    if x > 710.0:  # beyond float64, +inf included
        return f64_bits(float("inf"))
    if x < -746.0:  # below half the smallest float64 subnormal, -inf included
        return 0
    return f64_bits(float(Decimal(x).exp()))


def nearest_f32(value):
    """Bits of the float32 nearest to a Decimal (via float64, whose rounding is exact enough here)."""
    return f32_bits(float(value))


def f32_inputs():
    specials = [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # +-0, +-inf
        0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFC00001, 0x7FFFFFFF, 0xFFFFFFFF,  # quiet NaNs
        0x7F800001, 0xFF800001, 0x7FA00000, 0xFFB12345, 0x7FBFFFFF, 0x7F80ABCD,  # signalling NaNs
        0x00800000, 0x80800000,  # +-FLT_MIN
        0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,  # smallest and largest subnormals
        0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,  # +-1, +-FLT_MAX
        0x42B0C0A6,  # the first input a reassociated scaling overflows at
    ]
    cases = list(specials)
    # The two thresholds (-103.97f, 88.7228317f) with two neighbours on each side;
    # 0x42B17217 is the largest input with a finite result, 0x42B17218 its successor.
    for threshold in (0xC2CFF0A4, 0x42B17217):
        cases += [threshold + delta for delta in range(-2, 3)]
    # The subnormal boundary: the input nearest ln(2^-126) = -87.3366 and four neighbours each side.
    boundary = nearest_f32(Decimal(-126) * LN2)
    cases += [boundary + delta for delta in range(-4, 5)]
    # Every reduction boundary x = (k + 0.5) ln 2, k in [-150, 128], and its two neighbours.
    for k in range(-150, 129):
        centre = nearest_f32((Decimal(k) + Decimal("0.5")) * LN2)
        cases += [centre - 1, centre, centre + 1]
    # A stride through [-104, 89].
    steps = 1300
    for i in range(steps + 1):
        cases.append(nearest_f32(Decimal(-104) + Decimal(193) * i / steps))
    unique = []
    seen = set()
    for bits in cases:
        if bits not in seen:
            seen.add(bits)
            unique.append(bits)
    return unique


def round_f32(value):
    """Bits of the float32 nearest to a rational, ties to even; subnormals kept, overflow to +inf."""
    sign = 0x80000000 if value < 0 else 0
    a = abs(value)
    if a == 0:
        return sign
    exponent = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** exponent > a:
        exponent -= 1
    exponent = max(exponent, -126)
    quantum = Fraction(2) ** (exponent - 23)
    q = a / quantum
    whole = q.numerator // q.denominator
    rest = q - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2):
        whole += 1
    if whole == 1 << 24:
        whole, exponent = 1 << 23, exponent + 1
    if exponent > 127:
        return sign | 0x7F800000
    if whole < 1 << 23:
        return sign | whole
    return sign | ((exponent + 127) << 23) | (whole - (1 << 23))


def exact_f32(bits):
    """The finite float32 with these bits as an exact rational."""
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude >> 23
    mantissa = magnitude & 0x7FFFFF
    value = Fraction(mantissa, 1 << 23) * Fraction(2) ** -126 if exponent == 0 else \
        Fraction((1 << 23) | mantissa, 1 << 23) * Fraction(2) ** (exponent - 127)
    return -value if bits & 0x80000000 else value


def literal_f32(text):
    """A C float literal, rounded once from its decimal value as the compiler does."""
    return exact_f32(round_f32(Fraction(Decimal(text))))


def fmaf(a, b, c):
    return exact_f32(round_f32(a * b + c))


LOG2E = literal_f32("1.44269502")
LN2_HI = literal_f32("0.693145752")
LN2_LO = literal_f32("1.42860677e-06")
COEFFICIENTS = [literal_f32(c) for c in ("0.00138427224", "0.00837495551", "0.0416680574", "0.166664213", "0.49999994")]


def port_exp_f32(bits):
    """Bits arm_nn_exp_scalar_f32 returns for the float32 with these bits, step for step."""
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return bits | 0x00400000
    if bits > 0xC2CFF0A4:
        return 0
    if bits < 0x80000000 and bits > 0x42B17217:
        return 0x7F800000
    x = exact_f32(bits)
    round_half = Fraction(-1, 2) if bits & 0x80000000 else Fraction(1, 2)
    t = fmaf(x, LOG2E, round_half)
    n = int(t)  # truncation toward zero, as the C cast
    nf = Fraction(n)
    r = fmaf(nf, -LN2_HI, x)
    r = fmaf(nf, -LN2_LO, r)
    c6, c5, c4, c3, c2 = COEFFICIENTS
    p = fmaf(c6, r, c5)
    for c in (c4, c3, c2, Fraction(1), Fraction(1)):
        p = fmaf(p, r, c)
    half = abs(n) // 2 * (1 if n >= 0 else -1)
    scaled = (round_f32(p) + (half << 23)) & 0xFFFFFFFF
    second = ((n - half + 127) << 23) & 0xFFFFFFFF
    return round_f32(exact_f32(scaled) * exact_f32(second))


def port_exp_f16(bits):
    """Bits arm_nn_exp_f16 returns for the binary16 with these bits: NaN quieted, else the
    body at the widened input rounded once to binary16."""
    if (bits & 0x7FFF) > 0x7C00:
        return bits | 0x0200
    x = f16_value(bits)
    y = port_exp_f32(f32_bits(x))  # binary16 widens to float32 exactly
    if y == 0x7F800000:
        return 0x7C00
    return round_f16(Decimal(exact_f32(y).numerator) / Decimal(exact_f32(y).denominator))


def rows(values, fmt, per_line):
    lines = []
    for start in range(0, len(values), per_line):
        lines.append("    " + ", ".join(fmt.format(v) for v in values[start:start + per_line]) + ",")
    return lines


def f32_header():
    inputs = f32_inputs()
    references = [exp_f64(bits) for bits in inputs]
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: 2026 Ambiq",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * Generated by ../generate_exp_flt_data.py; do not edit. float32 inputs and the float64",
        " * exponential of each, exp(x) in 60-digit decimal arithmetic rounded once to float64.",
        " * NaN inputs carry a NaN placeholder the suite does not read.",
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define EXP_F32_CASES {len(inputs)}",
        "",
        "static const uint32_t exp_f32_input_bits[EXP_F32_CASES] = {",
    ]
    lines += rows(inputs, "0x{:08X}u", 8)
    lines += ["};", "", "static const uint64_t exp_f32_reference_bits[EXP_F32_CASES] = {"]
    lines += rows(references, "0x{:016X}ull", 4)
    lines += ["};", "", "/* The exact bits of the scalar body, arm_nn_exp_scalar_f32, for each input. */",
              "static const uint32_t exp_f32_exact_bits[EXP_F32_CASES] = {"]
    lines += rows([port_exp_f32(bits) for bits in inputs], "0x{:08X}u", 8)
    lines += ["};", ""]
    return "\n".join(lines)


def f16_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def round_f16(value):
    """Bits of the binary16 nearest to a non-negative Decimal, ties to even, exactly."""
    if value == 0:
        return 0
    exponent = -14
    while Decimal(2) ** (exponent + 1) <= value:
        exponent += 1
    quantum = Decimal(2) ** (exponent - 10)
    q = value / quantum
    whole = int(q)
    rest = q - whole
    if rest > Decimal("0.5") or (rest == Decimal("0.5") and whole % 2):
        whole += 1
    if whole == 2048:
        whole, exponent = 1024, exponent + 1
    if exponent > 15:
        return 0x7C00
    if whole < 1024:
        return whole  # subnormal, or zero
    return ((exponent + 15) << 10) | (whole - 1024)


def exp_f16(bits):
    """binary16 nearest the exponential of the binary16 with these bits; NaN inputs give 0."""
    if (bits & 0x7FFF) > 0x7C00:
        return 0
    x = f16_value(bits)
    if x > 12.0:  # beyond binary16, +inf included
        return 0x7C00
    if x < -18.0:  # below half the smallest binary16 subnormal, -inf included
        return 0
    return round_f16(Decimal(x).exp())


def f16_inputs():
    specials = [
        0x0000, 0x8000, 0x7C00, 0xFC00,  # +-0, +-inf
        0x7E00, 0xFE00, 0x7E5A, 0xFFFF, 0x7C01, 0xFC01, 0x7D00, 0xFDFF,  # quiet and signalling NaNs
        0x0400, 0x8400, 0x0001, 0x8001, 0x03FF, 0x83FF,  # +-min normal, smallest and largest subnormals
        0x3C00, 0xBC00, 0x7BFF, 0xFBFF,  # +-1, +-max
    ]
    cases = list(specials)
    # Where the result crosses a binary16 boundary: overflow (exp(x) >= 65520), the
    # subnormal range (exp(x) < 2^-14) and zero (exp(x) <= 2^-25); eight neighbours each side.
    for edge in (Decimal(65520).ln(), Decimal(-14) * LN2, Decimal(-25) * LN2):
        centre = struct.unpack("<H", struct.pack("<e", float(edge)))[0]
        cases += [centre + delta for delta in range(-8, 9)]
    # Every 16th bit pattern.
    cases += list(range(0, 0x10000, 16))
    unique = []
    seen = set()
    for bits in cases:
        if bits not in seen:
            seen.add(bits)
            unique.append(bits)
    return unique


def f16_header():
    inputs = f16_inputs()
    references = [exp_f16(bits) for bits in inputs]
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: 2026 Ambiq",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * Generated by ../generate_exp_flt_data.py; do not edit. binary16 inputs and the binary16",
        " * nearest the exponential of each, exp(x) in 60-digit decimal arithmetic rounded once,",
        " * ties to even. NaN inputs carry 0, which the suite does not read.",
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define EXP_F16_CASES {len(inputs)}",
        "",
        "static const uint16_t exp_f16_input_bits[EXP_F16_CASES] = {",
    ]
    lines += rows(inputs, "0x{:04X}u", 10)
    lines += ["};", "", "static const uint16_t exp_f16_reference_bits[EXP_F16_CASES] = {"]
    lines += rows(references, "0x{:04X}u", 10)
    lines += ["};", "", "/* The exact bits arm_nn_exp_f16 returns for each input. */",
              "static const uint16_t exp_f16_exact_bits[EXP_F16_CASES] = {"]
    lines += rows([port_exp_f16(bits) for bits in inputs], "0x{:04X}u", 10)
    lines += ["};", ""]
    return "\n".join(lines)


def outputs():
    return {
        HERE / "test_arm_nn_exp_f32" / "exp_f32_data.h": f32_header(),
        HERE / "test_arm_nn_exp_f16" / "exp_f16_data.h": f16_header(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a checked-in header differs")
    args = parser.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
        else:
            path.write_text(text)
    for path in stale:
        print(f"stale: {path.relative_to(HERE)}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
