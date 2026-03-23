#!/usr/bin/env python3
"""Generate RSQRT s16 unit-test data headers.

This script emits deterministic test data for the `test_arm_rsqrt_s16` unit test.
It generates:

- shared config macros
- input tensors
- universal/per-op LUTs
- expected universal/per-op outputs

The output layout matches the existing `Tests/UnitTest/TestCases/TestData/*`
pattern used by other CMSIS-NN unit tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


TEST_LUT_SIZE = 513
NEGATIVE_BLOCK_SIZE = 4
SLOT_SHIFT = 7
SLOT_MASK = (1 << SLOT_SHIFT) - 1
INTERP_ROUND_BIAS = 1 << (SLOT_SHIFT - 1)
BASE_STEP_SHIFT = 6
BASE_LUT_SIZE = 513
INPUT_OFFSET = 0
OUT_OFFSET = 0
OUT_MULT = 0
OUT_SHIFT = 0
NEEDS_RESCALE = 0
OUT_ACTIVATION_MIN = -32768
OUT_ACTIVATION_MAX = 32767


def build_input_values() -> list[int]:
    """Build a deterministic RSQRT stress sweep across the full int16 input domain.

    The base stride of 64 exercises every 128-wide interpolation slot at its
    midpoint. Additional `boundary - 1`, `boundary`, and `boundary + 1` samples for
    every slot boundary stress the interpolation transitions used by the universal
    path. The final `32767` sample ensures the top-domain clamp path is covered.
    """
    values: set[int] = set(range(0, 32768, 64))

    for boundary in range(0, 32769, 1 << SLOT_SHIFT):
        if 0 <= boundary <= 32767:
            values.add(boundary)
        if 0 <= boundary - 1 <= 32767:
            values.add(boundary - 1)
        if 0 <= boundary + 1 <= 32767:
            values.add(boundary + 1)

    values.add(32767)
    return sorted(values)


def clamp_base_lut_index(q_value: int) -> int:
    """Clamp a universal-LUT sample index to the valid generated table range."""
    lut_index = (q_value + ((1 << BASE_STEP_SHIFT) - 1)) >> BASE_STEP_SHIFT
    return min(lut_index, BASE_LUT_SIZE - 1)


def round_div2_nearest(value: int) -> int:
    """Divide by two using the same sign-aware rounding as the C reference."""
    if value >= 0:
        return (value + 1) >> 1
    return -(((-value) + 1) >> 1)


def fill_per_op_lut() -> list[int]:
    """Build the deterministic per-op LUT used by the RSQRT unit test."""
    lut: list[int] = []
    for i in range(TEST_LUT_SIZE):
        value = 32767 - (i * 41)
        lut.append(max(-32768, value))
    return lut


def fill_universal_lut() -> list[int]:
    """Build the deterministic universal LUT used by the RSQRT unit test."""
    lut = [0]
    for i in range(1, TEST_LUT_SIZE):
        value = 32000 - (i * 53)
        lut.append(max(0, value))
    return lut


def universal_sample_slot(lut: Sequence[int], slot_index: int) -> int:
    """Sample one corrected universal LUT slot using the C test's reference math."""
    q0 = -32768 + (slot_index << SLOT_SHIFT)
    qmid = q0 + (1 << BASE_STEP_SHIFT)
    q1 = q0 + (1 << SLOT_SHIFT)

    s0 = 32767 if q0 <= 0 else lut[clamp_base_lut_index(q0)]
    sm = 32767 if qmid <= 0 else lut[clamp_base_lut_index(qmid)]
    s1 = 32767 if q1 <= 0 else lut[clamp_base_lut_index(q1)]

    s0 = min(32767, max(-32768, s0))
    sm = min(32767, max(-32768, sm))
    s1 = min(32767, max(-32768, s1))

    midpoint_interp = (s0 + s1 + 1) >> 1
    midpoint_err = midpoint_interp - sm
    bias = round_div2_nearest(midpoint_err)
    slot_value = s0 - bias
    return min(32767, max(-32768, slot_value))


def generate_universal_expected(input_values: Sequence[int], lut: Sequence[int]) -> list[int]:
    """Generate expected outputs for the universal RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw + INPUT_OFFSET)
        shifted_value = value - (-32768)
        index = (shifted_value >> SLOT_SHIFT) & 0x1FF
        frac = shifted_value & SLOT_MASK
        y0 = universal_sample_slot(lut, index)
        y1 = universal_sample_slot(lut, index + 1)
        interp_delta = (((y1 - y0) * frac) + INTERP_ROUND_BIAS) >> SLOT_SHIFT
        rsqrt_res = y0 + interp_delta + OUT_OFFSET
        rsqrt_res = min(OUT_ACTIVATION_MAX, max(OUT_ACTIVATION_MIN, rsqrt_res))
        output.append(rsqrt_res)
    return output


def generate_per_op_expected(input_values: Sequence[int], lut: Sequence[int]) -> list[int]:
    """Generate expected outputs for the per-op RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw + INPUT_OFFSET)
        index = 256 + (value >> SLOT_SHIFT)
        frac = value & SLOT_MASK
        rsqrt_res = lut[index] + (((lut[index + 1] - lut[index]) * frac + INTERP_ROUND_BIAS) >> SLOT_SHIFT)
        rsqrt_res += OUT_OFFSET
        rsqrt_res = min(OUT_ACTIVATION_MAX, max(OUT_ACTIVATION_MIN, rsqrt_res))
        output.append(rsqrt_res)
    return output


def format_scalar_macro(name: str, value: int) -> str:
    """Format a `#define` line for a generated configuration macro."""
    return f"#define {name} ({value})"


def format_array(name: str, c_type: str, values: Sequence[int], values_per_line: int = 8) -> str:
    """Format a generated C array definition."""
    lines = [f"static const {c_type} {name}[{len(values)}] = {{"]
    for start in range(0, len(values), values_per_line):
        chunk = values[start:start + values_per_line]
        lines.append("    " + ", ".join(str(value) for value in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_header(path: Path, lines: Iterable[str]) -> None:
    """Write a generated header file with a standard banner."""
    path.write_text("// Generated by generate_rsqrt_s16_test_data.py.\n" + "\n".join(lines) + "\n", encoding="utf-8")


def generate_headers(output_dir: Path) -> None:
    """Generate the RSQRT s16 test-data headers into the requested directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    input_values = build_input_values()
    negative_input_values = [-1, 1, 2, 3]
    universal_lut = fill_universal_lut()
    per_op_lut = fill_per_op_lut()
    universal_output = generate_universal_expected(input_values, universal_lut)
    per_op_output = generate_per_op_expected(input_values, per_op_lut)

    write_header(
        output_dir / "config_data.h",
        [
            format_scalar_macro("RSQRT_S16_BLOCK_SIZE", len(input_values)),
            format_scalar_macro("RSQRT_S16_LUT_SIZE", TEST_LUT_SIZE),
            format_scalar_macro("RSQRT_S16_NEGATIVE_BLOCK_SIZE", NEGATIVE_BLOCK_SIZE),
            format_scalar_macro("RSQRT_S16_INPUT_OFFSET", INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OUT_OFFSET", OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OUT_MULT", OUT_MULT),
            format_scalar_macro("RSQRT_S16_OUT_SHIFT", OUT_SHIFT),
            format_scalar_macro("RSQRT_S16_NEEDS_RESCALE", NEEDS_RESCALE),
            format_scalar_macro("RSQRT_S16_OUT_ACTIVATION_MIN", OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_OUT_ACTIVATION_MAX", OUT_ACTIVATION_MAX),
        ],
    )
    write_header(output_dir / "input_tensor.h", [format_array("rsqrt_s16_input", "int16_t", input_values)])
    write_header(
        output_dir / "negative_input_tensor.h",
        [format_array("rsqrt_s16_negative_input", "int16_t", negative_input_values)],
    )
    write_header(output_dir / "universal_lut_data.h", [format_array("rsqrt_s16_universal_lut", "int32_t", universal_lut)])
    write_header(output_dir / "per_op_lut_data.h", [format_array("rsqrt_s16_per_op_lut", "int16_t", per_op_lut)])
    write_header(output_dir / "universal_output.h", [format_array("rsqrt_s16_universal_output", "int16_t", universal_output)])
    write_header(output_dir / "per_op_output.h", [format_array("rsqrt_s16_per_op_output", "int16_t", per_op_output)])
    write_header(
        output_dir / "test_data.h",
        [
            "#include \"config_data.h\"",
            "#include \"input_tensor.h\"",
            "#include \"negative_input_tensor.h\"",
            "#include \"universal_lut_data.h\"",
            "#include \"per_op_lut_data.h\"",
            "#include \"universal_output.h\"",
            "#include \"per_op_output.h\"",
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RSQRT test-data generator."""
    default_output = Path(__file__).resolve().parent.parent / "TestData" / "rsqrt_s16"
    parser = argparse.ArgumentParser(description="Generate RSQRT s16 unit-test data headers.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory where generated headers are written.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    generate_headers(args.output_dir)
    print(f"Generated RSQRT s16 test data in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())