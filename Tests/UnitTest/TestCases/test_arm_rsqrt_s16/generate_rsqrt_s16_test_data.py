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

OFFSET_INPUT_OFFSET = 321
OFFSET_OUT_OFFSET = 0
OFFSET_OUT_ACTIVATION_MIN = -32768
OFFSET_OUT_ACTIVATION_MAX = 32767

RESCALE_INPUT_OFFSET = 193
RESCALE_OUT_OFFSET = 0
RESCALE_OUT_MULT = 1 << 30
RESCALE_OUT_SHIFT = 0
RESCALE_NEEDS_RESCALE = 1
RESCALE_OUT_ACTIVATION_MIN = -32768
RESCALE_OUT_ACTIVATION_MAX = 32767


def build_input_values(max_value: int = 32767) -> list[int]:
    """Build a deterministic RSQRT stress sweep across the input domain.

    The base stride of 64 exercises every 128-wide interpolation slot at its
    midpoint. Additional `boundary - 1`, `boundary`, and `boundary + 1` samples for
    every slot boundary stress the interpolation transitions used by the universal
    path. The final `max_value` sample ensures the top-domain clamp path is covered.
    """
    values: set[int] = set(range(0, max_value + 1, 64))

    for boundary in range(0, 32769, 1 << SLOT_SHIFT):
        if 0 <= boundary <= max_value:
            values.add(boundary)
        if 0 <= boundary - 1 <= max_value:
            values.add(boundary - 1)
        if 0 <= boundary + 1 <= max_value:
            values.add(boundary + 1)

    values.add(max_value)
    return sorted(values)


def add_input_offset(values: Sequence[int], input_offset: int) -> list[int]:
    """Encode logical non-negative values into the quantized int16 storage domain."""
    return [input_offset + value for value in values]


def clamp_base_lut_index(q_value: int) -> int:
    """Clamp a universal-LUT sample index to the valid generated table range."""
    lut_index = (q_value + ((1 << BASE_STEP_SHIFT) - 1)) >> BASE_STEP_SHIFT
    return min(lut_index, BASE_LUT_SIZE - 1)


def round_div2_nearest(value: int) -> int:
    """Divide by two using the same sign-aware rounding as the C reference."""
    if value >= 0:
        return (value + 1) >> 1
    return -(((-value) + 1) >> 1)


def arm_nn_doubling_high_mult_no_sat(multiplicand: int, multiplier: int) -> int:
    """Mirror CMSIS-NN `arm_nn_doubling_high_mult_no_sat` for positive multipliers."""
    return ((1 << 30) + (multiplicand * multiplier)) >> 31


def arm_nn_divide_by_power_of_two(dividend: int, exponent: int) -> int:
    """Mirror CMSIS-NN midpoint-away-from-zero division by a power of two."""
    if exponent == 0:
        return dividend

    result = dividend >> exponent
    remainder_mask = (1 << exponent) - 1
    remainder = remainder_mask & dividend
    threshold = remainder_mask >> 1
    if result < 0:
        threshold += 1
    if remainder > threshold:
        result += 1
    return result


def arm_nn_requantize(value: int, multiplier: int, shift: int) -> int:
    """Mirror CMSIS-NN `arm_nn_requantize` default-branch behavior."""
    left_shift = shift if shift > 0 else 0
    right_shift = 0 if shift > 0 else -shift
    product = arm_nn_doubling_high_mult_no_sat(value * (1 << left_shift), multiplier)
    return arm_nn_divide_by_power_of_two(product, right_shift)


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


def universal_scaled_positive(
    lut: Sequence[int],
    q_value: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
) -> int:
    """Apply the shared LUT lookup plus optional CMSIS-NN requantization."""
    scaled = lut[clamp_base_lut_index(q_value)]
    if needs_rescale:
        scaled = arm_nn_requantize(scaled, out_mult, out_shift)
    return min(32767, max(-32768, scaled))


def universal_sample_slot(
    lut: Sequence[int],
    slot_index: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
) -> int:
    """Sample one corrected universal LUT slot using the C reference math."""
    q0 = -32768 + (slot_index << SLOT_SHIFT)
    qmid = q0 + (1 << BASE_STEP_SHIFT)
    q1 = q0 + (1 << SLOT_SHIFT)

    s0 = 32767 if q0 <= 0 else universal_scaled_positive(lut, q0, out_mult, out_shift, needs_rescale)
    sm = 32767 if qmid <= 0 else universal_scaled_positive(lut, qmid, out_mult, out_shift, needs_rescale)
    s1 = 32767 if q1 <= 0 else universal_scaled_positive(lut, q1, out_mult, out_shift, needs_rescale)

    midpoint_interp = (s0 + s1 + 1) >> 1
    midpoint_err = midpoint_interp - sm
    bias = round_div2_nearest(midpoint_err)
    slot_value = s0 - bias
    return min(32767, max(-32768, slot_value))


def generate_universal_expected(
    input_values: Sequence[int],
    lut: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
    out_activation_min: int,
    out_activation_max: int,
) -> list[int]:
    """Generate expected outputs for the universal RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw - input_offset)
        if value < 0:
            raise ValueError(
                "Generated universal test input must remain non-negative after subtracting input_offset."
            )
        shifted_value = value - (-32768)
        index = (shifted_value >> SLOT_SHIFT) & 0x1FF
        frac = shifted_value & SLOT_MASK
        y0 = universal_sample_slot(lut, index, out_mult, out_shift, needs_rescale)
        y1 = universal_sample_slot(lut, index + 1, out_mult, out_shift, needs_rescale)
        interp_delta = (((y1 - y0) * frac) + INTERP_ROUND_BIAS) >> SLOT_SHIFT
        rsqrt_res = y0 + interp_delta + out_offset
        rsqrt_res = min(out_activation_max, max(out_activation_min, rsqrt_res))
        output.append(rsqrt_res)
    return output


def generate_per_op_expected(
    input_values: Sequence[int],
    lut: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_activation_min: int,
    out_activation_max: int,
) -> list[int]:
    """Generate expected outputs for the per-op RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw - input_offset)
        if value < 0:
            raise ValueError(
                "Generated per-op test input must remain non-negative after subtracting input_offset."
            )
        index = 256 + (value >> SLOT_SHIFT)
        frac = value & SLOT_MASK
        rsqrt_res = lut[index] + (((lut[index + 1] - lut[index]) * frac + INTERP_ROUND_BIAS) >> SLOT_SHIFT)
        rsqrt_res += out_offset
        rsqrt_res = min(out_activation_max, max(out_activation_min, rsqrt_res))
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
    offset_input_values = add_input_offset(build_input_values(32767 - OFFSET_INPUT_OFFSET), OFFSET_INPUT_OFFSET)
    rescale_input_values = add_input_offset(build_input_values(32767 - RESCALE_INPUT_OFFSET), RESCALE_INPUT_OFFSET)
    negative_input_values = [-1, 1, 2, 3]
    universal_lut = fill_universal_lut()
    per_op_lut = fill_per_op_lut()

    universal_output = generate_universal_expected(
        input_values,
        universal_lut,
        INPUT_OFFSET,
        OUT_OFFSET,
        OUT_MULT,
        OUT_SHIFT,
        bool(NEEDS_RESCALE),
        OUT_ACTIVATION_MIN,
        OUT_ACTIVATION_MAX,
    )
    per_op_output = generate_per_op_expected(
        input_values,
        per_op_lut,
        INPUT_OFFSET,
        OUT_OFFSET,
        OUT_ACTIVATION_MIN,
        OUT_ACTIVATION_MAX,
    )
    offset_per_op_output = generate_per_op_expected(
        offset_input_values,
        per_op_lut,
        OFFSET_INPUT_OFFSET,
        OFFSET_OUT_OFFSET,
        OFFSET_OUT_ACTIVATION_MIN,
        OFFSET_OUT_ACTIVATION_MAX,
    )
    rescaled_universal_output = generate_universal_expected(
        rescale_input_values,
        universal_lut,
        RESCALE_INPUT_OFFSET,
        RESCALE_OUT_OFFSET,
        RESCALE_OUT_MULT,
        RESCALE_OUT_SHIFT,
        bool(RESCALE_NEEDS_RESCALE),
        RESCALE_OUT_ACTIVATION_MIN,
        RESCALE_OUT_ACTIVATION_MAX,
    )

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
            format_scalar_macro("RSQRT_S16_OFFSET_BLOCK_SIZE", len(offset_input_values)),
            format_scalar_macro("RSQRT_S16_OFFSET_INPUT_OFFSET", OFFSET_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_OFFSET", OFFSET_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MIN", OFFSET_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MAX", OFFSET_OUT_ACTIVATION_MAX),
            format_scalar_macro("RSQRT_S16_RESCALE_BLOCK_SIZE", len(rescale_input_values)),
            format_scalar_macro("RSQRT_S16_RESCALE_INPUT_OFFSET", RESCALE_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_OFFSET", RESCALE_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_MULT", RESCALE_OUT_MULT),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_SHIFT", RESCALE_OUT_SHIFT),
            format_scalar_macro("RSQRT_S16_RESCALE_NEEDS_RESCALE", RESCALE_NEEDS_RESCALE),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MIN", RESCALE_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MAX", RESCALE_OUT_ACTIVATION_MAX),
        ],
    )
    write_header(output_dir / "input_tensor.h", [format_array("rsqrt_s16_input", "int16_t", input_values)])
    write_header(output_dir / "offset_input_tensor.h", [format_array("rsqrt_s16_offset_input", "int16_t", offset_input_values)])
    write_header(output_dir / "negative_input_tensor.h", [format_array("rsqrt_s16_negative_input", "int16_t", negative_input_values)])
    write_header(output_dir / "rescale_input_tensor.h", [format_array("rsqrt_s16_rescale_input", "int16_t", rescale_input_values)])
    write_header(output_dir / "universal_lut_data.h", [format_array("rsqrt_s16_universal_lut", "int32_t", universal_lut)])
    write_header(output_dir / "per_op_lut_data.h", [format_array("rsqrt_s16_per_op_lut", "int16_t", per_op_lut)])
    write_header(output_dir / "universal_output.h", [format_array("rsqrt_s16_universal_output", "int16_t", universal_output)])
    write_header(output_dir / "per_op_output.h", [format_array("rsqrt_s16_per_op_output", "int16_t", per_op_output)])
    write_header(output_dir / "offset_per_op_output.h", [format_array("rsqrt_s16_offset_per_op_output", "int16_t", offset_per_op_output)])
    write_header(output_dir / "rescaled_universal_output.h", [format_array("rsqrt_s16_rescaled_universal_output", "int16_t", rescaled_universal_output)])
    write_header(
        output_dir / "test_data.h",
        [
            "#include \"config_data.h\"",
            "#include \"input_tensor.h\"",
            "#include \"offset_input_tensor.h\"",
            "#include \"negative_input_tensor.h\"",
            "#include \"rescale_input_tensor.h\"",
            "#include \"universal_lut_data.h\"",
            "#include \"per_op_lut_data.h\"",
            "#include \"universal_output.h\"",
            "#include \"per_op_output.h\"",
            "#include \"offset_per_op_output.h\"",
            "#include \"rescaled_universal_output.h\"",
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