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
import math
from pathlib import Path
from typing import Iterable, Sequence


TEST_LUT_SIZE = 513
NEGATIVE_BLOCK_SIZE = 4
SLOT_SHIFT = 7
BASE_STEP_SHIFT = 6
MIN_REFERENCE_INPUT_VALUE = 256

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

UNIVERSAL_TOLERANCE = 96
PER_OP_TOLERANCE = 160
OFFSET_PER_OP_TOLERANCE = 160
RESCALED_UNIVERSAL_TOLERANCE = 48


def build_input_values(min_value: int = MIN_REFERENCE_INPUT_VALUE, max_value: int = 32767) -> list[int]:
    """Build a deterministic RSQRT stress sweep across the stable input domain.

    The generated LUTs approximate a singular `1/sqrt(x)` curve, so we skip the
    near-zero region where any 128-step linear approximation would require a very
    loose tolerance. The remaining sweep still covers every interpolation slot
    boundary and midpoint that the kernels exercise.
    """
    values: set[int] = set(range(min_value, max_value + 1, 64))

    for boundary in range(0, 32769, 1 << SLOT_SHIFT):
        if min_value <= boundary <= max_value:
            values.add(boundary)
        if min_value <= boundary - 1 <= max_value:
            values.add(boundary - 1)
        if min_value <= boundary + 1 <= max_value:
            values.add(boundary + 1)

    values.add(max_value)
    return sorted(values)


def add_input_offset(values: Sequence[int], input_offset: int) -> list[int]:
    """Encode logical non-negative values into the quantized int16 storage domain."""
    return [input_offset + value for value in values]


def quantized_rsqrt_reference(value: int) -> int:
    """Return an INT16-domain floating-point RSQRT reference for one logical input."""
    domain_value = 1.0 + (float(max(value, 0)) / float(1 << SLOT_SHIFT))
    rsqrt = 1.0 / math.sqrt(domain_value)
    return min(32767, max(-32768, int(round(32767.0 * rsqrt))))


def fill_per_op_lut() -> list[int]:
    """Sample a real `1/sqrt(x)` curve at per-op slot boundaries."""
    lut: list[int] = []
    for i in range(TEST_LUT_SIZE):
        logical_value = max(0, (i - 256) << SLOT_SHIFT)
        lut.append(quantized_rsqrt_reference(logical_value))
    return lut


def fill_universal_lut() -> list[int]:
    """Sample the shared universal LUT from the same floating-point RSQRT curve."""
    return [quantized_rsqrt_reference(i << BASE_STEP_SHIFT) for i in range(TEST_LUT_SIZE)]


def requantize_float(value: int, multiplier: int, shift: int) -> int:
    """Apply the output scaling as a floating-point reference transform."""
    scale = math.ldexp(multiplier / float(1 << 31), shift)
    return int(round(value * scale))


def generate_universal_expected(
    input_values: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
    out_activation_min: int,
    out_activation_max: int,
) -> list[int]:
    """Generate floating-point reference outputs for the universal RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw - input_offset)
        if value < 0:
            raise ValueError(
                "Generated universal test input must remain non-negative after subtracting input_offset."
            )
        rsqrt_res = quantized_rsqrt_reference(value)
        if needs_rescale:
            rsqrt_res = requantize_float(rsqrt_res, out_mult, out_shift)
        rsqrt_res += out_offset
        rsqrt_res = min(out_activation_max, max(out_activation_min, rsqrt_res))
        output.append(rsqrt_res)
    return output


def generate_per_op_expected(
    input_values: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_activation_min: int,
    out_activation_max: int,
) -> list[int]:
    """Generate floating-point reference outputs for the per-op RSQRT entry point."""
    output: list[int] = []
    for raw in input_values:
        value = min(0x7FFF, raw - input_offset)
        if value < 0:
            raise ValueError(
                "Generated per-op test input must remain non-negative after subtracting input_offset."
            )
        rsqrt_res = quantized_rsqrt_reference(value) + out_offset
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
    offset_input_values = add_input_offset(
        build_input_values(max_value=32767 - OFFSET_INPUT_OFFSET),
        OFFSET_INPUT_OFFSET,
    )
    rescale_input_values = add_input_offset(
        build_input_values(max_value=32767 - RESCALE_INPUT_OFFSET),
        RESCALE_INPUT_OFFSET,
    )
    negative_input_values = [-1, 1, 2, 3]
    universal_lut = fill_universal_lut()
    per_op_lut = fill_per_op_lut()

    universal_output = generate_universal_expected(
        input_values,
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
        INPUT_OFFSET,
        OUT_OFFSET,
        OUT_ACTIVATION_MIN,
        OUT_ACTIVATION_MAX,
    )
    offset_per_op_output = generate_per_op_expected(
        offset_input_values,
        OFFSET_INPUT_OFFSET,
        OFFSET_OUT_OFFSET,
        OFFSET_OUT_ACTIVATION_MIN,
        OFFSET_OUT_ACTIVATION_MAX,
    )
    rescaled_universal_output = generate_universal_expected(
        rescale_input_values,
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
            format_scalar_macro("RSQRT_S16_UNIVERSAL_TOLERANCE", UNIVERSAL_TOLERANCE),
            format_scalar_macro("RSQRT_S16_PER_OP_TOLERANCE", PER_OP_TOLERANCE),
            format_scalar_macro("RSQRT_S16_OFFSET_BLOCK_SIZE", len(offset_input_values)),
            format_scalar_macro("RSQRT_S16_OFFSET_INPUT_OFFSET", OFFSET_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_OFFSET", OFFSET_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MIN", OFFSET_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MAX", OFFSET_OUT_ACTIVATION_MAX),
            format_scalar_macro("RSQRT_S16_OFFSET_PER_OP_TOLERANCE", OFFSET_PER_OP_TOLERANCE),
            format_scalar_macro("RSQRT_S16_RESCALE_BLOCK_SIZE", len(rescale_input_values)),
            format_scalar_macro("RSQRT_S16_RESCALE_INPUT_OFFSET", RESCALE_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_OFFSET", RESCALE_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_MULT", RESCALE_OUT_MULT),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_SHIFT", RESCALE_OUT_SHIFT),
            format_scalar_macro("RSQRT_S16_RESCALE_NEEDS_RESCALE", RESCALE_NEEDS_RESCALE),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MIN", RESCALE_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MAX", RESCALE_OUT_ACTIVATION_MAX),
            format_scalar_macro("RSQRT_S16_RESCALED_UNIVERSAL_TOLERANCE", RESCALED_UNIVERSAL_TOLERANCE),
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
