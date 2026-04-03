#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Generate RSQRT s16 unit-test fixtures using the RefactoredTestGen stack.

This keeps LUT generation independent from the expected-output oracle:

- expected outputs come from quantized TFLite interpreter runs
- LUTs are sampled independently from the analytic rsqrt curve
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable
from typing import Sequence

import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import Lib.test
from Lib.keras_compat import keras


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

MODEL_INPUT_SCALE = 1.0 / float(1 << SLOT_SHIFT)
MODEL_INPUT_ZERO_POINT = 0


def build_input_values(min_value: int = MIN_REFERENCE_INPUT_VALUE, max_value: int = 32767) -> list[int]:
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
    return [input_offset + value for value in values]


def rsqrt_curve(value: int) -> float:
    domain_value = 1.0 + (float(max(value, 0)) / float(1 << SLOT_SHIFT))
    return 1.0 / math.sqrt(domain_value)


def quantized_rsqrt_curve(value: int) -> int:
    return int(np.clip(np.rint(32767.0 * rsqrt_curve(value)), -32768, 32767))


def fill_per_op_lut() -> list[int]:
    lut: list[int] = []
    for i in range(TEST_LUT_SIZE):
        logical_value = max(0, (i - 256) << SLOT_SHIFT)
        lut.append(quantized_rsqrt_curve(logical_value))
    return lut


def fill_universal_lut() -> list[int]:
    lut: list[int] = []
    for i in range(TEST_LUT_SIZE):
        lut.append(quantized_rsqrt_curve(i << BASE_STEP_SHIFT))
    return lut


def build_fixed_shape_rsqrt_model(width: int, scale_factor: float):
    input_tensor = keras.layers.Input(batch_input_shape=(1, width), dtype="float32")
    output_tensor = keras.ops.rsqrt(input_tensor + 1.0) * scale_factor
    return keras.Model([input_tensor], [output_tensor])


def convert_model_to_tflite(tflite_path: Path, width: int, scale_factor: float) -> None:
    model = build_fixed_shape_rsqrt_model(width, scale_factor)
    shape = {
        "representational_dataset": (1, width),
        "representational_dataset_min": 0.0,
        "representational_dataset_max": 32767.0 / float(1 << SLOT_SHIFT),
    }
    Lib.test.convert_keras_to_tflite(
        output_fpath=tflite_path,
        keras_model=model,
        quantize=True,
        input_dtype="int16_t",
        bias_dtype=None,
        shape=shape,
        output_dtype="int16_t",
    )


def run_tflite_reference(
    logical_values: Sequence[int],
    scale_factor: float,
    tflite_path: Path,
) -> tuple[list[int], float, int]:
    width = len(logical_values)
    convert_model_to_tflite(tflite_path, width, scale_factor)

    interpreter = Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    if input_scale <= 0.0:
        raise RuntimeError("Invalid input quantization for RSQRT reference model")

    input_tensor = np.rint((np.asarray(logical_values, dtype=np.float32) * MODEL_INPUT_SCALE) / input_scale)
    input_tensor = input_tensor.astype(np.int32) + int(input_zero_point)
    input_tensor = np.clip(input_tensor, -32768, 32767).astype(np.int16).reshape((1, width))

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details["index"]).reshape(-1).astype(np.int16)
    return output_tensor.astype(int).tolist(), float(output_scale), int(output_zero_point)


def apply_float_rescale_reference(
    raw_outputs: Sequence[int],
    output_scale: float,
    output_zero_point: int,
    scale_factor: float,
) -> list[int]:
    if output_scale <= 0.0:
        raise RuntimeError("Invalid output quantization for RSQRT reference model")

    output: list[int] = []
    for raw in raw_outputs:
        dequantized = (int(raw) - output_zero_point) * output_scale
        rescaled = dequantized * scale_factor
        requantized = int(np.rint(rescaled / output_scale)) + output_zero_point
        output.append(int(np.clip(requantized, -32768, 32767)))
    return output


def arm_nn_doubling_high_mult_no_sat(value: int, multiplier: int) -> int:
    total = int(value) * int(multiplier)
    nudge = (1 << 30) if total >= 0 else (1 - (1 << 30))
    return int((total + nudge) >> 31)


def arm_nn_divide_by_power_of_two(value: int, exponent: int) -> int:
    if exponent <= 0:
        return int(value)
    remainder_mask = (1 << exponent) - 1
    remainder = value & remainder_mask
    threshold = remainder_mask >> 1
    if value < 0:
        threshold += 1
    return int((value >> exponent) + (1 if remainder > threshold else 0))


def arm_nn_requantize(value: int, multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    shifted = int(value) * (1 << left_shift)
    return arm_nn_divide_by_power_of_two(
        arm_nn_doubling_high_mult_no_sat(shifted, multiplier),
        right_shift,
    )


def arm_rsqrt_s16_round_div2_nearest(value: int) -> int:
    if value >= 0:
        return (value + 1) >> 1
    return -(((-value) + 1) >> 1)


def arm_rsqrt_s16_scaled_positive(
    lut: Sequence[int],
    q_value: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
) -> int:
    lut_index = (q_value + ((1 << BASE_STEP_SHIFT) - 1)) >> BASE_STEP_SHIFT
    lut_index = min(lut_index, TEST_LUT_SIZE - 1)

    scaled = int(lut[lut_index])
    if needs_rescale:
        scaled = arm_nn_requantize(scaled, out_mult, out_shift)
    return int(np.clip(scaled, -32768, 32767))


def arm_rsqrt_s16_sample_slot(
    lut: Sequence[int],
    slot_index: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
) -> int:
    q0 = -32768 + (slot_index << SLOT_SHIFT)
    qmid = q0 + (1 << BASE_STEP_SHIFT)
    q1 = q0 + (1 << SLOT_SHIFT)

    s0 = 32767 if q0 <= 0 else arm_rsqrt_s16_scaled_positive(lut, q0, out_mult, out_shift, needs_rescale)
    sm = 32767 if qmid <= 0 else arm_rsqrt_s16_scaled_positive(lut, qmid, out_mult, out_shift, needs_rescale)
    s1 = 32767 if q1 <= 0 else arm_rsqrt_s16_scaled_positive(lut, q1, out_mult, out_shift, needs_rescale)

    midpoint_interp = (s0 + s1 + 1) >> 1
    midpoint_err = midpoint_interp - sm
    bias = arm_rsqrt_s16_round_div2_nearest(midpoint_err)
    return int(np.clip(s0 - bias, -32768, 32767))


def simulate_universal_output(
    input_values: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_mult: int,
    out_shift: int,
    needs_rescale: bool,
    out_activation_min: int,
    out_activation_max: int,
    lut: Sequence[int],
) -> list[int]:
    output: list[int] = []
    for raw in input_values:
        value = raw - input_offset
        if value < 0:
            raise ValueError("Universal RSQRT simulation received a negative input")
        value = min(value, 0x7FFF)

        shifted_val = value - (-32768)
        idx = (shifted_val >> SLOT_SHIFT) & 0x1FF
        frac = shifted_val & ((1 << SLOT_SHIFT) - 1)
        y0 = arm_rsqrt_s16_sample_slot(lut, idx, out_mult, out_shift, needs_rescale)
        y1 = arm_rsqrt_s16_sample_slot(lut, idx + 1, out_mult, out_shift, needs_rescale)
        interp_delta = (((y1 - y0) * frac) + (1 << (SLOT_SHIFT - 1))) >> SLOT_SHIFT
        rsqrt_res = y0 + interp_delta + out_offset
        rsqrt_res = min(out_activation_max, max(out_activation_min, rsqrt_res))
        output.append(rsqrt_res)
    return output


def simulate_per_op_output(
    input_values: Sequence[int],
    input_offset: int,
    out_offset: int,
    out_activation_min: int,
    out_activation_max: int,
    lut: Sequence[int],
) -> list[int]:
    output: list[int] = []
    for raw in input_values:
        value = raw - input_offset
        if value < 0:
            raise ValueError("Per-op RSQRT simulation received a negative input")
        value = min(value, 0x7FFF)

        index = 256 + (value >> SLOT_SHIFT)
        frac = value & ((1 << SLOT_SHIFT) - 1)
        y0 = int(lut[index])
        y1 = int(lut[index + 1])
        rsqrt_res = y0 + (((y1 - y0) * frac + (1 << (SLOT_SHIFT - 1))) >> SLOT_SHIFT)
        rsqrt_res += out_offset
        rsqrt_res = min(out_activation_max, max(out_activation_min, rsqrt_res))
        output.append(rsqrt_res)
    return output


def calculate_tolerance(actual: Sequence[int], reference: Sequence[int]) -> int:
    max_error = max(abs(int(a) - int(b)) for a, b in zip(actual, reference))
    return max_error


def format_scalar_macro(name: str, value: int) -> str:
    return f"#define {name} ({value})"


def format_array(name: str, c_type: str, values: Sequence[int], values_per_line: int = 8) -> str:
    lines = [f"static const {c_type} {name}[{len(values)}] = {{"]
    for start in range(0, len(values), values_per_line):
        chunk = values[start:start + values_per_line]
        lines.append("    " + ", ".join(str(value) for value in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_header(path: Path, lines: Iterable[str]) -> None:
    header = Lib.test.get_header("keras", "tensorflow")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def generate_headers(output_dir: Path) -> None:
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

    logical_base_values = input_values
    logical_offset_values = [value - OFFSET_INPUT_OFFSET for value in offset_input_values]
    logical_rescale_values = [value - RESCALE_INPUT_OFFSET for value in rescale_input_values]

    universal_lut = fill_universal_lut()
    per_op_lut = fill_per_op_lut()

    base_model_path = output_dir / "rsqrt_s16_reference.tflite"
    rescale_factor = math.ldexp(RESCALE_OUT_MULT / float(1 << 31), RESCALE_OUT_SHIFT)

    universal_output, _, _ = run_tflite_reference(logical_base_values, 1.0, base_model_path)
    per_op_output = universal_output.copy()
    offset_per_op_output, _, _ = run_tflite_reference(logical_offset_values, 1.0, base_model_path)
    rescale_base_output, rescale_output_scale, rescale_output_zero_point = run_tflite_reference(
        logical_rescale_values,
        1.0,
        base_model_path,
    )
    rescaled_universal_output = apply_float_rescale_reference(
        rescale_base_output,
        rescale_output_scale,
        rescale_output_zero_point,
        rescale_factor,
    )

    universal_simulated = simulate_universal_output(
        input_values,
        INPUT_OFFSET,
        OUT_OFFSET,
        OUT_MULT,
        OUT_SHIFT,
        bool(NEEDS_RESCALE),
        OUT_ACTIVATION_MIN,
        OUT_ACTIVATION_MAX,
        universal_lut,
    )
    per_op_simulated = simulate_per_op_output(
        input_values,
        INPUT_OFFSET,
        OUT_OFFSET,
        OUT_ACTIVATION_MIN,
        OUT_ACTIVATION_MAX,
        per_op_lut,
    )
    offset_per_op_simulated = simulate_per_op_output(
        offset_input_values,
        OFFSET_INPUT_OFFSET,
        OFFSET_OUT_OFFSET,
        OFFSET_OUT_ACTIVATION_MIN,
        OFFSET_OUT_ACTIVATION_MAX,
        per_op_lut,
    )
    rescaled_universal_simulated = simulate_universal_output(
        rescale_input_values,
        RESCALE_INPUT_OFFSET,
        RESCALE_OUT_OFFSET,
        RESCALE_OUT_MULT,
        RESCALE_OUT_SHIFT,
        bool(RESCALE_NEEDS_RESCALE),
        RESCALE_OUT_ACTIVATION_MIN,
        RESCALE_OUT_ACTIVATION_MAX,
        universal_lut,
    )

    universal_tolerance = calculate_tolerance(universal_simulated, universal_output)
    per_op_tolerance = calculate_tolerance(per_op_simulated, per_op_output)
    offset_per_op_tolerance = calculate_tolerance(offset_per_op_simulated, offset_per_op_output)
    rescaled_universal_tolerance = calculate_tolerance(rescaled_universal_simulated, rescaled_universal_output)

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
            format_scalar_macro("RSQRT_S16_UNIVERSAL_TOLERANCE", universal_tolerance),
            format_scalar_macro("RSQRT_S16_PER_OP_TOLERANCE", per_op_tolerance),
            format_scalar_macro("RSQRT_S16_OFFSET_BLOCK_SIZE", len(offset_input_values)),
            format_scalar_macro("RSQRT_S16_OFFSET_INPUT_OFFSET", OFFSET_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_OFFSET", OFFSET_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MIN", OFFSET_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_OFFSET_OUT_ACTIVATION_MAX", OFFSET_OUT_ACTIVATION_MAX),
            format_scalar_macro("RSQRT_S16_OFFSET_PER_OP_TOLERANCE", offset_per_op_tolerance),
            format_scalar_macro("RSQRT_S16_RESCALE_BLOCK_SIZE", len(rescale_input_values)),
            format_scalar_macro("RSQRT_S16_RESCALE_INPUT_OFFSET", RESCALE_INPUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_OFFSET", RESCALE_OUT_OFFSET),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_MULT", RESCALE_OUT_MULT),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_SHIFT", RESCALE_OUT_SHIFT),
            format_scalar_macro("RSQRT_S16_RESCALE_NEEDS_RESCALE", RESCALE_NEEDS_RESCALE),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MIN", RESCALE_OUT_ACTIVATION_MIN),
            format_scalar_macro("RSQRT_S16_RESCALE_OUT_ACTIVATION_MAX", RESCALE_OUT_ACTIVATION_MAX),
            format_scalar_macro("RSQRT_S16_RESCALED_UNIVERSAL_TOLERANCE", rescaled_universal_tolerance),
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
    write_header(
        output_dir / "rescaled_universal_output.h",
        [format_array("rsqrt_s16_rescaled_universal_output", "int16_t", rescaled_universal_output)],
    )
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

    for tflite_path in [base_model_path]:
        if tflite_path.exists():
            tflite_path.unlink()


def parse_args() -> argparse.Namespace:
    default_output = THIS_DIR.parent / "TestCases" / "TestData" / "rsqrt_s16"
    parser = argparse.ArgumentParser(description="Generate RSQRT s16 unit-test data headers.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where generated headers are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_headers(args.output_dir)
    print(f"Generated RSQRT s16 test data in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
