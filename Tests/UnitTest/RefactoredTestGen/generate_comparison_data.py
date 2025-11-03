#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited
# SPDX-License-Identifier: Apache-2.0
#
# This script generates deterministic reference data for the comparison
# operator unit-tests without relying on external Python packages.
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
print("IDDQD ", SCRIPT_DIR)
BASE_DIR = (SCRIPT_DIR / ".." / "TestCases" / "TestData").resolve()
print("IDDQD ", BASE_DIR)

def to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & (1 << 31):
        value -= 1 << 32
    return value


def arm_nn_doubling_high_mult_no_sat(m1: int, m2: int) -> int:
    m1 = to_int32(m1)
    m2 = to_int32(m2)
    mult = (1 << 30) + m1 * m2
    result = mult >> 31
    return to_int32(result)


def arm_nn_divide_by_power_of_two(dividend: int, exponent: int) -> int:
    if exponent == 0:
        return to_int32(dividend)

    dividend_int32 = to_int32(dividend)
    result = dividend_int32 >> exponent  # Python arithmetic shift
    remainder_mask = (1 << exponent) - 1
    remainder = (dividend_int32 & 0xFFFFFFFF) & remainder_mask

    threshold = remainder_mask >> 1
    if result < 0:
        threshold += 1
    if remainder > threshold:
        result += 1

    return to_int32(result)


def arm_nn_requantize(val: int, multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = 0 if shift > 0 else -shift

    val = to_int32(val)
    if left_shift:
        val = to_int32(val << left_shift)

    result = arm_nn_doubling_high_mult_no_sat(val, multiplier)
    if right_shift:
        result = arm_nn_divide_by_power_of_two(result, right_shift)

    return to_int32(result)


def prepare_value(value: int, offset: int, multiplier: int, shift: int, left_shift: int) -> int:
    acc = value + offset
    acc = to_int32(acc << left_shift) if left_shift else to_int32(acc)
    return arm_nn_requantize(acc, multiplier, shift)


def broadcast_index(idx: Tuple[int, int, int, int], dims: Tuple[int, int, int, int]) -> int:
    n, h, w, c = idx
    dn, dh, dw, dc = dims
    n = min(n, dn - 1)
    h = min(h, dh - 1)
    w = min(w, dw - 1)
    c = min(c, dc - 1)
    return ((n * dh + h) * dw + w) * dc + c


def iterate_indices(dims: Tuple[int, int, int, int]) -> Iterable[Tuple[int, int, int, int]]:
    for n in range(dims[0]):
        for h in range(dims[1]):
            for w in range(dims[2]):
                for c in range(dims[3]):
                    yield (n, h, w, c)


def generate_input_data(length: int, start: int, step: int) -> List[int]:
    return [start + step * i for i in range(length)]


def duplicate_with_offset(data: Sequence[int], offset_every: int, delta: int) -> List[int]:
    result = list(data)
    for i in range(0, len(result), offset_every):
        result[i] = to_int32(result[i] + delta)
    return result


def clamp_to_dtype(data: Sequence[int], dtype: str) -> List[int]:
    if dtype == "int8_t":
        min_val, max_val = -128, 127
    elif dtype == "int16_t":
        min_val, max_val = -32768, 32767
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return [max(min(x, max_val), min_val) for x in data]


def compute_output(dataset: Dict) -> List[int]:
    dims1 = dataset["input_1_dims"]
    dims2 = dataset["input_2_dims"]
    out_dims = tuple(max(a, b) for a, b in zip(dims1, dims2))
    params = dataset["params"]

    output = []
    for index in iterate_indices(out_dims):
        idx1 = broadcast_index(index, dims1)
        idx2 = broadcast_index(index, dims2)

        val1 = dataset["input_1_data"][idx1]
        val2 = dataset["input_2_data"][idx2]

        scaled1 = prepare_value(
            val1,
            params["input_1_offset"],
            params["input_1_mult"],
            params["input_1_shift"],
            params["left_shift"],
        )
        scaled2 = prepare_value(
            val2,
            params["input_2_offset"],
            params["input_2_mult"],
            params["input_2_shift"],
            params["left_shift"],
        )

        op = params["operation"]
        if op == "ARM_COMPARE_EQUAL":
            result = int(scaled1 == scaled2)
        elif op == "ARM_COMPARE_NOT_EQUAL":
            result = int(scaled1 != scaled2)
        elif op == "ARM_COMPARE_GREATER":
            result = int(scaled1 > scaled2)
        elif op == "ARM_COMPARE_GREATER_EQUAL":
            result = int(scaled1 >= scaled2)
        elif op == "ARM_COMPARE_LESS":
            result = int(scaled1 < scaled2)
        elif op == "ARM_COMPARE_LESS_EQUAL":
            result = int(scaled1 <= scaled2)
        else:
            raise ValueError(f"Unsupported operation '{op}'")

        output.append(result)

    dataset["output_dims"] = out_dims
    dataset["output_data"] = [bool(v) for v in output]
    dataset["params"]["dst_size"] = len(output)
    dataset["params"]["output_n"] = out_dims[0]
    dataset["params"]["output_h"] = out_dims[1]
    dataset["params"]["output_w"] = out_dims[2]
    dataset["params"]["output_c"] = out_dims[3]

    return output


def write_config(prefix: str, params: Dict[str, int], header: Path) -> None:
    header.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "// Generated by generate_comparison_data.py",
        "#pragma once",
        "",
    ]
    for key, value in params.items():
        macro = f"{prefix}_{key}".upper()
        lines.append(f"#define {macro} {value}")
    header.write_text("\n".join(lines) + "\n")


def format_array(dtype: str, name: str, data: Sequence[int]) -> str:
    elements_per_line = 8
    lines = [
        "// Generated by generate_comparison_data.py",
        "#pragma once",
        "#include <stdint.h>" if dtype != "bool" else "#include <stdbool.h>",
        "",
        f"const {dtype} {name}[{len(data)}] = {{",
    ]

    for i in range(len(data)):
        if i % elements_per_line == 0:
            lines.append("    ")
        value = data[i]
        if dtype == "bool":
            lines[-1] += "true" if value else "false"
        else:
            lines[-1] += f"{value}"
        if i != len(data) - 1:
            lines[-1] += ", "
        if (i + 1) % elements_per_line == 0 and i + 1 != len(data):
            lines.append("")

    if len(lines[-1]) == 0:
        lines.pop()
        lines.append("    ")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def write_array(prefix: str, tensor_name: str, dtype: str, data: Sequence[int], folder: Path, alias: str = None):
    header = folder / f"{tensor_name}.h"
    content = format_array(dtype, f"{prefix}_{tensor_name}", data)
    if alias:
        content += f"\nconst {dtype} *const {prefix}_{alias} = {prefix}_{tensor_name};\n"
    header.write_text(content)


def write_test_data(prefix: str, tensors: List[str], folder: Path) -> None:
    lines = [
        "// Generated by generate_comparison_data.py",
        '#include "config_data.h"',
    ]
    for tensor in tensors:
        lines.append(f'#include "{tensor}.h"')
    (folder / "test_data.h").write_text("\n".join(lines) + "\n")


def build_dataset(name: str,
                  dtype: str,
                  dims1: Tuple[int, int, int, int],
                  dims2: Tuple[int, int, int, int],
                  data1: List[int],
                  data2: List[int],
                  comparison_type: str) -> Dict:
    params = {
        "input_1_offset": 0,
        "input_1_mult": 1073741824,
        "input_1_shift": 0,
        "input_2_offset": 0,
        "input_2_mult": 1073741824,
        "input_2_shift": 0,
        "left_shift": 0,
        "output_offset": 0,
        "output_mult": 0,
        "output_shift": 0,
        "output_multiplier": 0,
        "asymmetric_quantize_inputs": 0,
        "operation": comparison_type,
    }

    data1 = clamp_to_dtype(data1, dtype)
    data2 = clamp_to_dtype(data2, dtype)

    return {
        "name": name,
        "dtype": dtype,
        "input_1_dims": dims1,
        "input_2_dims": dims2,
        "input_1_data": data1,
        "input_2_data": data2,
        "params": params,
    }


def create_datasets() -> List[Dict]:
    datasets: List[Dict] = []

    # Base patterns for int8
    base_equal = generate_input_data(24, -30, 3)
    base_scalar = generate_input_data(24, -18, 2)
    base_greater = generate_input_data(6, -12, 4)
    base_greater_equal = generate_input_data(24, -24, 3)
    base_less = generate_input_data(24, -16, 3)
    base_less_equal = generate_input_data(8, -10, 3)

    datasets.append(
        build_dataset(
            "comparison_equal_ident_s8",
            "int8_t",
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            base_equal,
            duplicate_with_offset(base_equal, 7, 2),
            "ARM_COMPARE_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_not_equal_scalar_s8",
            "int8_t",
            (1, 2, 3, 4),
            (1, 1, 1, 1),
            base_scalar,
            [5],
            "ARM_COMPARE_NOT_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_greater_broadcast_h_s8",
            "int8_t",
            (1, 1, 2, 3),
            (1, 4, 2, 3),
            base_greater,
            duplicate_with_offset(generate_input_data(24, -8, 3), 5, -3),
            "ARM_COMPARE_GREATER",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_greater_equal_broadcast_w_s8",
            "int8_t",
            (1, 2, 3, 4),
            (1, 2, 1, 4),
            base_greater_equal,
            duplicate_with_offset(generate_input_data(8, -12, 4), 3, 1),
            "ARM_COMPARE_GREATER_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_less_broadcast_c_s8",
            "int8_t",
            (1, 2, 3, 4),
            (1, 2, 3, 1),
            base_less,
            duplicate_with_offset(generate_input_data(6, -20, 4), 2, 6),
            "ARM_COMPARE_LESS",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_less_equal_broadcast_hw_s8",
            "int8_t",
            (1, 2, 1, 4),
            (1, 1, 3, 1),
            base_less_equal,
            duplicate_with_offset(generate_input_data(3, -7, 3), 1, -1),
            "ARM_COMPARE_LESS_EQUAL",
        )
    )

    # Int16 datasets reuse the int8 shapes but different value ranges.
    def scale_data(data: Sequence[int], scale: int) -> List[int]:
        return [x * scale for x in data]

    scaled_equal = scale_data(base_equal, 128)
    scaled_scalar = scale_data(base_scalar, 200)
    scaled_greater = scale_data(base_greater, 150)
    scaled_greater_equal = scale_data(base_greater_equal, 120)
    scaled_less = scale_data(base_less, 140)
    scaled_less_equal = scale_data(base_less_equal, 160)

    datasets.append(
        build_dataset(
            "comparison_equal_ident_s16",
            "int16_t",
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            scaled_equal,
            duplicate_with_offset(scaled_equal, 5, -256),
            "ARM_COMPARE_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_not_equal_scalar_s16",
            "int16_t",
            (1, 2, 3, 4),
            (1, 1, 1, 1),
            scaled_scalar,
            [1150],
            "ARM_COMPARE_NOT_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_greater_broadcast_h_s16",
            "int16_t",
            (1, 1, 2, 3),
            (1, 4, 2, 3),
            scaled_greater,
            duplicate_with_offset(scale_data(generate_input_data(24, -6, 2), 180), 4, -190),
            "ARM_COMPARE_GREATER",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_greater_equal_broadcast_w_s16",
            "int16_t",
            (1, 2, 3, 4),
            (1, 2, 1, 4),
            scaled_greater_equal,
            duplicate_with_offset(scale_data(generate_input_data(8, -10, 3), 200), 2, 120),
            "ARM_COMPARE_GREATER_EQUAL",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_less_broadcast_c_s16",
            "int16_t",
            (1, 2, 3, 4),
            (1, 2, 3, 1),
            scaled_less,
            duplicate_with_offset(scale_data(generate_input_data(6, -14, 3), 180), 2, 400),
            "ARM_COMPARE_LESS",
        )
    )

    datasets.append(
        build_dataset(
            "comparison_less_equal_broadcast_hw_s16",
            "int16_t",
            (1, 2, 1, 4),
            (1, 1, 3, 1),
            scaled_less_equal,
            duplicate_with_offset(scale_data(generate_input_data(3, -5, 2), 220), 1, -120),
            "ARM_COMPARE_LESS_EQUAL",
        )
    )

    return datasets


def generate():
    datasets = create_datasets()

    for dataset in datasets:
        prefix = dataset["name"]
        dtype = dataset["dtype"]
        folder = BASE_DIR / prefix
        compute_output(dataset)
        print(f"Generating test data for dataset '{prefix}'")

        params = dict(dataset["params"])
        dims1 = dataset["input_1_dims"]
        dims2 = dataset["input_2_dims"]

        params["input_1_n"], params["input_1_h"], params["input_1_w"], params["input_1_c"] = dims1
        params["input_2_n"], params["input_2_h"], params["input_2_w"], params["input_2_c"] = dims2

        write_config(prefix, params, folder / "config_data.h")

        write_array(prefix, "input_1_input_tensor", dtype, dataset["input_1_data"], folder)
        write_array(prefix, "input_2_input_tensor", dtype, dataset["input_2_data"], folder)
        write_array(prefix, "output", "bool", dataset["output_data"], folder, alias="output_ref")

        write_test_data(prefix, ["input_1_input_tensor", "input_2_input_tensor", "output"], folder)


if __name__ == "__main__":
    generate()
