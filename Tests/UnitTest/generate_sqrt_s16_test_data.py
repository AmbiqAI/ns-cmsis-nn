#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited or its affiliates
#
# SPDX-License-Identifier: Apache-2.0
#
"""Generate test data for arm_sqrt_s16 unit tests.

This script writes test vectors to:
  Tests/UnitTest/TestCases/TestData/sqrt_*_s16/

It mirrors the integer sqrt + requantize logic used in arm_sqrt_s16.
"""

import os
import math
import random

OUT_ROOT = os.path.join("Tests", "UnitTest", "TestCases", "TestData")


def arm_nn_isqrt_u32(x: int) -> int:
    res = 0
    bit = 1 << 30
    while bit > x:
        bit >>= 2
    while bit != 0:
        if x >= res + bit:
            x -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def arm_nn_doubling_high_mult_no_sat(m1: int, m2: int) -> int:
    mult = (1 << 30) + (m1 * m2)
    return int(mult >> 31)


def arm_nn_divide_by_power_of_two(dividend: int, exponent: int) -> int:
    if exponent == 0:
        return dividend
    remainder_mask = (1 << exponent) - 1
    remainder = remainder_mask & dividend
    result = dividend >> exponent
    threshold = remainder_mask >> 1
    if result < 0:
        threshold += 1
    if remainder > threshold:
        result += 1
    return result


def arm_nn_requantize(val: int, multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    shifted = val * (1 << left_shift)
    return arm_nn_divide_by_power_of_two(arm_nn_doubling_high_mult_no_sat(shifted, multiplier), right_shift)


def compute_output(inputs, input_offset, out_offset, out_mult, out_shift, needs_rescale, act_min, act_max):
    outputs = []
    for x in inputs:
        val = int(x) - int(input_offset)
        if val < 0:
            val = 0
        if val > 0x7FFF:
            val = 0x7FFF
        # Treat input as Q15, compute sqrt in Q15.
        val_q15 = val << 15
        sqrt_res = arm_nn_isqrt_u32(val_q15)
        if needs_rescale:
            acc = arm_nn_requantize(sqrt_res, out_mult, out_shift)
        else:
            acc = int(sqrt_res)
        acc += out_offset
        if acc < act_min:
            acc = act_min
        if acc > act_max:
            acc = act_max
        outputs.append(int(acc))
    return outputs


def write_header(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def format_array(name: str, arr, ctype: str = "int16_t") -> str:
    lines = [f"const {ctype} {name}[{len(arr)}] = {{"]
    for i in range(0, len(arr), 8):
        chunk = ", ".join(str(v) for v in arr[i:i + 8])
        if i + 8 < len(arr):
            lines.append(f"  {chunk},")
        else:
            lines.append(f"  {chunk}")
    lines.append("};")
    return "\n".join(lines)


def quantize_scale(scale: float):
    if scale == 0.0:
        return 0, 0
    significand, shift = math.frexp(scale)
    significand_q31 = int(round(significand * (1 << 31)))
    return significand_q31, shift


def generate_float_dataset(name: str, length: int):
    random.seed(0xC0FFEE)

    # Keep fixed scales (si/so) and bias inputs toward small ~0.001 values.
    si = 0.001029206763651379
    so = 0.00017722826063936592

    small_count = int(length * 0.8)
    large_count = length - small_count

    small_vals = [abs(random.gauss(0.0, 1.0) * 0.0015) for _ in range(small_count)]
    large_vals = [abs(random.gauss(0.0, 1.0) * 10.0) for _ in range(large_count)]
    floats = small_vals + large_vals

    qi = [min(32767, max(0, int(round(v / si)))) for v in floats]
    if qi:
        qi[-1] = 32767  # Ensure full-scale coverage for fixed si/so.

    # Kernel sqrt_res is in Q15-like units: sqrt_res ~= sqrt(qi) * 2^7.5
    # So scale must include a 2^-7.5 factor to map to real output.
    scale = math.sqrt(si) / (so * (2.0 ** 7.5))
    mult, shift = quantize_scale(scale)

    return qi, mult, shift, si, so


def generate():
    datasets = [
        {
            "name": "sqrt_small_tensor_s16",
            "dims": {"BATCH_SIZE": 1, "HEIGHT_1": 2, "WIDTH_1": 3, "CHANNEL_1": 4},
            "input_offset": 0,
            "out_offset": 0,
            "needs_rescale": True,
            "out_mult": 2147483647,
            "out_shift": 0,
            "act_min": -32768,
            "act_max": 32767,
            "inputs": [
                -20000, -100, 0, 1, 10, 100, 500, 1000,
                2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000,
                30000, 31000, 32000, 32767, -32768, -1, 12345, 23456,
            ],
        },
        {
            "name": "sqrt_long_row_s16",
            "dims": {"BATCH_SIZE": 1, "HEIGHT_1": 1, "WIDTH_1": 1024, "CHANNEL_1": 1},
            "input_offset": -100,
            "out_offset": 0,
            "needs_rescale": True,
            "out_mult": 2147483647,
            "out_shift": 0,
            "act_min": -32768,
            "act_max": 32767,
            "inputs": [
                -30000, -20000, -10000, -5000, -1000, -100, 0, 50,
                100, 200, 400, 800, 1200, 1600, 2000, 2400,
                2800, 3200, 3600, 4000, 8000, 12000, 16000, 20000,
                22000, 24000, 26000, 28000, 30000, 31000, 32000, 32600,
            ],
        },
        {
            "name": "sqrt_multi_batch_s16",
            "dims": {"BATCH_SIZE": 2, "HEIGHT_1": 2, "WIDTH_1": 3, "CHANNEL_1": 4},
            "input_offset": 0,
            "out_offset": 0,
            "needs_rescale": True,
            "out_mult": 2147483647,
            "out_shift": 0,
            "act_min": -32768,
            "act_max": 32767,
            "inputs": [
                -300, -200, -100, -10, 0, 2, 4, 8,
                16, 32, 64, 128, 256, 512, 1024, 2048,
                4096, 8192, 10000, 12000, 14000, 16000, 18000, 20000,
                22000, 24000, 26000, 28000, 30000, 31000, 32000, 32767,
                123, 456, 789, 1357, 2468, 3579, 4680, 5791,
                100, 200, 300, 400, 500, 600, 700, 800,
            ],
        },
        {
            "name": "sqrt_float_tensor_s16",
            "dims": {"BATCH_SIZE": 1, "HEIGHT_1": 1, "WIDTH_1": 1024, "CHANNEL_1": 1},
            "input_offset": 0,
            "out_offset": 0,
            "needs_rescale": True,
            "out_mult": None,  # Filled in below
            "out_shift": None,  # Filled in below
            "act_min": -32768,
            "act_max": 32767,
            "inputs": None,  # Filled in below
            "input_scale": None,  # Filled in below
            "output_scale": None,  # Filled in below
        },
    ]

    header = "// Generated by generate_sqrt_s16_test_data.py\n"
    header += "// Deterministic integer-sqrt reference for arm_sqrt_s16.\n"

    for ds in datasets:
        name = ds["name"]
        if name == "sqrt_float_tensor_s16":
            qi, mult, shift, si, so = generate_float_dataset(name, 1024)
            ds["inputs"] = qi
            ds["out_mult"] = mult
            ds["out_shift"] = shift
            ds["input_scale"] = si
            ds["output_scale"] = so

        inputs = ds["inputs"]
        out = compute_output(
            inputs,
            ds["input_offset"],
            ds["out_offset"],
            ds["out_mult"],
            ds["out_shift"],
            ds["needs_rescale"],
            ds["act_min"],
            ds["act_max"],
        )
        block_size = len(inputs)

        dir_path = os.path.join(OUT_ROOT, name)
        os.makedirs(dir_path, exist_ok=True)

        prefix = name.upper()
        cfg_lines = [header, "#pragma once", ""]
        for k, v in ds["dims"].items():
            cfg_lines.append(f"#define {prefix}_{k} {v}")
        cfg_lines.append(f"#define {prefix}_INPUT_OFFSET {ds['input_offset']}")
        cfg_lines.append(f"#define {prefix}_OUTPUT_OFFSET {ds['out_offset']}")
        cfg_lines.append(f"#define {prefix}_NEEDS_RESCALE {'true' if ds['needs_rescale'] else 'false'}")
        cfg_lines.append(f"#define {prefix}_OUTPUT_MULTIPLIER {ds['out_mult']}")
        cfg_lines.append(f"#define {prefix}_OUTPUT_SHIFT {ds['out_shift']}")
        cfg_lines.append(f"#define {prefix}_OUT_ACTIVATION_MIN {ds['act_min']}")
        cfg_lines.append(f"#define {prefix}_OUT_ACTIVATION_MAX {ds['act_max']}")
        if ds.get("input_scale") is not None:
            cfg_lines.append(f"#define {prefix}_INPUT_SCALE {ds['input_scale']}")
            cfg_lines.append(f"#define {prefix}_OUTPUT_SCALE {ds['output_scale']}")
        cfg_lines.append(f"#define {prefix}_OUTPUT_LEN {block_size}")
        cfg_lines.append(f"#define {prefix}_BLOCK_SIZE {block_size}")
        cfg_content = "\n".join(cfg_lines) + "\n"
        write_header(os.path.join(dir_path, "config_data.h"), cfg_content)

        in_content = header + "#pragma once\n#include <stdint.h>\n\n"
        in_content += format_array(f"{name}_input_tensor", inputs, "int16_t") + "\n"
        write_header(os.path.join(dir_path, "input_tensor.h"), in_content)

        out_content = header + "#pragma once\n#include <stdint.h>\n\n"
        out_content += format_array(f"{name}_output", out, "int16_t") + "\n"
        write_header(os.path.join(dir_path, "output.h"), out_content)

        td_content = header + "#include \"config_data.h\"\n#include \"input_tensor.h\"\n#include \"output.h\"\n"
        write_header(os.path.join(dir_path, "test_data.h"), td_content)

    print("Generated datasets:", ", ".join(d["name"] for d in datasets))


if __name__ == "__main__":
    generate()
