#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Compile real target instructions without a target libc or board. Refs #835."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", required=True, help="Clang compiler")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = args.output / "fp_control.c"
    source.write_text('''#include "Internal/arm_nn_fp_control.h"
#if EXPECT_FP_CONTROL
#ifndef ARM_NN_FP_CONTROL_AVAILABLE
#error Missing floating-point control access
#endif
void probe(void)
{
    const arm_nn_fp_control_t saved = arm_nn_get_fp_control();
    arm_nn_set_fp_control(saved);
}
#elif defined(ARM_NN_FP_CONTROL_AVAILABLE)
#error Floating-point control access selected without supported hardware
#endif
''')
    cases = [
        ("aarch64", ["--target=aarch64-none-elf"], 1),
        ("m55", ["--target=arm-none-eabi", "-mcpu=cortex-m55", "-mfloat-abi=hard"], 1),
        ("m4", ["--target=arm-none-eabi", "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"], 1),
        ("m0", ["--target=arm-none-eabi", "-mcpu=cortex-m0", "-mfloat-abi=soft"], 0),
    ]
    for name, target, available in cases:
        subprocess.run([
            args.cc, *target, "-ffreestanding", "-O2", "-Werror",
            "-I" + str(root / "Include"), f"-DEXPECT_FP_CONTROL={available}",
            "-c", str(source), "-o", str(args.output / (name + ".o")),
        ], check=True)
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
