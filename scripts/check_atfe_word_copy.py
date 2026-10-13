#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Execute the eight M4 word-copy suites on QEMU's Cortex-M55 backend."""

import argparse
import re
import subprocess
import time
from pathlib import Path

SUITES = (
    "test_arm_convolve_s8", "test_arm_convolve_1_x_n_s8", "test_arm_grouped_convolve_s8",
    "test_arm_ds_cnn_s_s8", "test_arm_depthwise_conv_fast_s16", "test_arm_pad_s16",
    "test_arm_transpose_conv_s8", "test_arm_nn_copy_fill_words_s8",
)


def unity_pass(text):
    text = text.replace("\r", "").replace("\0", "").rstrip("\x04\n ")
    return bool(re.search(r"^[1-9][0-9]* Tests 0 Failures [0-9]+ Ignored[ \t]*\nOK\s*\Z", text, re.M))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(["cmake", "--build", str(args.build), "--target", *SUITES, "--parallel", "4"], check=True)
    logs = args.build / "qemu-logs"
    logs.mkdir(exist_ok=True)
    failed = []
    for suite in SUITES:
        uart = logs / f"{suite}.log"
        with uart.open("w") as out, (logs / f"{suite}.stderr").open("w") as err:
            proc = subprocess.Popen([
                "qemu-system-arm", "-M", "mps3-an547", "-cpu", "cortex-m55", "-nographic",
                "-monitor", "none", "-serial", "stdio", "-d", "int", "-kernel",
                str(args.build / "TestCases" / suite / f"{suite}.elf"),
            ], stdout=out, stderr=err)
            try:
                deadline = time.monotonic() + 60
                while proc.poll() is None and time.monotonic() < deadline:
                    if re.search(r"^(OK|FAIL)", uart.read_text(errors="replace"), re.M):
                        break
                    time.sleep(0.1)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        text = uart.read_text(errors="replace")
        print(text.replace("\0", "").replace("\r", ""))
        if not unity_pass(text):
            failed.append(suite)
    if failed:
        print("Word-copy execution failed: " + ", ".join(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
