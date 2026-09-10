#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for the refactored RSQRT s16 test-data generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent.parent / "TestData" / "rsqrt_s16"
    parser = argparse.ArgumentParser(description="Generate RSQRT s16 unit-test data headers.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory where generated headers are written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve().parents[2] / "RefactoredTestGen" / "generate_rsqrt_s16_test_data.py"
    subprocess.run([sys.executable, str(script_path), "--output-dir", str(args.output_dir)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
