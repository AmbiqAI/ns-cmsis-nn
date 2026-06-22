#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Zephyr Kconfig contract test for zephyr/Kconfig and zephyr/module.yml.
#
# Runs in plain Python 3 (no Zephyr, no west, no SDK). Asserts the wiring
# bits that the CMake-level wiring test (cmake/tests/zephyr_wiring) cannot
# reach because they live in Kconfig itself:
#
#   - Every kernel-group `config NS_CMSIS_NN_<KNOB>` referenced by
#     zephyr/CMakeLists.txt actually exists in zephyr/Kconfig.
#   - NS_CMSIS_NN_ALL `imply`s every renamed group knob (no group forgotten).
#   - Every deprecated alias config:
#       * lives under the [DEPRECATED] section,
#       * `select`s the matching renamed config,
#       * `select`s DEPRECATED.
#   - module.yml points at zephyr/CMakeLists.txt + zephyr/Kconfig.
#

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KCONFIG = REPO / "zephyr" / "Kconfig"
ZCML = REPO / "zephyr" / "CMakeLists.txt"
MODULE_YML = REPO / "zephyr" / "module.yml"

# (deprecated_name, new_name) pairs. If the rename in #160 ever grows or
# shrinks, update this list (and the matching configs in zephyr/Kconfig).
DEPRECATED_ALIASES = [
    ("BASICMATH", "BASICMATHSNN"),
    ("STRIDED_SLICE", "STRIDEDSLICE"),
    ("SVD", "SVDF"),
]

# Renamed/final group knobs. Must match the right-hand side of the
# translation table in zephyr/CMakeLists.txt (verified by the wiring test).
RENAMED_KNOBS = [
    "ACTIVATION",
    "BASICMATHSNN",
    "COMPARISON",
    "CONCATENATION",
    "CONVOLUTION",
    "FULLYCONNECTED",
    "GATHER",
    "LSTM",
    "NNSUPPORT",
    "PAD",
    "POOLING",
    "QUANTIZATION",
    "RESHAPE",
    "SOFTMAX",
    "STRIDEDSLICE",
    "SVDF",
    "TRANSPOSE",
    "TILE",
    "BROADCAST",
    "SCATTER",
    "SELECT",
    "REVERSESEQUENCE",
    "DYNAMICUPDATESLICE",
]

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def parse_configs(text: str) -> dict[str, str]:
    """Map each `config FOO` to the block of lines that follow until the
    next top-level keyword (config/menuconfig/comment/endif)."""
    blocks: dict[str, str] = {}
    cur_name: str | None = None
    cur: list[str] = []
    header_re = re.compile(r"^\s*(config|menuconfig)\s+(\S+)")
    end_re = re.compile(r"^\s*(config|menuconfig|comment|endif|endmenu)\b")
    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            if cur_name is not None:
                blocks[cur_name] = "\n".join(cur)
            cur_name = m.group(2)
            cur = [line]
        else:
            if cur_name is not None:
                if end_re.match(line) and not header_re.match(line):
                    blocks[cur_name] = "\n".join(cur)
                    cur_name = None
                    cur = []
                else:
                    cur.append(line)
    if cur_name is not None:
        blocks[cur_name] = "\n".join(cur)
    return blocks


def main() -> int:
    for p in (KCONFIG, ZCML, MODULE_YML):
        if not p.is_file():
            fail(f"missing file: {p.relative_to(REPO)}")
    if failures:
        report()
        return 1

    kctext = KCONFIG.read_text()
    blocks = parse_configs(kctext)

    # 1. Every renamed knob must have a config block.
    for knob in RENAMED_KNOBS:
        full = f"NS_CMSIS_NN_{knob}"
        if full not in blocks:
            fail(f"zephyr/Kconfig: missing `config {full}`")

    # 2. NS_CMSIS_NN_ALL must `imply` every renamed knob.
    all_block = blocks.get("NS_CMSIS_NN_ALL", "")
    if not all_block:
        fail("zephyr/Kconfig: missing `config NS_CMSIS_NN_ALL`")
    for knob in RENAMED_KNOBS:
        if re.search(rf"^\s*imply\s+NS_CMSIS_NN_{knob}\b", all_block, re.M) is None:
            fail(f"NS_CMSIS_NN_ALL does not `imply NS_CMSIS_NN_{knob}`")

    # 3. Deprecated aliases must select DEPRECATED + the renamed knob.
    for dep, new in DEPRECATED_ALIASES:
        full = f"NS_CMSIS_NN_{dep}"
        new_full = f"NS_CMSIS_NN_{new}"
        block = blocks.get(full)
        if block is None:
            fail(f"deprecated alias `config {full}` missing")
            continue
        if re.search(rf"^\s*select\s+{new_full}\b", block, re.M) is None:
            fail(f"`config {full}` does not `select {new_full}` (alias glue broken)")
        if re.search(r"^\s*select\s+DEPRECATED\b", block, re.M) is None:
            fail(f"`config {full}` does not `select DEPRECATED` (will not emit warning)")
        if "[DEPRECATED]" not in block:
            fail(f"`config {full}` prompt missing the [DEPRECATED] tag")

    # 4. module.yml must point at the right files.
    yml = MODULE_YML.read_text()
    if "cmake: zephyr" not in yml:
        fail("zephyr/module.yml does not set `cmake: zephyr`")
    if "kconfig: zephyr/Kconfig" not in yml:
        fail("zephyr/module.yml does not set `kconfig: zephyr/Kconfig`")

    report()
    return 1 if failures else 0


def report() -> None:
    if failures:
        print("Zephyr Kconfig contract FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            f"Zephyr Kconfig contract OK: {len(RENAMED_KNOBS)} renamed knobs, "
            f"{len(DEPRECATED_ALIASES)} deprecated aliases verified."
        )


if __name__ == "__main__":
    sys.exit(main())
