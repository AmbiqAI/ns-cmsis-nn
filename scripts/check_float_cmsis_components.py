#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Fast static guard for the float CMSIS unit-test wiring.
#
# `cbuild` cannot run in every environment (no CMSIS-Toolbox, no FVP),
# so this pure-Python check pins the invariants that previously caused
# "component not found" / cascade failures on the Cortex-M float FVP
# matrix. Runs in <1s, no Open-CMSIS-Pack tooling required.
#
#   1. Single pack source — the float csolution pins
#      Ambiq::NS-CMSIS-NN@<version> with no `path:` override, and that
#      version matches the pdsc <release version>. (The local pack is
#      registered once via cpackget in CI, not duplicated as a path
#      pack.)
#   2. Component selector — every float .cproject.yml references the
#      fully-qualified `Ambiq::Machine Learning:NN Lib:heliaCORE&Source`
#      selector, never a bare `Machine Learning:NN Lib:heliaCORE`.
#   3. Device-component gating — the SSE-300-only device components in
#      corstone300_unittest.clayer.yml are each restricted to
#      `for-context: +Corstone-300-FVP`, so generic ARMCM0/ARMCM4
#      targets do not try to resolve them.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PDSC = REPO / "Ambiq.NS-CMSIS-NN.pdsc"
CMSIS_DIR = REPO / "Tests" / "UnitTest" / "cmsis"
CSOLUTION = CMSIS_DIR / "cmsis_nn_unit_tests_flt.csolution.yml"
CLAYER = CMSIS_DIR / "corstone300_unittest.clayer.yml"

EXPECTED_SELECTOR = "Ambiq::Machine Learning:NN Lib:heliaCORE&Source"
# Device components only available when device == SSE-300-MPS3; generic
# ARMCM0/ARMCM4 devices do not provide them.
SSE300_ONLY_COMPONENTS = (
    "ARM::Device:Definition",
    "ARM::Device:Native Driver:Timeout",
    "ARM::Device:Native Driver:SysCounter",
    "ARM::Device:Native Driver:SysTimer",
)
SSE300_CONTEXT = "+Corstone-300-FVP"


def pdsc_release_version() -> str:
    text = PDSC.read_text(encoding="utf-8")
    match = re.search(r'<release\s+version="([^"]+)"', text)
    if not match:
        raise SystemExit(f"FAIL: no <release version> found in {PDSC}")
    return match.group(1)


def check_pack_source(errors: list[str]) -> None:
    text = CSOLUTION.read_text(encoding="utf-8")
    match = re.search(r"-\s*pack:\s*Ambiq::NS-CMSIS-NN(@([^\s]+))?", text)
    if not match:
        errors.append(f"{CSOLUTION.name}: no Ambiq::NS-CMSIS-NN pack entry found")
        return

    version = match.group(2)
    if version is None:
        errors.append(
            f"{CSOLUTION.name}: Ambiq::NS-CMSIS-NN must be pinned as "
            f"Ambiq::NS-CMSIS-NN@<version> (single cpackget-registered source)"
        )
    else:
        expected = pdsc_release_version()
        if version != expected:
            errors.append(
                f"{CSOLUTION.name}: pack pinned at @{version} but pdsc "
                f"<release version> is {expected}"
            )

    # A `path:` override re-introduces a duplicate pack definition that
    # competes with the cpackget .Local registration.
    block = re.search(
        r"-\s*pack:\s*Ambiq::NS-CMSIS-NN[^\n]*\n(\s+path:\s*\S+)?", text
    )
    if block and block.group(1):
        errors.append(
            f"{CSOLUTION.name}: Ambiq::NS-CMSIS-NN must not use a `path:` "
            f"override (single source = cpackget .Local registration)"
        )


def check_component_selector(errors: list[str]) -> None:
    cprojects = sorted(CMSIS_DIR.glob("test_arm_*_flt/*.cproject.yml"))
    if not cprojects:
        errors.append("no float .cproject.yml files found under Tests/UnitTest/cmsis")
        return

    for cproject in cprojects:
        text = cproject.read_text(encoding="utf-8")
        if EXPECTED_SELECTOR in text:
            continue
        # Flag a bare/under-qualified heliaCORE reference.
        if re.search(r"Machine Learning:NN Lib:heliaCORE", text):
            errors.append(
                f"{cproject.relative_to(REPO)}: heliaCORE component is not the "
                f"fully-qualified selector '{EXPECTED_SELECTOR}'"
            )
        else:
            errors.append(
                f"{cproject.relative_to(REPO)}: missing component selector "
                f"'{EXPECTED_SELECTOR}'"
            )


def check_device_gating(errors: list[str]) -> None:
    lines = CLAYER.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        for component in SSE300_ONLY_COMPONENTS:
            if stripped == f"- component: {component}":
                following = lines[index + 1].strip() if index + 1 < len(lines) else ""
                if following != f"for-context: {SSE300_CONTEXT}":
                    errors.append(
                        f"{CLAYER.name}: component '{component}' must be gated "
                        f"with 'for-context: {SSE300_CONTEXT}'"
                    )


def main() -> int:
    errors: list[str] = []
    check_pack_source(errors)
    check_component_selector(errors)
    check_device_gating(errors)

    if errors:
        print("Float CMSIS component check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Float CMSIS component check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
