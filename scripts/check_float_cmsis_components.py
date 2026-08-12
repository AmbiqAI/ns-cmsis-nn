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
#      Ambiq::NS-CMSIS-NN@<version> exactly once, with no `path:`
#      override and no range specifier, and that version matches the
#      pdsc <release version>. (The local pack is registered once via
#      cpackget in CI, not duplicated as a path pack.)
#   2. Component selector — every float .cproject.yml the csolution
#      builds references the fully-qualified
#      `Ambiq::Machine Learning:NN Lib:heliaCORE&Source` selector, never
#      a bare `Machine Learning:NN Lib:heliaCORE`, and every float
#      cproject on disk is registered in the csolution.
#   3. Device-component gating — the SSE-300-only device components in
#      corstone300_unittest.clayer.yml are all present and each
#      restricted to `for-context: +Corstone-300-FVP`, so generic
#      ARMCM0/ARMCM4 targets do not try to resolve them.
#
# These files are parsed as YAML rather than scanned line by line.
# Comments, quoting and key order are the toolchain's business; an
# earlier text-matching version of this script silently passed real
# regressions that happened to be spelled differently (a trailing
# comment on a `- component:` line was enough to skip its gating check
# entirely), and failed valid files that interleaved other keys.

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - environment problem, not a finding
    raise SystemExit(
        "FAIL: check_float_cmsis_components.py requires PyYAML.\n"
        "      Install it with: pip install pyyaml"
    )

REPO = Path(__file__).resolve().parents[1]
PDSC = REPO / "Ambiq.NS-CMSIS-NN.pdsc"
CMSIS_DIR = REPO / "Tests" / "UnitTest" / "cmsis"
CSOLUTION = CMSIS_DIR / "cmsis_nn_unit_tests_flt.csolution.yml"
CLAYER = CMSIS_DIR / "corstone300_unittest.clayer.yml"

PACK_NAME = "Ambiq::NS-CMSIS-NN"
EXPECTED_SELECTOR = "Ambiq::Machine Learning:NN Lib:heliaCORE&Source"
HELIA_CORE = re.compile(r"Machine Learning:NN Lib:heliaCORE")
# csolution accepts @>=x.y.z / @^x.y.z / @~x.y.z, but the local pack is
# registered from the working-tree pdsc, so only an exact pin makes a
# stale csolution fail loudly instead of resolving to some other
# NS-CMSIS-NN in the developer's pack root.
RANGE_SPECIFIER = re.compile(r"^[<>=^~]")

# Device components only available when device == SSE-300-MPS3; generic
# ARMCM0/ARMCM4 devices do not provide them.
SSE300_ONLY_COMPONENTS = (
    "ARM::Device:Definition",
    "ARM::Device:Native Driver:Timeout",
    "ARM::Device:Native Driver:SysCounter",
    "ARM::Device:Native Driver:SysTimer",
)
SSE300_CONTEXT = "+Corstone-300-FVP"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Parse `path` as a YAML mapping, recording a readable error instead
    of raising so one broken file does not mask the other checks."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{rel(path)}: cannot be read ({exc.strerror or exc})")
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        errors.append(f"{rel(path)}: is not valid YAML ({exc})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{rel(path)}: expected a mapping at the top level")
        return None
    return data


def section(data: dict[str, Any], key: str, path: Path, errors: list[str]):
    node = data.get(key)
    if not isinstance(node, dict):
        errors.append(f"{rel(path)}: missing top-level `{key}:` mapping")
        return None
    return node


def components_of(node: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return (name, entry) for each `- component:` in node's list.

    An entry may be a bare string or a mapping carrying extra keys such
    as `define:` or `for-context:` in any order.
    """
    raw = node.get("components")
    if not isinstance(raw, list):
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for entry in raw:
        if isinstance(entry, str):
            found.append((entry.strip(), {}))
        elif isinstance(entry, dict) and isinstance(entry.get("component"), str):
            found.append((entry["component"].strip(), entry))
    return found


def pdsc_release_version(errors: list[str]) -> str | None:
    try:
        text = PDSC.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{rel(PDSC)}: cannot be read ({exc.strerror or exc})")
        return None
    match = re.search(r'<release\s+version="([^"]+)"', text)
    if not match:
        errors.append(f"{rel(PDSC)}: no <release version> found")
        return None
    return match.group(1)


def check_pack_source(csolution: dict[str, Any], errors: list[str]) -> None:
    solution = section(csolution, "solution", CSOLUTION, errors)
    if solution is None:
        return
    packs = solution.get("packs")
    if not isinstance(packs, list):
        errors.append(f"{CSOLUTION.name}: missing `solution.packs:` list")
        return

    entries: list[tuple[str, dict[str, Any]]] = []
    for entry in packs:
        if isinstance(entry, str):
            spec, mapping = entry, {}
        elif isinstance(entry, dict) and isinstance(entry.get("pack"), str):
            spec, mapping = entry["pack"], entry
        else:
            continue
        if spec.split("@", 1)[0].strip() == PACK_NAME:
            entries.append((spec.strip(), mapping))

    if not entries:
        errors.append(f"{CSOLUTION.name}: no {PACK_NAME} pack entry found")
        return
    if len(entries) > 1:
        errors.append(
            f"{CSOLUTION.name}: {PACK_NAME} is declared {len(entries)} times; "
            f"exactly one entry may resolve the cpackget .Local registration"
        )

    expected = pdsc_release_version(errors)
    for spec, mapping in entries:
        # A `path:` override re-introduces a duplicate pack definition
        # that competes with the cpackget .Local registration.
        if "path" in mapping:
            errors.append(
                f"{CSOLUTION.name}: {PACK_NAME} must not use a `path:` "
                f"override (single source = cpackget .Local registration)"
            )
        _, at, version = spec.partition("@")
        version = version.strip()
        if not at or not version:
            errors.append(
                f"{CSOLUTION.name}: {PACK_NAME} must be pinned as "
                f"{PACK_NAME}@<version> (single cpackget-registered source)"
            )
        elif RANGE_SPECIFIER.match(version):
            errors.append(
                f"{CSOLUTION.name}: {PACK_NAME} must use an exact pin, not "
                f"the range specifier '@{version}'"
            )
        elif expected is not None and version != expected:
            errors.append(
                f"{CSOLUTION.name}: pack pinned at @{version} but pdsc "
                f"<release version> is {expected}"
            )


def csolution_projects(
    csolution: dict[str, Any], errors: list[str]
) -> list[tuple[str, Path]]:
    """Resolve `solution.projects:` to (as-written, absolute path) pairs."""
    solution = section(csolution, "solution", CSOLUTION, errors)
    if solution is None:
        return []
    projects = solution.get("projects")
    if not isinstance(projects, list) or not projects:
        errors.append(f"{CSOLUTION.name}: missing or empty `solution.projects:` list")
        return []

    resolved: list[tuple[str, Path]] = []
    for entry in projects:
        if isinstance(entry, str):
            spec = entry
        elif isinstance(entry, dict) and isinstance(entry.get("project"), str):
            spec = entry["project"]
        else:
            continue
        resolved.append((spec, (CMSIS_DIR / spec).resolve()))
    return resolved


def check_component_selector(csolution: dict[str, Any], errors: list[str]) -> None:
    # Driven by the csolution's own project list rather than a glob, so a
    # project cbuild actually builds can never be skipped by this check.
    for spec, cproject in csolution_projects(csolution, errors):
        if not cproject.is_file():
            errors.append(
                f"{CSOLUTION.name}: project '{spec}' does not exist on disk"
            )
            continue
        data = load_yaml(cproject, errors)
        if data is None:
            continue
        project = section(data, "project", cproject, errors)
        if project is None:
            continue

        names = [name for name, _ in components_of(project)]
        if EXPECTED_SELECTOR in names:
            continue
        # Flag a bare/under-qualified heliaCORE reference.
        bare = [name for name in names if HELIA_CORE.search(name)]
        if bare:
            errors.append(
                f"{rel(cproject)}: heliaCORE component is declared as "
                f"'{bare[0]}', not the fully-qualified selector "
                f"'{EXPECTED_SELECTOR}'"
            )
        else:
            errors.append(
                f"{rel(cproject)}: missing component selector "
                f"'{EXPECTED_SELECTOR}'"
            )


def check_project_registration(csolution: dict[str, Any], errors: list[str]) -> None:
    """A float cproject on disk but absent from the csolution is never
    built, so its selector is never validated by the check above."""
    registered = {path for _, path in csolution_projects(csolution, errors)}
    on_disk = sorted(CMSIS_DIR.glob("test_arm_*_flt/*.cproject.yml"))
    if not on_disk:
        errors.append("no float .cproject.yml files found under Tests/UnitTest/cmsis")
        return
    for cproject in on_disk:
        if cproject.resolve() not in registered:
            errors.append(
                f"{rel(cproject)}: not registered in {CSOLUTION.name} "
                f"`solution.projects:`"
            )


def check_device_gating(errors: list[str]) -> None:
    data = load_yaml(CLAYER, errors)
    if data is None:
        return
    layer = section(data, "layer", CLAYER, errors)
    if layer is None:
        return

    declared = dict(components_of(layer))
    for component in SSE300_ONLY_COMPONENTS:
        entry = declared.get(component)
        if entry is None:
            errors.append(
                f"{CLAYER.name}: SSE-300-only component '{component}' is not "
                f"declared, so its '{SSE300_CONTEXT}' gating cannot be enforced"
            )
            continue
        context = entry.get("for-context")
        contexts = context if isinstance(context, list) else [context]
        if SSE300_CONTEXT not in [c for c in contexts if isinstance(c, str)]:
            errors.append(
                f"{CLAYER.name}: component '{component}' must be gated with "
                f"'for-context: {SSE300_CONTEXT}' (found: {context!r})"
            )


def main() -> int:
    errors: list[str] = []

    csolution = load_yaml(CSOLUTION, errors)
    if csolution is not None:
        check_pack_source(csolution, errors)
        check_component_selector(csolution, errors)
        check_project_registration(csolution, errors)
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
