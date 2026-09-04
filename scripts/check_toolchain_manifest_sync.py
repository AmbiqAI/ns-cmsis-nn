#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Guard against toolchain version drift between the two independent manifests
# that pin armclang and ATfE.
#
# Why this exists:
#   This repository declares each proprietary/embedded compiler in two places
#   that had no relationship enforced between them, and they drifted:
#
#     ci/toolchains/armclang.json   armclang 6.23.32   <- SHIPPED
#     ci/toolchains/atfe.json       ATfE     19.1.5    <- SHIPPED
#     ci/tools/manifest.json        armclang 6.24.0    <- DEV CONTAINER
#     ci/tools/manifest.json        ATfE     20.1.0    <- DEV CONTAINER
#
#   ci/toolchains/*.json is read by scripts/fetch_toolchain.sh and drives
#   release.yml, staticlib-dryrun.yml and toolchain-matrix-strict-link.yml --
#   i.e. it selects the compilers the shipped static libraries are actually
#   built and strict-link smoke-tested with.
#
#   ci/tools/manifest.json is read by scripts/install_ci_tools.sh from
#   .devcontainer/Dockerfile and bakes the published CI image, which is what
#   helia-core-tester.yml, legacy-tester.yml and interactive development run
#   inside.
#
#   So a developer reproducing a customer-visible codegen or ABI question in
#   the container was reaching for a different compiler than the one that
#   produced the artifact, with nothing anywhere saying so. That is a
#   "cannot reproduce" class of defect, and the fix is worthless without a
#   gate, because the two files are edited by different workflows for
#   different reasons and will drift again the moment someone bumps one.
#
# What is checked:
#   1. Version equality. The version string in ci/tools/manifest.json must
#      equal the version string in the corresponding ci/toolchains/*.json.
#   2. sha256 equality. Equal versions pointing at different bytes is the
#      same defect wearing a disguise -- it would let "armclang 6.23.32" in
#      the container be a different build of 6.23.32 than the one that ships,
#      which is exactly the reproducibility hole this guard closes, and a
#      version-only check would wave it straight through.
#
#   Both files carry a sha256 that CI verifies on download, so pinning them
#   equal costs nothing and means the container and the release build are
#   provably the same archive, not merely the same version label.
#
# Deliberately NOT checked: url equality. Equal sha256 already proves equal
# bytes, so a mirror or a re-host is a legitimate difference that should not
# fail a merge gate.
#
# Modelled on check_header_placeholders.py / check_api_group_classification.py:
# pure Python, stdlib only, no build, sub-second, wired into pdsc.yml beside
# the other per-PR textual guards.

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS_MANIFEST = REPO / "ci" / "tools" / "manifest.json"
TOOLCHAIN_DIR = REPO / "ci" / "toolchains"

# (tool id in ci/tools/manifest.json, filename under ci/toolchains/, label).
#
# The ids differ between the two files for historical reasons -- the container
# manifest calls ATfE "llvm-embedded" (its pre-rename product name) while the
# release manifest calls it "atfe". That mismatch is precisely why nobody
# noticed the drift, so the mapping is spelled out here rather than inferred.
PAIRS = (
    ("armclang", "armclang.json", "Arm Compiler for Embedded (armclang)"),
    ("llvm-embedded", "atfe.json", "Arm Toolchain for Embedded (ATfE/clang)"),
)

# ci/toolchains/ entries that must NOT be pinned to the container, with the
# reason. Every file in ci/toolchains/ has to appear here or in PAIRS, so a
# manifest added later cannot sit unexamined: an unlisted file is a failure,
# not a silent skip.
UNPAIRED = {
    "arm-gnu-floor.json": (
        "the oldest Arm GNU release the float16 kernels may be built with, "
        "which is deliberately older than the container's arm-gnu -- holding "
        "the two equal would delete the second compiler this pin exists to "
        "provide. .github/workflows/unity-f16-exec-gcc-floor.yml is what "
        "keeps it honest, by failing unless it names the GCC_F16_FLOOR "
        "declared in toolchain-matrix-strict-link.yml "
        "(AmbiqAI/ns-cmsis-nn#427)"
    ),
}

# Where each side is consumed, quoted in failure messages so whoever trips
# this guard does not have to go trace the wiring to understand the stakes.
CONTAINER_ROLE = (
    "the published CI dev container (.devcontainer/Dockerfile via "
    "scripts/install_ci_tools.sh) -- what helia-core-tester.yml, "
    "legacy-tester.yml and interactive development run inside"
)
SHIPPED_ROLE = (
    "the shipped static libraries (scripts/fetch_toolchain.sh via "
    "release.yml, staticlib-dryrun.yml and toolchain-matrix-strict-link.yml)"
)

failures: list[str] = []
_stats: dict[str, int] = {}


def fail(msg: str) -> None:
    failures.append(msg)


def load_json(path: Path) -> dict | None:
    """Read a JSON object, recording a failure rather than raising if the
    file is missing or malformed. A guard that dies with a traceback on a
    renamed file is indistinguishable from a broken CI runner."""
    if not path.exists():
        fail(
            f"{_display(path)} does not exist, so the armclang/ATfE versions "
            "cannot be cross-checked. If a manifest was deliberately moved or "
            "renamed, update PAIRS/TOOLS_MANIFEST in "
            "scripts/check_toolchain_manifest_sync.py to match -- do not "
            "simply delete this check, or the container and the shipped "
            "compilers are free to drift apart again."
        )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{_display(path)} is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        fail(f"{_display(path)} does not contain a JSON object at the top level.")
        return None
    return data


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def container_entry(manifest: dict, tool_id: str, manifest_path: Path) -> dict | None:
    """The tools[] entry for `tool_id`, or None after recording a failure.

    A missing entry is loud on purpose: silently skipping an unmatched id is
    how this guard would rot into passing over nothing at all the first time
    somebody renames a tool.
    """
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        fail(f"{_display(manifest_path)} has no 'tools' array to search for {tool_id!r}.")
        return None
    matches = [t for t in tools if isinstance(t, dict) and t.get("id") == tool_id]
    if not matches:
        seen = sorted(t.get("id", "<no id>") for t in tools if isinstance(t, dict))
        fail(
            f"{_display(manifest_path)} has no tool with id {tool_id!r} "
            f"(saw {seen}). This check pairs it against "
            f"ci/toolchains/ to keep the container and the shipped compiler "
            "on the same version. If the tool was renamed, update PAIRS in "
            "scripts/check_toolchain_manifest_sync.py."
        )
        return None
    if len(matches) > 1:
        fail(f"{_display(manifest_path)} declares tool id {tool_id!r} more than once.")
        return None
    return matches[0]


def check_pair(
    manifest: dict,
    manifest_path: Path,
    toolchain_dir: Path,
    tool_id: str,
    filename: str,
    label: str,
) -> None:
    entry = container_entry(manifest, tool_id, manifest_path)
    toolchain_path = toolchain_dir / filename
    shipped = load_json(toolchain_path)
    if entry is None or shipped is None:
        return

    container_version = entry.get("version")
    shipped_version = shipped.get("version")

    if not isinstance(shipped_version, str) or not shipped_version:
        fail(f"{_display(toolchain_path)} has no usable 'version' string.")
        return
    if not isinstance(container_version, str) or not container_version:
        fail(f"{_display(manifest_path)} tool {tool_id!r} has no usable 'version' string.")
        return

    _stats["pairs_checked"] = _stats.get("pairs_checked", 0) + 1

    if container_version != shipped_version:
        fail(
            f"{label} version drift: "
            f"{_display(manifest_path)} (tool id {tool_id!r}) pins "
            f"{container_version}, but {_display(toolchain_path)} pins "
            f"{shipped_version}. These must be equal. "
            f"{_display(manifest_path)} provisions {CONTAINER_ROLE}; "
            f"{_display(toolchain_path)} selects the compiler that builds "
            f"{SHIPPED_ROLE}. When they disagree, developers and the "
            "functional test suite are using a different compiler than the "
            "one the product is built with, and a codegen or ABI difference "
            "reproduces in neither direction. Align them -- and prefer "
            "moving the container to match the shipped version, because "
            "changing the shipped version is a real toolchain upgrade that "
            "needs artifact re-validation, not a manifest edit. Any version "
            "change must carry the matching artifact url and a verified "
            "sha256."
        )
        return

    container_sha = entry.get("sha256")
    shipped_sha = _shipped_sha256(shipped)
    if not isinstance(container_sha, str) or not isinstance(shipped_sha, str):
        fail(
            f"{label}: could not read a sha256 from both "
            f"{_display(manifest_path)} and {_display(toolchain_path)}."
        )
        return

    if container_sha.lower() != shipped_sha.lower():
        fail(
            f"{label} artifact drift: {_display(manifest_path)} and "
            f"{_display(toolchain_path)} both say version {shipped_version}, "
            f"but they point at different archives -- sha256 "
            f"{container_sha} vs {shipped_sha}. Same version label, different "
            "bytes, which defeats the point of pinning the version at all: "
            f"{_display(manifest_path)} provisions {CONTAINER_ROLE} while "
            f"{_display(toolchain_path)} builds {SHIPPED_ROLE}. Point both at "
            "the same artifact, and verify the sha256 by hashing the real "
            "download rather than copying a plausible-looking value."
        )


def _shipped_sha256(shipped: dict) -> str | None:
    """ci/toolchains/*.json nests the artifact under platforms[<platform>].

    Only linux-x86_64 exists today and it is the only platform CI builds on;
    if a second platform is ever added, this check needs to grow a platform
    loop rather than silently comparing the wrong one -- hence the explicit
    failure below instead of a .get() that would pick an arbitrary entry.
    """
    platforms = shipped.get("platforms")
    if not isinstance(platforms, dict):
        return None
    entry = platforms.get("linux-x86_64")
    if not isinstance(entry, dict):
        return None
    return entry.get("sha256")


def check_coverage(toolchain_dir: Path) -> None:
    """Every ci/toolchains/*.json is either paired or explicitly excused."""
    if not toolchain_dir.is_dir():
        fail(f"{_display(toolchain_dir)} is not a directory.")
        return

    known = {filename for _id, filename, _label in PAIRS} | set(UNPAIRED)
    on_disk = {p.name for p in sorted(toolchain_dir.glob("*.json"))}

    for name in sorted(on_disk - known):
        fail(
            f"{_display(toolchain_dir / name)} is not covered by this check. "
            "Add it to PAIRS in scripts/check_toolchain_manifest_sync.py if "
            "the dev container ships the same compiler, so the two are held "
            "on one version and one sha256; or add it to UNPAIRED with the "
            "reason it must differ. A new shipped-compiler pin that nobody "
            "registers is how the drift this guard exists for gets back in."
        )
    for name in sorted(known - on_disk):
        fail(
            f"scripts/check_toolchain_manifest_sync.py names "
            f"ci/toolchains/{name}, which does not exist. Update PAIRS or "
            "UNPAIRED to match the tree rather than leaving a dead entry."
        )
    _stats["manifests_covered"] = len(on_disk & known)


def check_toolchain_manifest_sync(
    tools_manifest: Path = TOOLS_MANIFEST,
    toolchain_dir: Path = TOOLCHAIN_DIR,
) -> None:
    check_coverage(toolchain_dir)
    manifest = load_json(tools_manifest)
    if manifest is None:
        return
    for tool_id, filename, label in PAIRS:
        check_pair(manifest, tools_manifest, toolchain_dir, tool_id, filename, label)

    # A pass with zero comparisons is a pass over nothing. Every id in PAIRS
    # should have produced a comparison; if none did, discovery is broken and
    # saying "OK" would be a lie.
    if not _stats.get("pairs_checked"):
        fail(
            "no armclang/ATfE version pairs were compared at all -- manifest "
            "discovery is broken and this check would otherwise pass "
            "vacuously."
        )


def report() -> None:
    if failures:
        print("Toolchain manifest sync check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "Toolchain manifest sync check OK: "
            f"{_stats.get('pairs_checked', 0)} compiler pin(s) agree between "
            "ci/tools/manifest.json (dev container) and ci/toolchains/*.json "
            "(shipped artifacts), on both version and sha256; all "
            f"{_stats.get('manifests_covered', 0)} ci/toolchains manifests are "
            f"registered here ({len(UNPAIRED)} deliberately unpaired)."
        )


def main() -> int:
    check_toolchain_manifest_sync()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
