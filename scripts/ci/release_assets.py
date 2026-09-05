#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Expands ci/release-assets.json into the concrete asset names a given
# release version must (or may) carry.
#
# Why this exists:
#   The required-asset list was transcribed by hand into three places --
#   release.yml's release-verify job, scripts/ci/audit_release_assets.sh's
#   required_assets(), and the "Required vs optional assets" table in
#   docs/guides/releases.md -- with nothing relating them. The copies agreed
#   by luck. An asset added to release.yml alone would be verified at release
#   time and then never audited again, which is the #339 defect one layer up:
#   a gate that reads as coverage while covering nothing.
#
#   Both executable consumers now call this script instead of building names.
#   scripts/check_release_asset_manifest.py holds the docs table to the same
#   manifest, and asserts that neither consumer has grown a copy back.
#
# Usage:
#   release_assets.py <version> [--class required|optional|all]
#                               [--armclang-required] [--format lines|json]
#
#   <version> is the bare version with no leading 'v' (e.g. 7.31.0), matching
#   what release-please hands release.yml.
#
#   --armclang-required promotes the optional rows into the required set, as
#   the ARMCLANG_REQUIRED repository variable does for a release run. The
#   nightly audit deliberately never passes it: see audit_promotable in the
#   manifest.
#
# Exit codes: 0 = names written to stdout, 2 = usage or manifest error.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "ci" / "release-assets.json"

CLASSES = ("required", "optional")


class ManifestError(Exception):
    """The manifest is unusable. Raised rather than exiting so the checker and
    its mutation tests can import this module and see the reason."""


def load_manifest(path: Path = MANIFEST) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path} does not contain a JSON object at the top level")

    cpus = data.get("cpus")
    if not isinstance(cpus, list) or not cpus or not all(isinstance(c, str) for c in cpus):
        raise ManifestError(f"{path} has no usable 'cpus' array")

    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"{path} has no usable 'rows' array")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ManifestError(f"{path} rows[{index}] is not an object")
        for key in ("id", "template", "class", "docs_asset"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ManifestError(f"{path} rows[{index}] has no usable {key!r}")
        if row["class"] not in CLASSES:
            raise ManifestError(
                f"{path} row {row['id']!r} has class {row['class']!r}; "
                f"expected one of {', '.join(CLASSES)}"
            )
        for key in ("sha256_sidecar", "per_cpu"):
            if not isinstance(row.get(key), bool):
                raise ManifestError(f"{path} row {row['id']!r} has no boolean {key!r}")
        if "{version}" not in row["template"]:
            raise ManifestError(
                f"{path} row {row['id']!r} template {row['template']!r} does not "
                "interpolate {version}, so every release would collide on one name"
            )
        if row["per_cpu"] != ("{cpu}" in row["template"]):
            raise ManifestError(
                f"{path} row {row['id']!r} sets per_cpu={row['per_cpu']} but its "
                f"template {row['template']!r} "
                f"{'contains' if '{cpu}' in row['template'] else 'does not contain'} "
                "{cpu}. A per-cpu row that expands once silently drops two thirds "
                "of its assets from every check that reads this manifest."
            )
        if row["id"] in seen:
            raise ManifestError(f"{path} declares row id {row['id']!r} more than once")
        seen.add(row["id"])

    return data


def expand_row(row: dict, version: str, cpus: list[str]) -> list[str]:
    names: list[str] = []
    for cpu in cpus if row["per_cpu"] else [None]:
        name = row["template"].format(version=version, cpu=cpu)
        names.append(name)
        if row["sha256_sidecar"]:
            names.append(f"{name}.sha256")
    return names


def asset_names(
    version: str,
    wanted: str = "required",
    armclang_required: bool = False,
    manifest: dict | None = None,
) -> list[str]:
    """Asset names for `version`, in manifest order.

    `wanted` is "required", "optional" or "all". With `armclang_required`,
    optional rows count as required, exactly as the ARMCLANG_REQUIRED
    repository variable makes them at release time.
    """
    data = manifest if manifest is not None else load_manifest()
    cpus = data["cpus"]
    names: list[str] = []
    for row in data["rows"]:
        effective = "required" if armclang_required else row["class"]
        if wanted != "all" and effective != wanted:
            continue
        names.extend(expand_row(row, version, cpus))
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand ci/release-assets.json for one release version."
    )
    parser.add_argument("version", help="bare version, no leading 'v' (e.g. 7.31.0)")
    parser.add_argument(
        "--class",
        dest="wanted",
        choices=("required", "optional", "all"),
        default="required",
    )
    parser.add_argument("--armclang-required", action="store_true")
    parser.add_argument("--format", choices=("lines", "json"), default="lines")
    args = parser.parse_args(argv)

    if args.version.startswith("v"):
        parser.error(
            f"version {args.version!r} carries a leading 'v'; pass the bare "
            "version, since the asset names embed it verbatim"
        )
    if not args.version:
        parser.error("version must not be empty")

    try:
        names = asset_names(args.version, args.wanted, args.armclang_required)
    except ManifestError as exc:
        print(f"release_assets: {exc}", file=sys.stderr)
        return 2

    # An empty expansion is never legitimate for the required set: it would
    # let every consumer report "all assets present" over nothing at all,
    # which is the failure class this manifest exists to end.
    if not names and args.wanted != "optional":
        print(
            f"release_assets: {MANIFEST} expanded to no {args.wanted} assets; "
            "refusing to hand a consumer an empty contract",
            file=sys.stderr,
        )
        return 2

    if args.format == "json":
        print(json.dumps(names, indent=2))
    else:
        for name in names:
            print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
