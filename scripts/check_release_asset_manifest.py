#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Holds the "Required vs optional assets" table in docs/guides/releases.md to
# ci/release-assets.json, and holds the two executable consumers to expanding
# that manifest rather than carrying their own copy of it.
#
# Why this exists:
#   The asset contract lived in three hand-maintained places -- release.yml,
#   scripts/ci/audit_release_assets.sh and the docs table -- with nothing
#   relating them (AmbiqAI/ns-cmsis-nn#376). The copies agreed by luck. An
#   asset added to release.yml alone would be verified once at release time
#   and never audited again; an asset dropped from the docs table would leave
#   customers told to expect something the pipeline no longer promises.
#
#   Making the manifest authoritative is worthless without a gate, because
#   the docs and the manifest are edited by different people for different
#   reasons and will drift again. The docs table is checked rather than
#   generated: it carries three rows that are not release assets (the tag and
#   Release object, the CI image, the Pages deploy) and a "why" column of
#   editorial prose, so generating it would either lose that or push prose
#   into a machine-readable manifest.
#
# What is checked:
#   1. Table membership. Every manifest row (assets and the declared
#      non-asset rows) appears exactly once in the table, and every table row
#      is registered in the manifest. An unregistered asset-shaped row is the
#      drift this gate exists for; ignoring unknown rows would wave it
#      through.
#   2. Status agreement. Required/Optional in the table matches `class` in
#      the manifest.
#   3. The asset counts quoted in prose match what the manifest expands to,
#      with and without the ARMCLANG_REQUIRED promotion. Both guides are
#      read: the count appears below the table in releases.md and again in
#      the release section of verification.md. A number written into prose
#      goes stale silently; this is the only thing that can notice.
#   4. Both consumers invoke scripts/ci/release_assets.py, and neither still
#      enumerates the toolchain set to build names itself. Without this the
#      manifest could become authoritative for the docs while a consumer
#      quietly kept its own list.
#
# Modeled on check_toolchain_manifest_sync.py: pure Python, stdlib only, no
# build, sub-second, wired into pdsc.yml beside the other per-PR textual
# guards. Mutation-tested by scripts/tests/test_check_release_asset_manifest.py.

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ci"))

from release_assets import (  # noqa: E402
    MANIFEST,
    ManifestError,
    asset_names,
    load_manifest,
)

REPO = Path(__file__).resolve().parents[1]
RELEASES_DOC = REPO / "docs" / "guides" / "releases.md"
VERIFICATION_DOC = REPO / "docs" / "guides" / "verification.md"
GENERATOR = "scripts/ci/release_assets.py"

# Every place a bare asset count is written into prose. Each entry names the
# expansion it must equal: 'base' is the required set, 'promoted' is that set
# with the ARMCLANG_REQUIRED promotion applied. The verification guide carries
# a fourth copy of the number the other three quote, and a copy nothing checks
# is how the count went stale before.
RELEASES_PROSE = (
    (re.compile(r"from (\d+) assets to (\d+)"), ("base", "promoted")),
    (re.compile(r"the bar stays at (\d+)"), ("base",)),
)
VERIFICATION_PROSE = ((re.compile(r"publishes \*\*(\d+) required assets\*\*"), ("base",)),)
# (path, construct that must NOT come back, what that construct was).
#
# The forbidden construct is per-consumer rather than one pattern for both.
# release.yml legitimately loops over `gcc atfe armclang` in
# publish-staticlib-bundles, where it is building bundles rather than
# declaring a contract, so the anchor there is the contract array itself.
CONSUMERS = (
    (
        REPO / ".github" / "workflows" / "release.yml",
        re.compile(r"^\s*required=\(", re.MULTILINE),
        "release-verify's hand-built required-asset array",
    ),
    (
        REPO / "scripts" / "ci" / "audit_release_assets.sh",
        re.compile(r"\bgcc\s+atfe\b"),
        "the audit's hand-rolled loop over the toolchain set",
    ),
)

TABLE_HEADING = "## Required vs optional assets"

STATUS_TEXT = {"**Required**": "required", "**Optional**": "optional"}

failures: list[str] = []
_stats: dict[str, int] = {}


def fail(msg: str) -> None:
    failures.append(msg)


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_docs_table(doc: Path) -> list[tuple[str, str]] | None:
    """The (asset cell, status) pairs under TABLE_HEADING, in file order.

    Returns None after recording a failure if the section or the table cannot
    be found: a parser that silently returns nothing would turn this gate into
    a pass over zero rows, which is the shape of defect it exists to catch.
    """
    try:
        lines = doc.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        fail(
            f"{_display(doc)} does not exist, so the asset table cannot be "
            f"checked against {_display(MANIFEST)}. If the guide moved, update "
            "RELEASES_DOC in scripts/check_release_asset_manifest.py rather "
            "than deleting this check."
        )
        return None

    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == TABLE_HEADING)
    except StopIteration:
        fail(
            f"{_display(doc)} has no {TABLE_HEADING!r} heading. The asset "
            "contract table is what this gate reads; if it was renamed, update "
            "TABLE_HEADING in scripts/check_release_asset_manifest.py."
        )
        return None

    rows: list[tuple[str, str]] = []
    in_table = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if set(cells[0]) <= set("- :") and cells[0]:
            in_table = True
            continue
        if cells[0].lower() == "asset":
            continue
        if not in_table:
            continue
        rows.append((cells[0], cells[1]))

    if not rows:
        fail(
            f"{_display(doc)} has a {TABLE_HEADING!r} section but no table rows "
            "were parsed from it. This gate would otherwise pass over nothing."
        )
        return None
    return rows


def check_table(manifest: dict, doc_rows: list[tuple[str, str]], doc: Path, manifest_path: Path) -> None:
    expected: dict[str, str] = {}
    for row in manifest["rows"]:
        expected[row["docs_asset"]] = row["class"]
    for row in manifest.get("non_asset_rows", []):
        if not isinstance(row, dict):
            fail(f"{_display(manifest_path)} has a non_asset_rows entry that is not an object.")
            return
        asset = row.get("docs_asset")
        klass = row.get("class")
        if not isinstance(asset, str) or klass not in ("required", "optional"):
            fail(
                f"{_display(manifest_path)} non_asset_rows entry {row.get('id')!r} needs "
                "a 'docs_asset' string and a 'required'/'optional' class."
            )
            return
        expected[asset] = klass

    seen: dict[str, str] = {}
    for asset, status in doc_rows:
        if asset in seen:
            fail(
                f"{_display(doc)} lists {asset} twice in the asset "
                "table. Two rows for one asset can disagree with each other."
            )
            continue
        if status not in STATUS_TEXT:
            fail(
                f"{_display(doc)} row {asset} has status {status!r}; "
                f"expected one of {', '.join(sorted(STATUS_TEXT))}."
            )
            continue
        seen[asset] = STATUS_TEXT[status]

    for asset in sorted(set(seen) - set(expected)):
        fail(
            f"{_display(doc)} lists {asset}, which is not registered in "
            f"{_display(manifest_path)}. Add it there (as a 'rows' entry if the "
            "pipeline publishes it as a release asset, or a 'non_asset_rows' "
            "entry if it is not an asset), so release-verify and the nightly "
            "audit judge the same contract customers are shown."
        )
    for asset in sorted(set(expected) - set(seen)):
        fail(
            f"{_display(manifest_path)} declares {asset}, which the asset table in "
            f"{_display(doc)} does not list. An asset the pipeline "
            "requires and the guide never mentions is invisible to whoever has "
            "to judge whether a release shipped complete."
        )
    for asset in sorted(set(expected) & set(seen)):
        if expected[asset] != seen[asset]:
            fail(
                f"{asset} is '{expected[asset]}' in {_display(manifest_path)} but "
                f"'{seen[asset]}' in {_display(doc)}. Whichever is "
                "wrong, a release is being judged against one contract and "
                "described by another."
            )
        _stats["rows_compared"] = _stats.get("rows_compared", 0) + 1


def check_prose_counts(
    manifest: dict,
    doc: Path,
    manifest_path: Path,
    patterns: tuple[tuple["re.Pattern[str]", tuple[str, ...]], ...],
) -> None:
    """The bare numbers quoted in a guide's prose, against the expansion."""
    try:
        text = doc.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(
            f"{_display(doc)} does not exist, so the asset counts quoted in it "
            "cannot be checked. If the guide moved, update the doc constants in "
            "scripts/check_release_asset_manifest.py rather than dropping the check."
        )
        return

    version = "0.0.0"
    counts = {
        "base": len(asset_names(version, "required", False, manifest)),
        "promoted": len(asset_names(version, "required", True, manifest)),
    }

    for pattern, keys in patterns:
        wanted = tuple(counts[key] for key in keys)
        match = pattern.search(text)
        if match is None:
            fail(
                f"{_display(doc)} no longer contains a sentence matching "
                f"/{pattern.pattern}/. That prose is where the required-asset "
                "counts are quoted, and this gate is the only thing that keeps "
                "them true. Restore it or update the pattern here."
            )
            continue
        found = tuple(int(g) for g in match.groups())
        if found != wanted:
            fail(
                f"{_display(doc)} says {match.group(0)!r}, but "
                f"{_display(manifest_path)} expands to {wanted} for "
                f"{', '.join(keys)} ('base' is the required set, 'promoted' is "
                "the required set with the ARMCLANG_REQUIRED promotion applied). "
                "Update the prose to match the manifest."
            )
        _stats["counts_compared"] = _stats.get("counts_compared", 0) + 1


def check_consumers(consumers: tuple[tuple[Path, "re.Pattern[str]", str], ...]) -> None:
    for path, forbidden, description in consumers:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            fail(f"{_display(path)} does not exist; update CONSUMERS in this check.")
            continue
        if GENERATOR not in text:
            fail(
                f"{_display(path)} does not invoke {GENERATOR}. Both the release "
                "gate and the nightly audit must expand ci/release-assets.json, "
                "or they are back to judging releases against two lists that "
                "agree only by luck."
            )
        if forbidden.search(text):
            fail(
                f"{_display(path)} has grown {description} back "
                f"(matched /{forbidden.pattern}/). Building the contract in the "
                "consumer is how the asset list got triplicated in the first "
                f"place. Expand {GENERATOR} instead."
            )
        _stats["consumers_checked"] = _stats.get("consumers_checked", 0) + 1


def check_release_asset_manifest(
    manifest_path: Path = MANIFEST,
    doc: Path = RELEASES_DOC,
    consumers: tuple[tuple[Path, "re.Pattern[str]", str], ...] = CONSUMERS,
    verification_doc: Path = VERIFICATION_DOC,
) -> None:
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        fail(str(exc))
        return

    doc_rows = parse_docs_table(doc)
    if doc_rows is not None:
        check_table(manifest, doc_rows, doc, manifest_path)
    check_prose_counts(manifest, doc, manifest_path, RELEASES_PROSE)
    check_prose_counts(manifest, verification_doc, manifest_path, VERIFICATION_PROSE)
    check_consumers(consumers)

    # A pass with zero comparisons is a pass over nothing.
    if not _stats.get("rows_compared"):
        fail(
            "no asset rows were compared at all -- table discovery is broken "
            "and this check would otherwise pass vacuously."
        )


def report() -> None:
    if failures:
        print("Release asset manifest check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "Release asset manifest check OK: the asset table in "
            "docs/guides/releases.md agrees with ci/release-assets.json on "
            f"{_stats.get('rows_compared', 0)} row(s) and "
            f"{_stats.get('counts_compared', 0)} quoted count(s), and "
            f"{_stats.get('consumers_checked', 0)} consumer(s) expand the "
            "manifest rather than carrying their own copy."
        )


def main() -> int:
    check_release_asset_manifest()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
