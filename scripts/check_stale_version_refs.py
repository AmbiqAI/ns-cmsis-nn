#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Release-time guard: no tracked file still stamps the *previous* release
# version (AmbiqAI/ns-cmsis-nn#347).
#
# Why this exists, given check_pdsc.py already audits extra-files
# ----------------------------------------------------------------
# check_pdsc.py's check #8 asks "does every path listed in
# release-please-config.json's extra-files carry a working annotation, and
# does its value match Include/arm_nn_types.h?". That is a check over the
# *list*. It is blind in exactly one direction: a file that stamps the
# release version but was never added to the list at all. Nothing names it,
# so nothing checks it, and it drifts one release behind forever while every
# gate stays green -- the same silent-omission shape as the toolchains.md bug
# that prompted this script, one level up.
#
# This check comes at it from the other side: it does not care what is
# listed anywhere. It reads the previous released version out of
# CHANGELOG.md and asserts that no tracked file mentions it, except the
# files that are *supposed* to talk about older releases -- the changelog
# itself, the recovery-command documentation, the release plumbing whose
# comments cite the runs that motivated it, and test fixtures that pin
# literal versions on purpose. Anything else mentioning the previous
# version, one release after that version shipped, is either a bump that
# did not happen or a stamp nobody knew was there.
#
# On the release-please PR itself (where arm_nn_types.h and CHANGELOG.md
# have moved to the new version but a silently-skipped extra-file has not),
# this fires with the offending path and line, before the release is cut.
#
# Scope note: this is deliberately a *previous version* check, not a "no
# hardcoded versions anywhere" check. A file can legitimately name an older
# release -- the minimum-version example in Include/arm_nn_types.h's doc
# comment, release.yml's post-mortem notes about specific past releases --
# and those can never come back round, since versions only go up. Only the
# immediately previous release is evidence of a missed bump, because that is
# the value a stamp left behind by a skipped update still holds.
#
# Run with: python3 scripts/check_stale_version_refs.py

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TYPES_H = REPO / "Include" / "arm_nn_types.h"
CHANGELOG = REPO / "CHANGELOG.md"

# `## [1.2.3](https://.../compare/v1.2.2...v1.2.3) (2026-01-01)`
CHANGELOG_HEADING_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)

# Paths allowed to mention the previous release, each with the reason it is
# allowed. A whole-file grant, because these files discuss old releases in
# prose and in fixtures throughout, not on identifiable lines. Adding a file
# here says "this file talks about release history"; it must never be used
# to silence a file that *stamps* the current version.
ALLOWED: dict[str, str] = {
    "CHANGELOG.md": "the release history itself",
    "docs/contributing.md": "recovery-command examples (`gh workflow run ... recover_tag=vX.Y.Z`)",
    "docs/guides/releases.md": "recovery-command examples and per-release post-mortem notes",
    ".github/workflows/release.yml": "comments citing the specific historical runs the recovery paths exist for",
    ".github/workflows/build_publish_docker.yml": "comments on recovering historical tags that predate the helper scripts",
    "scripts/ci/resolve_release_commit.sh": "comments naming the tags whose mis-resolution motivated the script",
    "scripts/ci/ensure_local_tag_annotation.sh": "comments naming the tags whose lightweight annotation motivated the script",
    "scripts/check_release_image_tags.sh": "contract-test fixtures asserting on literal tag strings",
    "scripts/check_release_commit_resolution.sh": "contract-test fixtures asserting on literal tag strings",
    "scripts/check_docker_checkout_order.sh": "comment describing the historical recovery build it guards",
    "scripts/check_publish_pack_tooling_checkout.sh": "comment describing the historical recovery build it guards",
    "scripts/tests/test_check_pdsc.py": "synthetic fixtures that hardcode both sides of a version comparison",
    "scripts/tests/test_check_stale_version_refs.py": "synthetic fixtures that hardcode both sides of a version comparison",
}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def parse_header_version(text: str) -> str | None:
    """(major, minor, patch) joined, from arm_nn_types.h's NS_CMSIS_NN_VERSION_*
    macros -- the same canonical source check_pdsc.py uses."""
    parts = []
    for scope in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"NS_CMSIS_NN_VERSION_{scope}\s*\((\d+)\)", text)
        if not m:
            return None
        parts.append(m.group(1))
    return ".".join(parts)


def previous_version(changelog_text: str, canonical: str) -> str | None:
    """The release before `canonical`, read off CHANGELOG.md's headings.

    Asserts the newest heading *is* canonical first: if the changelog and
    arm_nn_types.h disagree, "the previous version" is not well defined and
    every result below would be measured against the wrong string. That
    disagreement is itself a release-plumbing bug, so it is reported rather
    than worked around.
    """
    headings = CHANGELOG_HEADING_RE.findall(changelog_text)
    if len(headings) < 2:
        fail(
            f"CHANGELOG.md has {len(headings)} `## [x.y.z]` heading(s); need at least two "
            "to identify the previous release -- has the changelog format changed?"
        )
        return None
    if headings[0] != canonical:
        fail(
            f"CHANGELOG.md's newest heading is {headings[0]!r} but Include/arm_nn_types.h "
            f"says {canonical!r} -- one of them was bumped without the other"
        )
        return None
    return headings[1]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def scan(prev: str) -> None:
    # `v` optional so both `1.2.3` and the tag form `v1.2.3` are caught.
    # Left boundary rejects a longer version ending in this one (11.2.3);
    # right boundary rejects a longer one starting with it (1.2.30) while
    # still matching a filename suffix such as `...-1.2.3.a`.
    #
    # Example versions in this file are deliberately not 7.x: a real one
    # would make the check flag its own source the release after it is
    # written -- the same self-inflicted staleness it exists to catch.
    pattern = re.compile(rf"(?<![0-9.])v?{re.escape(prev)}(?![0-9])")

    for rel in tracked_files():
        if rel in ALLOWED:
            continue
        path = REPO / rel
        if not path.is_file():  # submodule entry, or a broken symlink
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: nothing version-shaped to read
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                fail(
                    f"{rel}:{lineno}: still mentions the previous release {prev} "
                    f"(current release is in Include/arm_nn_types.h). If this is a "
                    f"version stamp, it was not bumped -- add the file to "
                    f"release-please-config.json's extra-files with an "
                    f"x-release-please annotation. If it is a deliberate reference to "
                    f"release history, add the file to ALLOWED in "
                    f"scripts/check_stale_version_refs.py with the reason.\n"
                    f"    {line.strip()}"
                )

    for rel in sorted(ALLOWED):
        if not (REPO / rel).is_file():
            fail(
                f"ALLOWED names {rel}, which is not in the tree -- remove the stale "
                "entry so the allowlist keeps meaning what it says"
            )


def main() -> int:
    if not TYPES_H.is_file():
        print(f"FAIL: {TYPES_H} not found", file=sys.stderr)
        return 1
    if not CHANGELOG.is_file():
        print(f"FAIL: {CHANGELOG} not found", file=sys.stderr)
        return 1

    canonical = parse_header_version(TYPES_H.read_text(encoding="utf-8"))
    if canonical is None:
        print(
            "FAIL: Include/arm_nn_types.h NS_CMSIS_NN_VERSION_* markers unreadable",
            file=sys.stderr,
        )
        return 1

    prev = previous_version(CHANGELOG.read_text(encoding="utf-8"), canonical)
    if prev is not None:
        scan(prev)

    if failures:
        print(f"FAIL: stale version references ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"Stale version refs OK: current release {canonical}, previous {prev}; "
        f"no tracked file outside the {len(ALLOWED)}-entry release-history allowlist "
        f"mentions {prev}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
