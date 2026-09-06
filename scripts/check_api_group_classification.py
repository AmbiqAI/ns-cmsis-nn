#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Guard for docs/_ext/api_group_index.py's GROUP_PATTERNS: every public
# top-level kernel must match at least one group, or it silently drops off
# the customer-facing "Browse By Kernel Family" page with an otherwise
# green docs build (#283).
#
# Modelled on check_dsp_symbol_collisions.py (#282/#285): pure Python, no
# build, sub-second, wired into pdsc.yml beside the existing guards.
#
# Layer decision -- headers directly, not the generated docs/api/ tree:
#   An earlier version of this check lived inside docs/_ext/api_group_index.py
#   itself and iterated the doxygen-generated function_*.rst pages,
#   intersected with names parsed from the public headers. That is
#   strictly weaker than iterating the headers directly: the generated-page
#   set is doxygen/exhale/breathe output, and a public kernel whose page
#   never gets generated -- which is not hypothetical, this repo already
#   carries ~150 pre-existing Breathe "Cannot find function" warnings for
#   unrelated reasons -- would be silently absent from that set and never
#   reach the check at all, on top of costing a full doxygen+sphinx cycle
#   (a couple of minutes, plus a network fetch of doxygen in CI) to
#   discover what is really a one-line pattern omission. Reading
#   Include/arm_nnfunctions*.h directly needs neither doxygen nor Sphinx,
#   runs in milliseconds, and cannot be blinded by a generation failure
#   somewhere else in the docs toolchain.
#
# GROUP_PATTERNS itself is imported from docs/_ext/api_group_index.py
# (importlib, matching test_check_pdsc.py's pattern for importing a
# script that is not on a normal import path) rather than duplicated, so
# there is exactly one place that defines "what group does this kernel
# name belong to".
#
# Header discovery is a glob (Include/arm_nnfunctions*.h), not a hardcoded
# filename list, and a minimum-file-count assertion, not just a
# non-empty-result one. A hardcoded two-item list silently degrades in two
# different ways: delete Include/ entirely and it produces zero names (an
# empty result is at least loud), but rename just arm_nnfunctions_flt.h --
# the float API, which is explicitly experimental and still churning --
# and it produces a smaller-but-nonzero name set with every float kernel
# quietly exempted from ever being checked again, which a bare
# non-empty-set check would wave through in silence. The glob plus a
# minimum count catches both: today there are two conceptually distinct
# public surfaces (the integer/quantized API and the float API), so
# finding fewer than two is itself the failure signal, independent of
# knowing either file's exact name.

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE_DIR = REPO / "Include"
API_GROUP_INDEX = REPO / "docs" / "_ext" / "api_group_index.py"

# See "Header discovery" above.
PUBLIC_HEADER_GLOB = "arm_nnfunctions*.h"
MIN_EXPECTED_PUBLIC_HEADERS = 2  # today: arm_nnfunctions.h + arm_nnfunctions_flt.h

# Matches a declaration/reference of the form `arm_some_name(` in a header.
# Deliberately permissive: it also picks up @copydoc/@ref mentions inside
# doc comments, not just the declarations themselves. That over-matching is
# harmless because every such mention necessarily names a real function
# that is itself declared somewhere in these same headers, so it can only
# add genuine public names, never fabricate one -- and it sidesteps having
# to parse multi-line C prototypes (return type and name often sit on
# different lines in these headers). Preprocessor conditionals (the float
# surface guards its declarations with `#if ARM_NN_ENABLE_F32`) are not
# parsed at all -- deliberately: nn.dxy.in's PREDEFINED already sets
# ARM_NN_ENABLE_F32=1/ARM_NN_ENABLE_F16=1 so Doxygen documents those
# kernels unconditionally, and this check's notion of "public" should
# match what Doxygen actually publishes, not the build-time flag state.
DECL_RE = re.compile(r"\b(arm_[A-Za-z0-9_]*[A-Za-z0-9])\s*\(")

failures: list[str] = []
_stats: dict[str, int] = {}


def fail(msg: str) -> None:
    failures.append(msg)


def load_api_group_index(path: Path = API_GROUP_INDEX):
    """Import docs/_ext/api_group_index.py by file path (it is not on a
    normal import path -- Sphinx only puts it on sys.path at build time via
    conf.py), so this check and the directive it guards always agree on
    GROUP_PATTERNS and the matching rule. Returns None (after recording a
    failure) if `path` does not exist.
    """
    if not path.is_file():
        fail(f"{path} not found -- cannot load GROUP_PATTERNS")
        return None
    spec = importlib.util.spec_from_file_location("api_group_index", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: api_group_index.py's ApiFunction is a
    # @dataclass, and dataclass's own class-processing looks the defining
    # module up in sys.modules by name -- skip this and it raises
    # AttributeError on a module that is perfectly valid.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_names(headers: list[Path]) -> set[str]:
    names: set[str] = set()
    for header in headers:
        names |= set(DECL_RE.findall(header.read_text(encoding="utf-8", errors="replace")))
    return names


def load_public_names(include_dir: Path) -> set[str] | None:
    """The public top-level kernel API surface: every name declared (or
    doc-referenced) in Include/arm_nnfunctions*.h. Returns None (after
    recording a failure) if header discovery looks broken, so callers can
    bail out without a confusing "0 gaps found" result -- see the
    "Header discovery" note above for why a bare non-empty check is not
    enough on its own.
    """
    paths = sorted(include_dir.glob(PUBLIC_HEADER_GLOB))
    if len(paths) < MIN_EXPECTED_PUBLIC_HEADERS:
        fail(
            f"found only {len(paths)} public API header(s) matching "
            f"'{PUBLIC_HEADER_GLOB}' under {include_dir} "
            f"({[p.name for p in paths]}), expected at least "
            f"{MIN_EXPECTED_PUBLIC_HEADERS}. A public header may have been "
            "renamed or removed -- if so this check would silently stop "
            "covering whatever moved, which is exactly the failure class "
            "this check exists to catch one layer up. If the reduction is "
            "deliberate (e.g. two public headers were merged into one), "
            "update PUBLIC_HEADER_GLOB / MIN_EXPECTED_PUBLIC_HEADERS in "
            "this script to match."
        )
        return None
    names = extract_names(paths)
    if not names:
        fail(
            f"{[p.name for p in paths]} matched '{PUBLIC_HEADER_GLOB}' but "
            "yielded zero function names -- the extraction regex or header "
            "contents changed in a way that broke this check."
        )
        return None
    return names


def check_api_group_classification(
    include_dir: Path = INCLUDE_DIR,
    api_group_index_path: Path = API_GROUP_INDEX,
) -> None:
    public_names = load_public_names(include_dir)
    if public_names is None:
        return

    agi = load_api_group_index(api_group_index_path)
    if agi is None:
        return

    _stats["public"] = len(public_names)

    unmatched = sorted(
        name
        for name in public_names
        if not any(agi._matches(name, patterns) for patterns in agi.GROUP_PATTERNS.values())
    )
    if unmatched:
        # api_group_index_path may not be under REPO in a test fixture, so
        # don't let a display-path computation raise ValueError and mask
        # the real report (same reasoning as check_dsp_symbol_collisions.py's
        # load_dsp_symbols()).
        try:
            display = api_group_index_path.relative_to(REPO)
        except ValueError:
            display = api_group_index_path
        fail(
            f"{len(unmatched)} public API function(s) declared in "
            f"Include/{PUBLIC_HEADER_GLOB} do not match any GROUP_PATTERNS "
            f"group in {display}, so they are missing from the 'Browse By "
            f"Kernel Family' page: {unmatched}. Add a pattern for each, or "
            "fix a drifted one -- see #283."
        )


def report() -> None:
    if failures:
        print("API group classification check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "API group classification check OK: "
            f"{_stats.get('public', 0)} public API functions "
            f"(Include/{PUBLIC_HEADER_GLOB}) each match at least one "
            "GROUP_PATTERNS group."
        )


def main() -> int:
    check_api_group_classification()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
