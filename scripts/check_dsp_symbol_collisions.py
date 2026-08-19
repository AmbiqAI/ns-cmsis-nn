#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# CMSIS-DSP symbol collision guard for the public ns-cmsis-nn API.
#
# Modelled on check_pdsc.py (the idiom #264 established): pure Python, no
# build, sub-second, wired into pdsc.yml beside the existing guards.
#
# Motivation (#282): ns-cmsis-nn and CMSIS-DSP are two separately-versioned
# libraries that Ambiq ships and links together. Upstream CMSIS-NN maintains
# a zero-collision invariant against CMSIS-DSP's public API -- every bare
# float verb name CMSIS-DSP owns, upstream lengthened (arm_elementwise_add_f32
# instead of arm_add_f32, arm_maximum_f32 instead of arm_max_f32, etc.). That
# invariant has no enforcement anywhere in this repo. It was broken once
# already: arm_abs_f16/arm_abs_f32 (#240) collided with CMSIS-DSP's own
# arm_abs_f16/arm_abs_f32 -- same name, different signature, declared in
# different headers, so the compiler had no way to warn, and link order
# would have silently picked whichever one the linker resolved first. It
# was caught by hand in review (#281) and renamed to arm_nn_abs_f16/f32 one
# release before it shipped. Four more of our shipped integer kernel
# families (arm_add_s8/s16, arm_sub_s8/s16, arm_mean_s8/s16, arm_sqrt_s8/s16)
# have an "obvious" float name that CMSIS-DSP already owns -- this guard
# exists so the next one is caught by CI, not by hand.
#
# Method: parse `arm_*` function declarations directly out of header text
# (regex, no compiler/build) and intersect the two symbol sets. This was
# validated during the #281 review against an independent nm(1)/archive
# audit of built libraries and produced the identical answer, so header
# parsing is sufficient -- no build required.
#
# Scope decision -- Include/*.h only, not Include/Internal/:
#   Both this repo and CMSIS-DSP split their headers into a public,
#   consumer-facing set (Include/*.h here; Include/ + Include/dsp/ upstream)
#   and an internal, cross-translation-unit-sharing set (Include/Internal/
#   here; PrivateInclude/ upstream) that ships in the same archive but is
#   not meant to be #included by a consumer. Internal symbols ARE still
#   linked, so they are not risk-free -- but empirically, none of this
#   repo's 41 Include/Internal/*.h symbols intersect CMSIS-DSP's public
#   surface, which is exactly what you'd expect: they carry compound,
#   already-scoped names (arm_nn_*_scalar_f16, *_opt_f16, *_common, *_impl)
#   precisely because they already have to avoid colliding with this
#   repo's OWN public names. The check scans only Include/*.h -- matching
#   both the validated #282 prototype exactly and CMSIS-DSP's own
#   public/private split on the other side of the comparison. Widening to
#   Include/Internal/*.h later is a one-line change (glob -> rglob) if a
#   future internal symbol ever takes a bare-verb shape.
#
# Regex notes (two deliberate departures from the literal #282 prototype,
# both required for the CMSIS-DSP side and both verified to be no-ops on
# this repo's own Include/*.h and on upstream ARM-software/CMSIS-NN's):
#
#   1. Leading whitespace is allowed (`^[ \t]*` instead of a bare `^`).
#      This repo's headers declare functions at column 0, but CMSIS-DSP's
#      public headers indent every declaration two spaces inside their
#      `extern "C" { ... }` block -- e.g. Include/dsp/basic_math_functions.h
#      has `  void arm_abs_f32(`. A column-0-anchored regex over CMSIS-DSP's
#      headers finds only 341 of its 763 real declarations and silently
#      misses arm_abs_f32/f16 themselves -- which would have made this
#      exact check blind to the collision it exists to catch. Confirmed
#      this is a no-op on the "ours" side: ns-cmsis-nn's own Include/*.h
#      and upstream CMSIS-NN's both parse to the identical symbol set with
#      or without the whitespace allowance.
#   2. A `(?!\s*\*)` guard excludes `RETTYPE (*name)(args)` shapes --
#      function-pointer typedefs and fields, where the naive regex
#      captures the return type as if it were a declared function. Real
#      example in this repo today: Include/Internal/arm_conv1x1_opt_f16.h
#      has `typedef arm_cmsis_nn_status (*arm_conv1x1_call_f16)(...)`,
#      which without the guard spuriously "declares" a function named
#      arm_cmsis_nn_status. Harmless today (that name isn't a CMSIS-DSP
#      symbol, and Include/Internal/ isn't scanned anyway -- see above),
#      but the same shape appears in CMSIS-DSP's own headers in other
#      forms, so the guard is kept on both sides rather than relying on
#      today's scope decision to make it moot forever.
#
# Data file: scripts/data/cmsis_dsp_symbols.txt is a checked-in snapshot of
# CMSIS-DSP's public symbol surface, extracted with extract_symbols() below
# against a fresh CMSIS-DSP checkout (see that file's header for the exact
# ref/date and refresh recipe). ns-cmsis-nn is public and Ambiq's vendored
# CMSIS-DSP fork (helia-dsp) is private, so CI cannot fetch it live -- it
# must be checked in.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE_DIR = REPO / "Include"
DSP_SYMBOLS_FILE = REPO / "scripts" / "data" / "cmsis_dsp_symbols.txt"

# Matches a `arm_*` function declaration or definition at file scope:
# an optional-whitespace-led type/qualifier prefix (letters, digits,
# underscore, space, '*', tab -- deliberately no '(' or ')', so it cannot
# skip over an intervening parameter list or macro invocation to reach a
# later, unrelated arm_* token), then the arm_* name itself immediately
# followed by '(' and NOT a '*' -- see the "Regex notes" above for why both
# the leading-whitespace allowance and the (?!\s*\*) guard are there.
DECL_RE = re.compile(
    r"^[ \t]*[A-Za-z_][A-Za-z0-9_ \*\t]*\b(arm_[a-z0-9_]+)\s*\((?!\s*\*)",
    re.M,
)

# Symbols permitted to collide with CMSIS-DSP despite the check below.
#
# This must stay empty or near-empty. A collision here is a real hazard,
# not a style nit: two same-named, differently-signed exported functions
# in libraries Ambiq ships and links together, where the linker picks
# whichever one resolves first with no warning from the compiler (see
# arm_abs_f16/arm_abs_f32, caught by hand in #281 one release before they
# shipped). The fix is almost always to rename the ns-cmsis-nn symbol
# before it ships -- see the naming rule in AGENTS.md -- not to allowlist
# it.
#
# The only legitimate reason to add an entry: a symbol that has ALREADY
# shipped under a colliding name, so renaming it would itself be the
# breaking change, and the collision has been reviewed and deliberately
# accepted. Every entry must cite the issue/PR that made that call.
ALLOWLIST: frozenset[str] = frozenset()

failures: list[str] = []
_stats: dict[str, int] = {}


def fail(msg: str) -> None:
    failures.append(msg)


def extract_symbols(headers: list[Path]) -> set[str]:
    """Return every `arm_*` symbol DECL_RE finds declared across `headers`.

    Shared by the live check (against this repo's Include/*.h) and the
    scripts/data/cmsis_dsp_symbols.txt refresh recipe (against a fresh
    CMSIS-DSP checkout's Include/), so both sides of the comparison are
    always built with exactly the same extraction rule.
    """
    names: set[str] = set()
    for header in headers:
        names |= set(DECL_RE.findall(header.read_text(encoding="utf-8")))
    return names


def load_dsp_symbols(path: Path) -> set[str] | None:
    """Read the checked-in CMSIS-DSP snapshot: one symbol per non-blank,
    non-comment line. Returns None (after recording a failure) if the file
    is missing, so callers can bail out without a confusing empty-set
    "clean" result."""
    if not path.is_file():
        # path may not be under REPO (e.g. a test fixture in a tempdir), so
        # don't let a display-path computation raise ValueError and mask
        # the real "file not found" report.
        try:
            display = path.relative_to(REPO)
        except ValueError:
            display = path
        fail(
            f"{display} not found -- see that file's header (once it "
            "exists) for the refresh recipe, or scripts/data/ in the PR "
            "that added this check"
        )
        return None
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line)
    return names


def check_dsp_symbol_collisions(
    include_dir: Path = INCLUDE_DIR,
    dsp_symbols_file: Path = DSP_SYMBOLS_FILE,
) -> None:
    ours = extract_symbols(sorted(include_dir.glob("*.h")))
    dsp = load_dsp_symbols(dsp_symbols_file)
    if dsp is None:
        return

    _stats["ours"] = len(ours)
    _stats["dsp"] = len(dsp)

    hits = sorted((ours & dsp) - ALLOWLIST)
    if hits:
        fail(
            f"public symbol(s) collide with CMSIS-DSP's public API: {hits}. "
            "Same name, two libraries Ambiq ships and links together -- the "
            "linker silently picks whichever one resolves first, with no "
            "compiler warning (this is exactly how arm_abs_f16/arm_abs_f32 "
            "shipped almost-broken in #240, caught by hand in #281). Rename "
            "the ns-cmsis-nn symbol before it ships -- see AGENTS.md's "
            "naming rule (arm_elementwise_*/arm_maximum_*-style longer "
            "forms, or the arm_nn_ prefix, which CMSIS-DSP does not use at "
            "all). If a collision is truly unavoidable and has been "
            "reviewed and accepted, add it to ALLOWLIST in this script with "
            "a comment citing the issue/PR that approved the exception."
        )


def report() -> None:
    if failures:
        print("DSP symbol collision check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "DSP symbol collision check OK: "
            f"{_stats.get('ours', 0)} public ns-cmsis-nn symbols "
            f"(Include/*.h) checked against {_stats.get('dsp', 0)} "
            "CMSIS-DSP symbols (scripts/data/cmsis_dsp_symbols.txt), "
            "zero collisions."
        )


def main() -> int:
    check_dsp_symbol_collisions()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
