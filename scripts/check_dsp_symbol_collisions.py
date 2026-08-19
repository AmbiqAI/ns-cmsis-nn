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
# release before it shipped. It was not a one-off: run this script with
# --list-hazards to DERIVE every "stem" (dtype-suffix stripped name) our
# public API and CMSIS-DSP's both use today -- each is a future kernel
# that must not take the bare, shorter name. Hand-maintaining that list in
# a comment is exactly how it goes stale: an earlier draft of this file
# named four families (add/sub/mean/sqrt) and missed a fifth, "abs"
# itself -- #281 fixed only the float half of that collision, so
# arm_abs_s8/s16 still shares a stem with CMSIS-DSP's arm_abs_q7/q15/q31/
# f16/f32/f64. --list-hazards cannot go stale the same way: it looks the
# answer up in the same two symbol sets check_dsp_symbol_collisions()
# compares, every time it runs. This guard exists so the next actual
# collision is caught by CI, not by hand.
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
#   repo's 57 Include/Internal/*.h symbols intersect CMSIS-DSP's public
#   surface, which is exactly what you'd expect: they carry compound,
#   already-scoped names (arm_nn_*_scalar_f16, *_opt_f16, *_common, *_impl)
#   precisely because they already have to avoid colliding with this
#   repo's OWN public names. The check scans only Include/*.h -- matching
#   both the validated #282 prototype exactly and CMSIS-DSP's own
#   public/private split on the other side of the comparison. Widening to
#   Include/Internal/*.h later is a one-line change (glob -> rglob) if a
#   future internal symbol ever takes a bare-verb shape.
#
# Regex notes (three deliberate departures from the literal #282 prototype,
# all three verified to be no-ops on this repo's own Include/*.h and on
# upstream ARM-software/CMSIS-NN's; #1 and #3 are also required for a
# complete CMSIS-DSP-side snapshot -- see each note):
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
#   3. The type/qualifier prefix before the arm_* name is OPTIONAL, to
#      match a split-line declaration where the return type sits alone on
#      the previous line and the arm_* name starts its own line with
#      nothing before it. Real example, ten lines apart, in this repo's
#      own Include/arm_nnfunctions.h: `arm_cmsis_nn_status arm_sqrt_s8(`
#      (same line) vs. `arm_cmsis_nn_status` / `arm_sqrt_s16(` (split).
#      A required-prefix regex found the first and missed the second --
#      17 public symbols invisible, including arm_sqrt_s16, the exact
#      family this guard watches. clang-format's 120-column limit is what
#      produces the split form, so a long float prototype is a likely
#      future trigger. Verified a no-op on the CMSIS-DSP side (upstream
#      never splits a declaration across lines: same 763-symbol snapshot
#      with or without this) and a strict superset on the "ours" side
#      (finds everything the required-prefix version found, plus the 17).
#      See DECL_RE's own comment below for the mechanics.
#
# Coverage boundary: this is the DECLARED public API (488 symbols today),
# not the full exported link surface. Two things beyond that boundary,
# neither scanned:
#   - Include/Internal/*.h itself declares 57 more symbols -- excluded by
#     the scope decision above, not by oversight.
#   - Source/**/*.c defines 25 more non-static (externally linked)
#     symbols that no header, public or Internal, declares anywhere (e.g.
#     arm_avg_pool_nhwc_f16, arm_mean_reduce_generic_s8 -- verified by a
#     one-off extension of extract_symbols() to Source/, not something
#     this script checks on every run). They compile clean because C
#     lets a definition serve as its own prototype for later use in the
#     same file, and today each is only ever called from the file that
#     defines it -- but "compiles clean" and "not externally linked" are
#     different properties, and nothing here verifies the second one. A
#     collision hiding among those ~82 symbols would be exactly as fatal
#     as one in the declared 488, and entirely invisible to this guard.
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
# an OPTIONAL whitespace-led type/qualifier prefix (letters, digits,
# underscore, space, '*', tab -- deliberately no '(' or ')', so it cannot
# skip over an intervening parameter list or macro invocation to reach a
# later, unrelated arm_* token), then the arm_* name itself immediately
# followed by '(' and NOT a '*' -- see the "Regex notes" above for why the
# leading-whitespace allowance and the (?!\s*\*) guard are there.
#
# The prefix is optional to cover split-line declarations, where the
# return type sits alone on its own line and the arm_* name starts the
# next line with nothing before it (clang-format's 120-column limit wraps
# a long prototype exactly this way -- Include/arm_nnfunctions.h has both
# styles ten lines apart: `arm_cmsis_nn_status arm_sqrt_s8(...)` then
# `arm_cmsis_nn_status\narm_sqrt_s16(...)`). A prior version of this regex
# required the prefix, so a same-line declaration was found but its
# split-line neighbor was invisible -- 17 public symbols missed, including
# arm_sqrt_s16 itself, the exact family this guard is watching. Verified
# empirically that making the prefix optional changes nothing on the
# CMSIS-DSP side (upstream never splits a declaration across lines) and
# is a strict superset on the "ours" side (finds everything the required-
# prefix version found, plus the 17 split-line names).
DECL_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z_][A-Za-z0-9_ \*\t]*\b)?(arm_[a-z0-9_]+)\s*\((?!\s*\*)",
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


# --- --list-hazards: derive, don't hand-maintain, the future-collision list

# Dtype suffixes stripped from a symbol name to get its "stem" -- covers
# every dtype either library attaches: ns-cmsis-nn's _s8/_s16/_s32/_u8/...
# /_f16/_f32, plus CMSIS-DSP's additional _q7/_q15/_q31/_q63/_f64.
_DTYPE_SUFFIX_RE = re.compile(
    r"_(?:f16|f32|f64|s8|s16|s32|s64|u8|u16|u32|u64|q7|q15|q31|q63)$"
)


def _stem(name: str) -> str:
    return _DTYPE_SUFFIX_RE.sub("", name)


def list_hazards(
    include_dir: Path = INCLUDE_DIR, dsp_symbols_file: Path = DSP_SYMBOLS_FILE
) -> dict[str, tuple[list[str], list[str]]]:
    """Stems (dtype-suffix stripped names) our public API and CMSIS-DSP's
    both use today. A shared stem is not itself a collision -- the check
    above is what enforces that -- it is advance warning: it means one
    library's convention for extending that verb to a new dtype is a bare
    name the other library may already own. Derived from the same two
    symbol sets check_dsp_symbol_collisions() compares, so unlike a
    hand-maintained comment this cannot silently go stale or drop a name.
    """
    ours = extract_symbols(sorted(include_dir.glob("*.h")))
    dsp = load_dsp_symbols(dsp_symbols_file)
    if dsp is None:
        return {}

    our_by_stem: dict[str, list[str]] = {}
    for n in ours:
        our_by_stem.setdefault(_stem(n), []).append(n)
    dsp_by_stem: dict[str, list[str]] = {}
    for n in dsp:
        dsp_by_stem.setdefault(_stem(n), []).append(n)

    shared = set(our_by_stem) & set(dsp_by_stem)
    return {s: (sorted(our_by_stem[s]), sorted(dsp_by_stem[s])) for s in sorted(shared)}


def print_hazards() -> None:
    hazards = list_hazards()
    if not hazards:
        print("No stems shared between our public symbols and CMSIS-DSP's.")
        return
    print(
        f"{len(hazards)} stem(s) shared with CMSIS-DSP -- porting one of "
        "our names below to a dtype CMSIS-DSP already lists is a future "
        "collision (see AGENTS.md's naming rule before writing that "
        "kernel):\n"
    )
    for stem, (our_names, dsp_names) in hazards.items():
        print(f"  {stem}")
        print(f"    ours: {', '.join(our_names)}")
        print(f"    dsp:  {', '.join(dsp_names)}")


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
    if "--list-hazards" in sys.argv[1:]:
        print_hazards()
        return 0
    check_dsp_symbol_collisions()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
