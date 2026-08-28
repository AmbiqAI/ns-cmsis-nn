#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Guard against unsubstituted documentation template tokens surviving into a
# public header (#288).
#
# Why this exists -- this is not a cosmetic docs check:
#   Include/arm_nnfunctions.h carried a literal, never-substituted
#   `{API}_get_buffer_size()` in the `ctx` doxygen of 17 public kernels. The
#   text tells the caller "call {API}_get_buffer_size() to size ctx->buf" and
#   then declines to say what {API} is, so every caller and every subsequent
#   copy-paste has to resolve it by hand. Twice (#269) someone resolved it by
#   hand and got it wrong -- both times landing on
#   arm_fully_connected_s8_get_buffer_size(), which reads a different
#   cmsis_nn_dims field than the kernel in question needs, under-sizes the
#   allocation, and the kernel then writes past it. A heap overflow, shipped,
#   from a docs placeholder. The placeholder is the defect source, so the
#   placeholder is what gets gated.
#
# Modelled on check_api_group_classification.py / check_dsp_symbol_collisions.py
# (#283/#282): pure Python, no build, sub-second, wired into pdsc.yml beside
# the existing per-PR textual guards.
#
# Why tokens are only flagged inside COMMENTS (plus an unconditional literal
# `{API}` scan):
#   The naive implementation -- regex `\{IDENT\}` over the whole file -- is
#   correct on today's tree (it currently matches the 17 placeholders and
#   nothing else, in all nine public headers) but is a false-positive
#   generator waiting to happen: perfectly ordinary C such as
#   `static const cmsis_nn_context empty = {NULL};` or a `{MACRO}` in a
#   designated initializer matches it exactly. A merge gate that fires on
#   valid code is a gate that gets deleted, so scanning is scoped to comment
#   text, where a `{...}` token has no legitimate meaning and a documentation
#   template placeholder is by construction the only thing it can be. The
#   literal token `{API}` is additionally flagged anywhere in the file
#   regardless of context, because it is never valid C in these headers and
#   catching it cannot cost a false positive.
#
# Second rule -- cited sizers must resolve:
#   The token rule alone cannot catch #269. Both of those defects were a
#   placeholder resolved to a REAL function name that was simply the wrong
#   one, and 8 of the 11 sites fixed in #288 had no same-named sizer at all,
#   so the "obvious" substitution would have named a function that does not
#   exist. Neither failure mode leaves a `{...}` token behind. So every
#   arm_*_get_buffer_size* name cited in a public-header comment must resolve
#   to something actually declared in the public headers.
#
#   Documentation deliberately names sizers that do NOT exist, to stop the
#   reader inventing one ("there is deliberately no arm_max_pool_s8_get_buffer_size()").
#   Those live in KNOWN_ABSENT_SIZERS. That allowlist is self-cleaning: it is
#   an error for a name in it to become declared, so if someone later adds a
#   real arm_max_pool_s8_get_buffer_size() the check fails and forces the
#   prose that says it does not exist to be corrected.
#
# Scope is Include/*.h -- the public, customer-facing surface, per #288.
# Include/Internal/*.h is excluded as non-customer-facing API, though doxygen
# does ingest it (nn.dxy.in INPUTs Include/ recursively). Flipping that is a
# one-line change to PUBLIC_HEADER_GLOB / rglob if the scope decision ever
# changes. docs/ markdown is deliberately not scanned either: the {VERSION}
# tokens there are legitimate template syntax, not leftovers.
#
# Header discovery uses a glob plus a required-filename anchor, not a bare
# non-empty-result check, for the reason spelled out at length in
# check_api_group_classification.py: an empty result is at least loud, but a
# *renamed* header yields a smaller-but-nonzero file set with whatever moved
# silently exempted from ever being checked again. Anchoring on the two
# headers that actually carry the templated kernel documentation catches that
# case by name rather than by count.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE_DIR = REPO / "Include"

# Public headers only -- non-recursive, so Include/Internal/ is out of scope.
PUBLIC_HEADER_GLOB = "*.h"

# The headers that carry templated kernel documentation. If either stops
# existing under PUBLIC_HEADER_GLOB, this check has silently stopped covering
# the surface it was written for and must say so rather than pass.
REQUIRED_HEADERS = ("arm_nnfunctions.h", "arm_nnfunctions_flt.h")

# A `{identifier}` token. Applied to comment text only -- see the module
# docstring for why. Deliberately rejects internal whitespace: `{API}` is a
# template placeholder, `{ x }` in a comment is far more likely to be prose
# or quoted code.
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

# The specific token behind #269/#288, flagged anywhere in the file
# regardless of comment context.
LITERAL_API_TOKEN = "{API}"

# A buffer-size query cited in prose, e.g. arm_avgpool_s8_get_buffer_size_mve.
SIZER_CITATION_RE = re.compile(r"\barm_[A-Za-z0-9_]*get_buffer_size[A-Za-z0-9_]*")

# Any `arm_foo(` in code -- what counts as "declared in the public headers".
DECL_RE = re.compile(r"\b(arm_[A-Za-z0-9_]+)\s*\(")

# Sizers the documentation names precisely BECAUSE they do not exist, so the
# reader does not invent one or reach for a plausible neighbour. Adding a name
# here asserts "no such function"; the check verifies that claim (see
# check_cited_sizers_resolve), so this list cannot silently rot.
KNOWN_ABSENT_SIZERS = {
    # No buffer needed; documented as such in #288.
    "arm_depthwise_conv_s8_get_buffer_size",
    "arm_depthwise_conv_s4_get_buffer_size",
    "arm_depthwise_conv_s16_get_buffer_size",
    "arm_fully_connected_s4_get_buffer_size",
    "arm_max_pool_s8_get_buffer_size",
    "arm_max_pool_s16_get_buffer_size",
    # Pre-existing, from the batch-matmul ctx documentation (#287).
    "arm_batch_matmul_s16_get_buffer_size",
}

failures: list[str] = []
_stats: dict[str, int] = {}


def fail(msg: str) -> None:
    failures.append(msg)


def comment_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of every C comment in `text`.

    Handles /* block */ and // line comments, and skips over string and char
    literals (including escapes) so that a "/*" appearing inside a string
    constant is not mistaken for the start of a comment. Unterminated block
    comments run to end-of-file, which matches how a compiler would see them.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"' or ch == "'":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "*":
                end = text.find("*/", i + 2)
                end = n if end == -1 else end + 2
                spans.append((i, end))
                i = end
                continue
            if nxt == "/":
                end = text.find("\n", i + 2)
                end = n if end == -1 else end
                spans.append((i, end))
                i = end
                continue
        i += 1
    return spans


def find_placeholders(text: str) -> list[tuple[int, str]]:
    """Every offending token in `text`, as (line_number, token).

    Two rules, unioned and de-duplicated:
      1. PLACEHOLDER_RE anywhere inside a comment.
      2. The literal `{API}` anywhere at all.
    """
    hits: dict[tuple[int, str], None] = {}

    def line_of(offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    for start, end in comment_spans(text):
        for m in PLACEHOLDER_RE.finditer(text, start, end):
            hits[(line_of(m.start()), m.group(0))] = None

    start = text.find(LITERAL_API_TOKEN)
    while start != -1:
        hits[(line_of(start), LITERAL_API_TOKEN)] = None
        start = text.find(LITERAL_API_TOKEN, start + 1)

    return sorted(hits)


def code_outside_comments(text: str, spans: list[tuple[int, int]]) -> str:
    """`text` with every comment region removed, so declarations can be
    collected without a comment that merely *mentions* a name counting as
    declaring it -- which would defeat the whole rule."""
    parts, prev = [], 0
    for start, end in spans:
        parts.append(text[prev:start])
        prev = end
    parts.append(text[prev:])
    return "".join(parts)


def check_cited_sizers_resolve(headers: list[Path]) -> None:
    """Every arm_*_get_buffer_size* named in a public-header comment must be
    declared in the public headers, or be an acknowledged absent sizer."""
    declared: set[str] = set()
    cited: dict[str, str] = {}  # name -> first "file:line" citing it

    for header in headers:
        text = header.read_text(encoding="utf-8", errors="replace")
        spans = comment_spans(text)
        declared |= set(DECL_RE.findall(code_outside_comments(text, spans)))
        try:
            display = header.relative_to(REPO)
        except ValueError:
            display = header
        for start, end in spans:
            for m in SIZER_CITATION_RE.finditer(text, start, end):
                cited.setdefault(m.group(0), f"{display}:{text.count(chr(10), 0, m.start()) + 1}")

    _stats["cited_sizers"] = len(cited)

    for name in sorted(cited):
        if name in declared or name in KNOWN_ABSENT_SIZERS:
            continue
        fail(
            f"{cited[name]}: public header documentation cites {name}(), which "
            "is not declared in any public header. Either the name is wrong or "
            "the function does not exist. A caller who follows this will reach "
            "for the nearest plausible sizer instead, which is how the #269 "
            "heap overflows happened. Fix the name, or if the doc is "
            "deliberately saying no such sizer exists, add it to "
            "KNOWN_ABSENT_SIZERS in this script. See #288."
        )

    for name in sorted(KNOWN_ABSENT_SIZERS & declared):
        fail(
            f"{name}() is listed in KNOWN_ABSENT_SIZERS as deliberately "
            "nonexistent, but it is now declared in a public header. The "
            "documentation that says it does not exist is now wrong: update "
            "that prose to point callers at it, and drop the name from "
            "KNOWN_ABSENT_SIZERS."
        )


def load_public_headers(include_dir: Path) -> list[Path] | None:
    """Public headers to scan. Returns None (after recording a failure) if
    discovery looks broken, so callers bail out rather than report a
    vacuous "0 placeholders found, clean"."""
    paths = sorted(p for p in include_dir.glob(PUBLIC_HEADER_GLOB) if p.is_file())
    if not paths:
        fail(
            f"no public headers matching '{PUBLIC_HEADER_GLOB}' found under "
            f"{include_dir} -- the header layout changed, or Include/ is "
            "missing. This check would otherwise pass vacuously."
        )
        return None
    names = {p.name for p in paths}
    missing = [h for h in REQUIRED_HEADERS if h not in names]
    if missing:
        fail(
            f"expected public header(s) {missing} not found under {include_dir} "
            f"(saw {sorted(names)}). These carry the templated kernel "
            "documentation this check exists to police, so their absence means "
            "the check has silently stopped covering it. If a header was "
            "deliberately renamed or merged, update REQUIRED_HEADERS in this "
            "script to match."
        )
        return None
    return paths


def check_header_placeholders(include_dir: Path = INCLUDE_DIR) -> None:
    headers = load_public_headers(include_dir)
    if headers is None:
        return

    _stats["headers"] = len(headers)
    total = 0
    for header in headers:
        text = header.read_text(encoding="utf-8", errors="replace")
        hits = find_placeholders(text)
        total += len(hits)
        for line, token in hits:
            try:
                display = header.relative_to(REPO)
            except ValueError:
                display = header
            fail(
                f"{display}:{line}: unsubstituted documentation template token "
                f"{token} in a public header. Resolve it to the real function "
                "name. Do NOT guess the name from the kernel's own spelling: "
                "resolving `{API}_get_buffer_size()` by hand, to a sizer that "
                "reads a different cmsis_nn_dims field than the kernel needs, "
                "is exactly how the #269 heap overflows shipped. Verify "
                "against the sizer's declaration and the kernel's actual "
                "ctx->buf usage, and if no sizer exists for this kernel, say "
                "so explicitly instead of naming a plausible one. See #288."
            )
    _stats["placeholders"] = total

    check_cited_sizers_resolve(headers)


def report() -> None:
    if failures:
        print("Header placeholder check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "Header placeholder check OK: "
            f"{_stats.get('headers', 0)} public header(s) "
            f"(Include/{PUBLIC_HEADER_GLOB}) carry no unsubstituted "
            "documentation template tokens, and all "
            f"{_stats.get('cited_sizers', 0)} cited arm_*_get_buffer_size* "
            "names resolve."
        )


def main() -> int:
    check_header_placeholders()
    report()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
