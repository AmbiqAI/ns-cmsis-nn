#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for AmbiqAI/ns-cmsis-nn#405.
#
# .github/workflows/pack-dryrun.yml and .github/workflows/release.yml each
# pin `uses: Open-CMSIS-Pack/gen-pack-action@<sha>` independently -- the
# pack-dryrun rehearsal is only a rehearsal of the release path if it runs
# the exact same action revision, and the two files are edited separately
# for different reasons, so nothing else stops them drifting apart. An
# earlier revision of this PR placed this guard as a workflow_dispatch-only
# step inside pack-dryrun.yml, which only runs on manual dispatch and so
# could not catch drift before merge; it now runs on every PR as
# scripts/check_gen_pack_action_pin.sh, wired into release-contract.yml
# (invoked from ci.yml).
#
# What is checked, per file: exactly one distinct `gen-pack-action@<sha>`
# pin (zero is a missing/renamed pin, more than one is an unresolved-merge
# or copy-paste leftover, either of which would make "the pin" ambiguous).
# Then the two files' single pins must be equal (case-insensitively --
# GitHub resolves a ref regardless of hex-digit case, so this compares
# pins normalised to lowercase rather than requiring the files to agree on
# casing too).
#
# A pin must be exactly 40 hex characters, bounded by whitespace or
# end-of-line. Anchoring on "the next character is not a hex digit" is NOT
# enough: git ref/branch names freely contain non-hex characters, so
# `gen-pack-action@<40-hex-sha>-rc1` would satisfy that weaker test while
# actually naming a 44-character ref that is not the pinned commit (and
# that GitHub may fail to resolve at all). Requiring whitespace/EOL as the
# boundary instead means: an accidental 41-hex-character value is rejected
# outright (found 0 pins) rather than silently truncated to a matching
# 40-character prefix, and a `-rc1`-style suffix is rejected the same way.
#
# Deliberately NOT checked: the trailing `# main @ <date>; bootstraps
# gen-pack <version>` comment on the `uses:` line. The SHA is the contract
# that actually pins behaviour; the comment is documentation and comparing
# it too would fail this check on a comment-only edit (e.g. correcting a
# typo in one file) that carries no behavioural drift at all.
#
# Deliberate policy, not an oversight: a commented-out `uses:` line (e.g.
# `# uses: Open-CMSIS-Pack/gen-pack-action@<sha>`) still counts as a pin.
# This script has no YAML/comment awareness -- it greps raw text -- and
# that is intentional here: a stale commented-out pin naming a different
# SHA than the live one is exactly the kind of copy-paste leftover this
# check exists to catch (see the ">1 distinct pin" ambiguity case above),
# so treating comments as invisible would create a blind spot rather than
# close one.
#
# Runs entirely locally via static grep assertions on the workflow file
# text: no GitHub Actions context, no network access.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DRYRUN="${REPO}/.github/workflows/pack-dryrun.yml"
RELEASE="${REPO}/.github/workflows/release.yml"

fail=0
report() {
  echo "FAIL: $1" >&2
  fail=1
}

# List the distinct gen-pack-action pins in a workflow file, newline
# separated, lowercase-normalised (possibly empty, possibly more than one
# -- counted by the caller). Pure extraction, no failure tracking here: a
# function's `fail=1` would not survive being invoked inside a `$(...)`
# command substitution, since that runs in a subshell, so ambiguity is
# reported by the caller in the main shell instead. Existence of the file
# is likewise checked by the caller before this is invoked, so a grep on a
# missing file (which would itself error under `set -e`) never happens.
list_pins() {
  # `-oE` with `([[:space:]]|$)` matches the trailing boundary as part of
  # the result; `sed` then strips exactly that one trailing whitespace
  # byte off each match (the `$`-alternative match has no such byte to
  # strip, since `$` is a zero-width assertion). `tr` lowercase-normalises
  # so `ABCDEF...` and `abcdef...` compare equal. `|| true`: a file with
  # zero pins makes `grep -o` (and, under pipefail, the whole pipe) exit
  # non-zero, which under `set -e` would abort this script's
  # `var=$(list_pins ...)` assignment before the explicit zero-count check
  # below ever runs. Absence of a pin is a reportable condition, not a
  # script bug.
  grep -ohE 'gen-pack-action@[0-9a-fA-F]{40}([[:space:]]|$)' "$1" \
    | sed -E 's/[[:space:]]$//' \
    | tr 'A-F' 'a-f' \
    | sort -u \
    || true
}

for f in "${PACK_DRYRUN}" "${RELEASE}"; do
  [[ -f "${f}" ]] || report "workflow file missing: ${f}"
done

if [[ "${fail}" -ne 0 ]]; then
  echo "gen-pack-action pin sync contract FAILED" >&2
  exit 1
fi

pack_dryrun_pins="$(list_pins "${PACK_DRYRUN}")"
release_pins="$(list_pins "${RELEASE}")"
pack_dryrun_count="$(printf '%s\n' "${pack_dryrun_pins}" | grep -c . || true)"
release_count="$(printf '%s\n' "${release_pins}" | grep -c . || true)"

# Zero or more than one distinct pin makes "the pin" ambiguous, so this is a
# hard failure rather than an arbitrary pick (e.g. `head -n1`) that would
# silently paper over the ambiguity.
if [[ "${pack_dryrun_count}" -eq 0 ]]; then
  report "no 'gen-pack-action@<40-hex-sha>' pin found in ${PACK_DRYRUN}"
elif [[ "${pack_dryrun_count}" -gt 1 ]]; then
  report "expected exactly one distinct gen-pack-action pin in ${PACK_DRYRUN}, found ${pack_dryrun_count}: $(printf '%s' "${pack_dryrun_pins}" | tr '\n' ' ')"
fi

if [[ "${release_count}" -eq 0 ]]; then
  report "no 'gen-pack-action@<40-hex-sha>' pin found in ${RELEASE}"
elif [[ "${release_count}" -gt 1 ]]; then
  report "expected exactly one distinct gen-pack-action pin in ${RELEASE}, found ${release_count}: $(printf '%s' "${release_pins}" | tr '\n' ' ')"
fi

# Only compare equality once both sides are unambiguous; comparing against
# an ambiguous or missing side would either duplicate the failure above
# with a confusing empty-vs-real-value message, or mask it entirely.
if [[ "${pack_dryrun_count}" -eq 1 && "${release_count}" -eq 1 \
      && "${pack_dryrun_pins}" != "${release_pins}" ]]; then
  report "gen-pack-action pin differs between ${PACK_DRYRUN} (${pack_dryrun_pins}) and ${RELEASE} (${release_pins}) -- align the two 'uses:' lines"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "gen-pack-action pin sync contract FAILED" >&2
  exit 1
fi

echo "gen-pack-action pin sync contract OK: ${PACK_DRYRUN} and ${RELEASE} both pin ${pack_dryrun_pins}."
