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
# for different reasons, so nothing else stops them drifting apart. The
# original guard lived as a workflow_dispatch-only step inside
# pack-dryrun.yml, which never runs on a PR and so could not catch drift
# before merge; this script is wired into release-contract.yml instead,
# which runs on every PR via ci.yml.
#
# What is checked, per file: exactly one distinct
# `gen-pack-action@<40-hex-sha>` pin (zero is a missing/renamed pin, more
# than one is an unresolved-merge or copy-paste leftover, either of which
# would make "the pin" ambiguous). Then the two files' single pins must be
# equal.
#
# Deliberately NOT checked: the trailing `# main @ <date>; bootstraps
# gen-pack <version>` comment on the `uses:` line. The SHA is the contract
# that actually pins behaviour; the comment is documentation and comparing
# it too would fail this check on a comment-only edit (e.g. correcting a
# typo in one file) that carries no behavioural drift at all.
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
# separated (possibly empty, possibly more than one -- counted by the
# caller). Pure extraction, no failure tracking here: a function's `fail=1`
# would not survive being invoked inside a `$(...)` command substitution,
# since that runs in a subshell, so ambiguity is reported by the caller in
# the main shell instead.
list_pins() {
  # `|| true`: a file with zero pins makes `grep -o` (and, under
  # pipefail, the whole pipe) exit non-zero, which under `set -e` would
  # abort this script's `var=$(list_pins ...)` assignment before the
  # explicit zero-count check below ever runs. Absence of a pin is a
  # reportable condition, not a script bug.
  grep -oh 'gen-pack-action@[0-9a-f]\{40\}' "$1" | sort -u || true
}

pack_dryrun_pins="$(list_pins "${PACK_DRYRUN}")"
release_pins="$(list_pins "${RELEASE}")"
pack_dryrun_count="$(printf '%s\n' "${pack_dryrun_pins}" | grep -c . || true)"
release_count="$(printf '%s\n' "${release_pins}" | grep -c . || true)"

# Zero or more than one distinct pin makes "the pin" ambiguous, so this is a
# hard failure rather than an arbitrary pick (e.g. `head -n1`) that would
# silently paper over the ambiguity.
if [[ "${pack_dryrun_count}" -eq 0 ]]; then
  report "no 'gen-pack-action@<sha>' pin found in ${PACK_DRYRUN}"
elif [[ "${pack_dryrun_count}" -gt 1 ]]; then
  report "expected exactly one distinct gen-pack-action pin in ${PACK_DRYRUN}, found ${pack_dryrun_count}: $(printf '%s' "${pack_dryrun_pins}" | tr '\n' ' ')"
fi

if [[ "${release_count}" -eq 0 ]]; then
  report "no 'gen-pack-action@<sha>' pin found in ${RELEASE}"
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
