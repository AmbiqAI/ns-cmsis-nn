#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for .github/workflows/release.yml's `gh release upload`
# call sites (AmbiqAI/ns-cmsis-nn#228, live recovery run 31333176320,
# defect 3).
#
# publish-staticlib-bundles has no checkout step, so `gh release upload`
# used to fail with "fatal: not a git repository" -- `gh` porcelain
# subcommands like `release upload`/`release view`/`release download` fall
# back to inferring the target repository from the local git remote when
# `--repo`/`-R` is omitted, and there is no local git repository at all in
# a no-checkout job. `gh api` calls are NOT affected (no such flag exists
# for `gh api`, and every call in this file already fully qualifies the API
# path with `${{ github.repository }}`, so it never needs remote
# inference) -- this test only requires `--repo`/`-R` on the porcelain
# `gh release *` commands, and explicitly asserts `gh api` calls are left
# alone.
#
# Runs entirely locally via static grep/awk assertions on the workflow
# file text: no GitHub Actions context, no network access.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW="${REPO}/.github/workflows/release.yml"

fail=0
report() {
  echo "FAIL: $1" >&2
  fail=1
}

# Every `gh release <verb> ...` invocation (view/upload/download/...) must
# carry --repo or -R somewhere on its (possibly multi-line, backslash
# continued) invocation. We approximate "the same invocation" by joining
# each line with the following line(s) while it ends in a backslash
# continuation, matching how these calls are written in this workflow.
joined="$(awk '
  # Skip pure comment lines entirely (e.g. explanatory prose that happens
  # to mention `gh release upload` in backticks) so they cannot be
  # mistaken for an actual call site.
  /^[[:space:]]*#/ { next }
  {
    line = $0
    while (line ~ /\\[[:space:]]*$/ && (getline nextline) > 0) {
      sub(/\\[[:space:]]*$/, "", line)
      line = line " " nextline
    }
    print line
  }
' "${WORKFLOW}")"

while IFS= read -r call; do
  [[ -n "${call}" ]] || continue
  if ! grep -qE -- '(--repo|-R)( |=)' <<< "${call}"; then
    report "gh release call is missing --repo/-R: ${call}"
  fi
done < <(grep -oE 'gh release [a-z]+[^`]*' <<< "${joined}" || true)

release_call_count="$(grep -cE 'gh release [a-z]+' <<< "${joined}" || true)"
if [[ "${release_call_count}" -lt 2 ]]; then
  report "expected at least 2 'gh release' call sites in release.yml (publish-staticlib-bundles upload + publish-pack upload), found ${release_call_count} -- test may be stale"
fi

# `gh api` calls must NOT have --repo/-R added (no such flag exists for
# `gh api`; adding one would be a CLI usage error). This guards against a
# regression seen during development of this fix, where --repo was
# mistakenly added to a `gh api repos/.../releases/tags/$TAG` call.
while IFS= read -r call; do
  [[ -n "${call}" ]] || continue
  if grep -qE -- '(--repo|-R)( |=)' <<< "${call}"; then
    report "gh api call must not use --repo/-R (unsupported flag): ${call}"
  fi
done < <(grep -oE 'gh api [^`]*' <<< "${joined}" || true)

if [[ "${fail}" -ne 0 ]]; then
  echo "release.yml gh --repo contract FAILED" >&2
  exit 1
fi

echo "release.yml gh --repo contract OK: every 'gh release' call site explicitly targets --repo/-R (safe for no-checkout jobs), and no 'gh api' call was given an unsupported --repo/-R flag."
