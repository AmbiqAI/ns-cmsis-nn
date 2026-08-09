#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for .github/workflows/release.yml's publish-pack job
# tooling/source checkout separation (AmbiqAI/ns-cmsis-nn#228, live
# recovery run 31335858426, job 93301568122, defect 2).
#
# publish-pack's ONLY checkout used to be pinned directly to the resolved
# historical release commit (e.g. v7.29.2) -- there was no bootstrap
# "tooling" checkout at all. That historical commit predates
# scripts/ci/ensure_local_tag_annotation.sh entirely, so invoking it as
# `bash scripts/ci/ensure_local_tag_annotation.sh ...` failed with
# "No such file or directory" (exit 127) the moment a recovery run
# resolved to a pre-#229 tag.
#
# The fix adds a SECOND checkout of the CURRENT, merged repository into a
# dedicated `_tooling/` subdirectory (unpinned, so it always has the
# helper script) and invokes the helper from there -- against the
# already-checked-out historical working directory -- instead of a bare
# `scripts/ci/...` path that would resolve inside the (historical) main
# checkout.
#
# This test asserts, entirely offline via grep/awk against the workflow
# file text, that publish-pack:
#   - has its main checkout pinned to the resolved release commit;
#   - has a SEPARATE tooling checkout, using a non-default, non-empty
#     `path:`, that does NOT pin `ref:` (must float to whatever triggered
#     the run);
#   - runs that tooling checkout strictly between the main checkout and
#     the annotation step;
#   - invokes ensure_local_tag_annotation.sh via that tooling path (e.g.
#     `_tooling/scripts/ci/...`), never a bare `scripts/ci/...` path.
#
# A synthetic "broken" fixture mirroring the pre-fix, single-checkout
# shape is also checked to prove this test would have caught the live
# defect.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW="${REPO}/.github/workflows/release.yml"

fail=0
report() {
  echo "FAIL: $1" >&2
  fail=1
}

# job_block <workflow-file> <job-name>
# Prints the line range of a top-level job block (from its `  <job>:` line
# up to, but not including, the next top-level `  <key>:` line or EOF).
job_block() {
  local file="$1" name="$2"
  local start end
  start="$(grep -nE "^  ${name}:\$" "${file}" | head -n1 | cut -d: -f1)"
  [[ -n "${start}" ]] || return 1
  end="$(awk -v start="${start}" 'NR > start && /^  [[:alnum:]_-]+:$/ { print NR; exit }' "${file}")"
  if [[ -z "${end}" ]]; then
    end="$(($(wc -l < "${file}") + 1))"
  fi
  sed -n "${start},$((end - 1))p" "${file}"
}

# assert_publish_pack_tooling_checkout <job-text> <label>
# Runs every structural assertion against a publish-pack job block's text.
# Calls `report` (bumping the shared $fail) for each violation found, so
# the caller gets a specific, actionable message per assertion instead of
# one opaque pass/fail bit.
assert_publish_pack_tooling_checkout() {
  local text="$1" label="$2"
  local main_line tooling_line annotate_line main_block tooling_block
  local tooling_path invoke_line invoke_arg

  main_line="$(grep -nE -m1 '^ *- name: Checkout$' <<< "${text}" | cut -d: -f1)"
  tooling_line="$(grep -nE -m1 '^ *- name: Checkout current recovery tooling$' <<< "${text}" | cut -d: -f1)"
  annotate_line="$(grep -nE -m1 '^ *- name: Ensure tag has a local annotation for gen-pack$' <<< "${text}" | cut -d: -f1)"

  if [[ -z "${main_line}" ]]; then
    report "${label}: missing the main '- name: Checkout' step"
  fi
  if [[ -z "${tooling_line}" ]]; then
    report "${label}: missing a '- name: Checkout current recovery tooling' step -- without it, ensure_local_tag_annotation.sh cannot exist in a historical (pre-#229) recovery checkout (AmbiqAI/ns-cmsis-nn#228 live recovery run 31335858426 job 93301568122)"
  fi
  if [[ -z "${annotate_line}" ]]; then
    report "${label}: missing the '- name: Ensure tag has a local annotation for gen-pack' step"
  fi
  if [[ -z "${main_line}" || -z "${tooling_line}" || -z "${annotate_line}" ]]; then
    return 1
  fi

  if ! [[ "${main_line}" -lt "${tooling_line}" ]]; then
    report "${label}: the main (historical, pinned) checkout must run BEFORE the tooling checkout"
  fi
  if ! [[ "${tooling_line}" -lt "${annotate_line}" ]]; then
    report "${label}: the tooling checkout must run BEFORE the annotation step"
  fi

  main_block="$(sed -n "${main_line},$((tooling_line - 1))p" <<< "${text}")"
  tooling_block="$(sed -n "${tooling_line},$((annotate_line - 1))p" <<< "${text}")"

  # shellcheck disable=SC2016 # literal match against workflow YAML text
  if ! grep -q 'ref: \${{ needs.release-please.outputs.commit_sha }}' <<< "${main_block}"; then
    report "${label}: the main checkout must pin ref: \${{ needs.release-please.outputs.commit_sha }}"
  fi

  tooling_path="$(grep -m1 -oE 'path: *\S+' <<< "${tooling_block}" | awk '{print $2}')"
  if [[ -z "${tooling_path}" ]]; then
    report "${label}: the tooling checkout must set a non-default 'path:' (e.g. '_tooling') so it can never overwrite/be overwritten by the main historical checkout"
  fi
  if grep -q 'ref:' <<< "${tooling_block}"; then
    report "${label}: the tooling checkout must NOT pin 'ref:' -- it must float to whatever triggered the run, so it always has the current merged helper script even when recovering a pre-#229 tag"
  fi

  invoke_line="$(sed -n "${annotate_line},\$p" <<< "${text}" | grep -m1 'ensure_local_tag_annotation.sh' || true)"
  if [[ -z "${invoke_line}" ]]; then
    report "${label}: the annotation step does not appear to invoke ensure_local_tag_annotation.sh"
  else
    invoke_arg="$(grep -oE '[^ ]*ensure_local_tag_annotation\.sh' <<< "${invoke_line}" | head -n1)"
    if [[ -z "${tooling_path}" || "${invoke_arg}" != "${tooling_path}"/* ]]; then
      report "${label}: the annotation step must invoke the helper via the tooling path (found '${invoke_arg}', expected it to start with '${tooling_path:-<tooling path missing>}/') -- a bare 'scripts/ci/...' path resolves inside the historical, pre-#229 main checkout and does not exist there"
    fi
  fi
}

publish_pack_block="$(job_block "${WORKFLOW}" 'publish-pack')"
if [[ -z "${publish_pack_block}" ]]; then
  report "could not find the 'publish-pack:' job in ${WORKFLOW}"
else
  assert_publish_pack_tooling_checkout "${publish_pack_block}" "release.yml publish-pack" || true
fi

# --- Fixture: prove this test rejects the pre-fix, single-checkout shape ---
# Mirrors the ACTUAL live-failing shape from run 31335858426 job
# 93301568122: a single checkout pinned to the historical commit, and the
# annotation step invoking a bare `scripts/ci/...` path that does not
# exist in that historical tree. Evaluated against a THROWAWAY fail
# counter so this fixture's own (expected) violations are never reported
# as if they were real repo defects, and never mask real assertions above.
WORK="${REPO}/build/check_publish_pack_tooling_checkout_test"
rm -rf "${WORK}"
mkdir -p "${WORK}"
broken_fixture_file="${WORK}/broken_fixture.yml"
cat > "${broken_fixture_file}" <<'EOF'
  publish-pack:
    steps:
      - name: Checkout
        uses: actions/checkout@v6
        with:
          ref: ${{ needs.release-please.outputs.commit_sha }}
          fetch-depth: 0

      - name: Fetch tags
        run: git fetch --tags --force

      - name: Wait for release API to be ready
        run: echo ok

      - name: Ensure tag has a local annotation for gen-pack
        run: |
          bash scripts/ci/ensure_local_tag_annotation.sh "${TAG}" "${{ github.repository }}"
  publish-ci-image:
    steps: []
EOF
broken_fixture="$(cat "${broken_fixture_file}")"

real_fail="${fail}"
fail=0
assert_publish_pack_tooling_checkout "${broken_fixture}" "pre-fix-single-checkout fixture" >/dev/null 2>&1 || true
fixture_violation_count="${fail}"
fail="${real_fail}"

if [[ "${fixture_violation_count}" -eq 0 ]]; then
  report "the single-checkout (no tooling separation) pre-fix fixture unexpectedly PASSED the tooling-checkout contract -- this test would not have caught live defect 2 (run 31335858426, job 93301568122)"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "publish-pack tooling-checkout contract FAILED" >&2
  exit 1
fi

echo "publish-pack tooling-checkout contract OK: the historical, pinned checkout runs first; a separate, unpinned '_tooling'-path checkout supplies the current ensure_local_tag_annotation.sh helper; the annotation step invokes it via that tooling path against the historical working directory; the pre-fix single-checkout shape is correctly rejected."
