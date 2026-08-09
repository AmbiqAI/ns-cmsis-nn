#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for scripts/ci/resolve_release_commit.sh -- the
# tag-to-commit resolution logic release.yml's release-please job uses to
# pin every downstream release asset build/test to the release tag's exact
# target commit (AmbiqAI/ns-cmsis-nn#228, blocker 1).
#
# Stubs `gh` as a fake executable earlier in PATH so this exercises the
# retry/validation logic with no network access and no live GitHub API
# credentials.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${REPO}/scripts/ci/resolve_release_commit.sh"
WORK="${REPO}/build/release_commit_resolution_test"
FAKE_BIN="${WORK}/bin"

rm -rf "${WORK}"
mkdir -p "${FAKE_BIN}"

fail=0

report() {
  echo "FAIL: $1" >&2
  fail=1
}

# install_fake_gh <script-body>
# Writes a fake `gh` executable to FAKE_BIN implementing the given body,
# which receives `gh api repos/<owner>/<repo>/commits/<tag> -q .sha` calls.
# WORK is exported so a body can reference it directly (e.g. to persist a
# call counter across invocations) without string-concatenation tricks.
install_fake_gh() {
  cat > "${FAKE_BIN}/gh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export WORK="${WORK}"
# Expected invocation: gh api repos/OWNER/REPO/commits/REF -q .sha
if [[ "\$1" != "api" ]]; then
  echo "unexpected gh subcommand: \$1" >&2
  exit 2
fi
path="\$2"
ref="\${path##*/commits/}"
${1}
EOF
  chmod +x "${FAKE_BIN}/gh"
}

# run_case <name> <fake-gh-body> <tag> <max_attempts> <sleep_secs> <expect_success> [expected_sha]
run_case() {
  local name="$1" body="$2" tag="$3" max_attempts="$4" sleep_secs="$5" expect_success="$6"
  local expected_sha="${7-}"

  install_fake_gh "${body}"
  local out="${WORK}/${name}.out"
  rm -f "${out}"

  local rc=0
  PATH="${FAKE_BIN}:${PATH}" bash "${SCRIPT}" "owner/repo" "${tag}" "${out}" \
    "${max_attempts}" "${sleep_secs}" > "${WORK}/${name}.log" 2>&1 || rc=$?

  if [[ "${expect_success}" == "true" ]]; then
    if [[ "${rc}" -ne 0 ]]; then
      report "${name}: expected success, got exit ${rc}. Log:
$(cat "${WORK}/${name}.log")"
      return
    fi
    local actual_sha
    actual_sha="$(grep -m1 '^commit_sha=' "${out}" | cut -d= -f2-)"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      report "${name}: commit_sha='${actual_sha}', expected '${expected_sha}'"
    fi
  else
    if [[ "${rc}" -eq 0 ]]; then
      report "${name}: expected failure, but script exited 0"
      return
    fi
    if [[ -f "${out}" ]] && grep -q '^commit_sha=' "${out}" 2>/dev/null; then
      report "${name}: expected no commit_sha written on failure, but found one"
    fi
  fi
}

VALID_SHA_A="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
VALID_SHA_B="0123456789abcdef0123456789abcdef01234567"

# --- Case 1: immediate success ----------------------------------------------
run_case "immediate_success" \
  'echo '"${VALID_SHA_A}"'' \
  "v7.29.2" 5 0 "true" "${VALID_SHA_A}"

# --- Case 2: eventual success after transient failures (retry loop) --------
# Fails the first 2 calls (simulating the eventual-consistency window right
# after a tag/release is created), succeeds on the 3rd.
run_case "eventual_success_after_retries" \
  "count_file=\"\${WORK}/eventual_success_after_retries.count\"
touch \"\${count_file}\"
n=\"\$(wc -l < \"\${count_file}\")\"
echo x >> \"\${count_file}\"
if [[ \"\${n}\" -lt 2 ]]; then
  exit 1
fi
echo ${VALID_SHA_B}" \
  "v7.29.1" 5 0 "true" "${VALID_SHA_B}"

# --- Case 3: permanent failure (tag never resolves) -------------------------
run_case "permanent_failure" \
  'exit 1' \
  "v7.29.2" 3 0 "false"

# --- Case 4: malformed SHA response (not 40 hex chars) ----------------------
run_case "malformed_sha" \
  'echo "not-a-real-sha"' \
  "v7.29.2" 3 0 "false"

# --- Case 5: empty response treated as a retryable miss then exhausted -----
run_case "empty_response" \
  'echo ""' \
  "v7.29.2" 2 0 "false"

# --- Case 6: invalid tag format rejected before ever calling gh -------------
# Confirms input validation happens pre-flight (injection hardening): a
# malicious/malformed tag must never reach the `gh api` call.
# shellcheck disable=SC2016
run_case "invalid_tag_format_rejected" \
  "echo should-never-be-called >> \"\${WORK}/invalid_tag_called.marker\"; echo ${VALID_SHA_A}" \
  '$(id)' 3 0 "false"

if [[ -f "${WORK}/invalid_tag_called.marker" ]]; then
  report "invalid_tag_format_rejected: gh was invoked despite an invalid tag -- injection validation regressed"
fi

# --- Case 7: another invalid-format tag (missing patch component) -----------
run_case "invalid_tag_format_no_patch" \
  'echo '"${VALID_SHA_A}"'' \
  "v7.29" 3 0 "false"

if [[ "${fail}" -ne 0 ]]; then
  echo "release commit-resolution contract FAILED" >&2
  exit 1
fi

echo "release commit-resolution contract OK: immediate/retried success, permanent failure, malformed/empty response, and invalid tag-format rejection all behave correctly."
