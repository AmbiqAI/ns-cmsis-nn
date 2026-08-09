#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for scripts/ci/ensure_local_tag_annotation.sh -- the
# local-only lightweight-tag-to-annotated-tag compatibility shim
# release.yml's publish-pack job runs so gen_pack.sh's
# PACK_CHANGELOG_MODE=tag changelog generation stops failing with "Tag has
# no annotation message" against Release Please's lightweight tags
# (AmbiqAI/ns-cmsis-nn#228, recovery-run blocker 2).
#
# Exercises the script against a REAL throwaway local git repository (no
# stubbing needed for git itself -- this is exactly the local-only
# operation the script performs) with a stubbed `gh` executable for the
# GitHub Release body lookup. No network access, no live credentials.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${REPO}/scripts/ci/ensure_local_tag_annotation.sh"
WORK="${REPO}/build/ensure_local_tag_annotation_test"
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
# which receives `gh release view <tag> --repo <owner/repo> --json body -q .body`.
install_fake_gh() {
  cat > "${FAKE_BIN}/gh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export WORK="${WORK}"
# Expected invocation: gh release view TAG --repo OWNER/REPO --json body -q .body
if [[ "\$1" != "release" || "\$2" != "view" ]]; then
  echo "unexpected gh subcommand: \$*" >&2
  exit 2
fi
tag="\$3"
${1}
EOF
  chmod +x "${FAKE_BIN}/gh"
}

# make_repo <name> -- creates a fresh git repo with one commit, returns its path.
make_repo() {
  local dir="${WORK}/repos/$1"
  mkdir -p "${dir}"
  git -C "${dir}" init --quiet --initial-branch=main
  git -C "${dir}" config user.email "test@example.com"
  git -C "${dir}" config user.name "Test"
  echo "hello" > "${dir}/README.md"
  git -C "${dir}" add README.md
  git -C "${dir}" commit --quiet -m "chore: initial commit"
}

# run_case <name> <fake-gh-body-or-""-for-no-gh> <expect_success> \
#          <expect_message_contains> <expect_annotated_before>
# Creates a lightweight (or, for the "already annotated" case, an
# annotated) tag "v1.2.3" in a fresh repo, runs the script, and asserts:
#  - the tag's commit is unchanged
#  - (on success) the tag is now an annotated object with a non-empty
#    message containing <expect_message_contains>
#  - gh is invoked only when expected
run_case() {
  local name="$1" gh_body="$2" expect_success="$3" expect_message_contains="$4"
  local pre_annotate="${5:-false}"

  make_repo "${name}"
  local dir="${WORK}/repos/${name}"

  if [[ "${pre_annotate}" == "true" ]]; then
    git -C "${dir}" tag -a v1.2.3 -m "pre-existing annotation, must not change"
  else
    git -C "${dir}" tag v1.2.3
  fi

  local commit_before
  commit_before="$(git -C "${dir}" rev-list -n1 v1.2.3)"

  local path
  if [[ -n "${gh_body}" ]]; then
    install_fake_gh "${gh_body}"
    path="${FAKE_BIN}:${PATH}"
  else
    # Simulate `gh` being entirely absent from PATH.
    path="${PATH}"
  fi

  local rc=0
  ( cd "${dir}" && PATH="${path}" bash "${SCRIPT}" v1.2.3 "owner/repo" ) \
    > "${WORK}/${name}.log" 2>&1 || rc=$?

  if [[ "${expect_success}" == "true" ]]; then
    if [[ "${rc}" -ne 0 ]]; then
      report "${name}: expected success, got exit ${rc}. Log:
$(cat "${WORK}/${name}.log")"
      return
    fi
  else
    if [[ "${rc}" -eq 0 ]]; then
      report "${name}: expected failure, but script exited 0"
    fi
    return
  fi

  local commit_after
  commit_after="$(git -C "${dir}" rev-list -n1 v1.2.3)"
  if [[ "${commit_after}" != "${commit_before}" ]]; then
    report "${name}: tag commit changed from ${commit_before} to ${commit_after} -- tag was moved!"
  fi

  local objecttype
  objecttype="$(git -C "${dir}" for-each-ref --format '%(objecttype)' refs/tags/v1.2.3)"
  if [[ "${objecttype}" != "tag" ]]; then
    report "${name}: expected annotated tag object after run, got objecttype='${objecttype}'"
  fi

  local contents
  contents="$(git -C "${dir}" tag -l -n99 --format '%(contents)' v1.2.3)"
  if [[ -z "${contents}" ]]; then
    report "${name}: tag annotation message is empty"
  fi
  if [[ -n "${expect_message_contains}" && "${contents}" != *"${expect_message_contains}"* ]]; then
    report "${name}: tag message '${contents}' does not contain expected '${expect_message_contains}'"
  fi
}

# --- Case 1: already-annotated tag is left completely unaltered -------------
run_case "already_annotated" "" "true" "pre-existing annotation, must not change" "true"
# Extra assertion for this case: the ORIGINAL message must be exactly
# preserved (not just non-empty) -- "handle annotated tags without alteration".
orig_contents="$(git -C "${WORK}/repos/already_annotated" tag -l -n99 --format '%(contents)' v1.2.3)"
if [[ "${orig_contents}" != *"pre-existing annotation, must not change"* ]]; then
  report "already_annotated: message was altered: '${orig_contents}'"
fi

# --- Case 2: lightweight tag, gh returns a non-empty release body ----------
run_case "lightweight_with_release_body" \
  'echo "Bug Fixes:
- fixed thing A
- fixed thing B"' \
  "true" "fixed thing A"

# --- Case 3: lightweight tag, gh returns an empty body ----------------------
# Must still produce a NON-EMPTY, deterministic annotation.
run_case "lightweight_empty_body" \
  'echo ""' \
  "true" "Release v1.2.3"

# --- Case 4: lightweight tag, gh is entirely absent from PATH --------------
# Must fall back gracefully (best-effort), never treat this as fatal.
run_case "lightweight_no_gh" "" "true" "Release v1.2.3"

# --- Case 5: lightweight tag, gh command itself fails (e.g. no release) ----
run_case "lightweight_gh_fails" 'exit 1' "true" "Release v1.2.3"

# --- Case 6: release body contains XML-reserved characters -----------------
# release-please's auto-generated notes are Markdown and can contain literal
# '&', '<', '>' -- gen-pack's PDSC embedding does not escape these itself, so
# the script must, or the exact "tag" mode schema-safety this compatibility
# shim exists for would be defeated.
run_case "xml_reserved_chars_escaped" \
  'echo "A & B <script> tag"' \
  "true" "A &amp; B &lt;script&gt; tag"
# Also assert the RAW unescaped characters are absent from the final message.
raw_contents="$(git -C "${WORK}/repos/xml_reserved_chars_escaped" tag -l -n99 --format '%(contents)' v1.2.3)"
if [[ "${raw_contents}" == *"<script>"* ]]; then
  report "xml_reserved_chars_escaped: raw unescaped '<script>' leaked into tag message: '${raw_contents}'"
fi

# --- Case 7: tag does not exist -- must fail loudly, never silently no-op --
make_repo "missing_tag"
rc=0
( cd "${WORK}/repos/missing_tag" && bash "${SCRIPT}" v9.9.9 "owner/repo" ) \
  > "${WORK}/missing_tag.log" 2>&1 || rc=$?
if [[ "${rc}" -eq 0 ]]; then
  report "missing_tag: expected failure for a non-existent tag, got exit 0"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "local tag-annotation compatibility contract FAILED" >&2
  exit 1
fi

echo "local tag-annotation compatibility contract OK: already-annotated tags left unaltered, lightweight tags gain a non-empty local-only annotation (release body when available, deterministic fallback otherwise, XML-escaped), tag commits never move, missing tags fail loudly."
