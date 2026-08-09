#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for scripts/ci/resolve_image_tags.sh -- the Docker
# image-tag/`:latest`-alias precedence logic used by
# .github/workflows/build_publish_docker.yml.
#
# This is the regression guard for AmbiqAI/ns-cmsis-nn#228's second
# blocker: build_publish_docker.yml used to prioritize
# `github.event_name == 'workflow_dispatch'` + `github.event.inputs.*` over
# the unified `inputs.*` context. That silently broke whenever this
# workflow was invoked as a reusable `workflow_call` from another
# workflow_dispatch-triggered run (release.yml's `recover_tag` recovery
# path): the recovery caller's `image_tag`/`publish_latest` `with:` values
# were ignored, every recovery run emitted only `:latest`, and (with
# `publish_latest` also ignored) it clobbered the `:latest` alias with an
# old recovered tag's image.
#
# Runs entirely locally: no GitHub Actions context, no network access.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${REPO}/scripts/ci/resolve_image_tags.sh"
WORK="${REPO}/build/release_image_tags_test"
REGISTRY_IMAGE="ghcr.io/ambiqai/ns-cmsis-nn-ci"

rm -rf "${WORK}"
mkdir -p "${WORK}"

fail=0

report() {
  echo "FAIL: $1" >&2
  fail=1
}

# run_case <name> <event_name> <input_image_tag> <input_publish_latest> \
#          <expected_image_tag> <expected_publish_latest> <expected_tag...>
run_case() {
  local name="$1" event_name="$2" input_image_tag="$3" input_publish_latest="$4"
  local expected_image_tag="$5" expected_publish_latest="$6"
  shift 6
  local expected_tags=("$@")

  local out="${WORK}/${name}.out"
  rm -f "${out}"
  bash "${SCRIPT}" "${event_name}" "${input_image_tag}" "${input_publish_latest}" \
    "${REGISTRY_IMAGE}" "${out}"

  local actual_image_tag actual_publish_latest
  actual_image_tag="$(grep -m1 '^image_tag=' "${out}" | cut -d= -f2-)"
  actual_publish_latest="$(grep -m1 '^publish_latest=' "${out}" | cut -d= -f2-)"

  local actual_tags=()
  local in_block=0
  while IFS= read -r line; do
    if [[ "${line}" == "docker_tags<<EOF" ]]; then
      in_block=1
      continue
    fi
    if [[ "${in_block}" -eq 1 && "${line}" == "EOF" ]]; then
      in_block=0
      continue
    fi
    if [[ "${in_block}" -eq 1 ]]; then
      actual_tags+=("${line}")
    fi
  done < "${out}"

  if [[ "${actual_image_tag}" != "${expected_image_tag}" ]]; then
    report "${name}: image_tag='${actual_image_tag}', expected '${expected_image_tag}'"
  fi
  if [[ "${actual_publish_latest}" != "${expected_publish_latest}" ]]; then
    report "${name}: publish_latest='${actual_publish_latest}', expected '${expected_publish_latest}'"
  fi
  if [[ "${#actual_tags[@]}" -ne "${#expected_tags[@]}" ]]; then
    report "${name}: docker_tags=(${actual_tags[*]}), expected (${expected_tags[*]})"
  else
    for i in "${!expected_tags[@]}"; do
      if [[ "${actual_tags[$i]}" != "${expected_tags[$i]}" ]]; then
        report "${name}: docker_tags[$i]='${actual_tags[$i]}', expected '${expected_tags[$i]}'"
      fi
    done
  fi
}

# --- Case 1: recovery-mode reusable call ------------------------------------
# release.yml's publish-ci-image job, dispatched via workflow_dispatch with
# recover_tag=v7.29.2, calls this workflow with
# `with: { image_tag: v7.29.2, publish_latest: false }`. Because it's a
# reusable call from a workflow_dispatch-triggered caller, github.event_name
# here is "workflow_dispatch" (inherited) even though this is really a
# workflow_call -- the fix must NOT let that fool it into reading
# github.event.inputs.* (which would be empty/wrong here). This is the exact
# regression this test guards against.
run_case "recovery_reusable_call" \
  "workflow_dispatch" "v7.29.2" "false" \
  "v7.29.2" "false" \
  "${REGISTRY_IMAGE}:v7.29.2"

# --- Case 2: genuine new-release reusable call ------------------------------
# release.yml's publish-ci-image job for a brand new release-please release:
# `with: { image_tag: v7.30.0, publish_latest: true }`.
run_case "new_release_reusable_call" \
  "workflow_dispatch" "v7.30.0" "true" \
  "v7.30.0" "true" \
  "${REGISTRY_IMAGE}:v7.30.0" "${REGISTRY_IMAGE}:latest"

# --- Case 3: genuine new-release reusable call triggered by a push ---------
# Same as case 2 but the outer caller was a push (not workflow_dispatch) --
# must behave identically regardless of what github.event_name happens to be.
run_case "new_release_reusable_call_push" \
  "push" "v7.30.0" "true" \
  "v7.30.0" "true" \
  "${REGISTRY_IMAGE}:v7.30.0" "${REGISTRY_IMAGE}:latest"

# --- Case 4: direct manual dispatch, explicit inputs ------------------------
# A human runs build_publish_docker.yml's own workflow_dispatch trigger and
# fills in the form.
run_case "direct_dispatch_explicit" \
  "workflow_dispatch" "v9.9.9" "true" \
  "v9.9.9" "true" \
  "${REGISTRY_IMAGE}:v9.9.9" "${REGISTRY_IMAGE}:latest"

# --- Case 5: direct manual dispatch, defaults -------------------------------
# A human runs the workflow_dispatch trigger without changing anything; GitHub
# resolves inputs.image_tag/inputs.publish_latest to that trigger's declared
# defaults (image_tag: 'latest', publish_latest: true). Must not double up
# the :latest tag.
run_case "direct_dispatch_defaults" \
  "workflow_dispatch" "latest" "true" \
  "latest" "true" \
  "${REGISTRY_IMAGE}:latest"

# --- Case 6: schedule (weekly cron) -----------------------------------------
# No `inputs` object exists at all on a schedule trigger.
run_case "schedule" \
  "schedule" "" "" \
  "latest" "true" \
  "${REGISTRY_IMAGE}:latest"

# --- Case 7: reusable call using workflow_call's own declared defaults ------
# A hypothetical caller that invokes `uses: .../build_publish_docker.yml`
# with no `with:` at all falls back to the workflow_call trigger's own
# defaults (image_tag: 'latest', publish_latest: false).
run_case "reusable_call_defaults" \
  "workflow_dispatch" "" "" \
  "latest" "false" \
  "${REGISTRY_IMAGE}:latest"

if [[ "${fail}" -ne 0 ]]; then
  echo "release image-tag resolution contract FAILED" >&2
  exit 1
fi

echo "release image-tag resolution contract OK: recovery/new-release/direct-dispatch/schedule cases all resolve image_tag, publish_latest, and docker_tags correctly."
