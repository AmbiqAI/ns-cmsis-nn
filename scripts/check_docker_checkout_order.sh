#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for .github/workflows/build_publish_docker.yml's
# tooling-vs-source checkout separation (AmbiqAI/ns-cmsis-nn#228, live
# recovery runs 31333176320 (defect 1, "No such file or directory") and
# 31335858426 job 93301489901 (defect 1b, historical vcpkg-bootstrap
# Dockerfile shadowing the current hardened one)).
#
# A recovery build for a historical tag (e.g. v7.29.2) must resolve image
# tags using scripts/ci/resolve_image_tags.sh from the CURRENT, merged
# workflow tooling -- that historical tag predates the helper script, so a
# checkout pinned to `source_ref` *before* running it fails with "No such
# file or directory".
#
# Ordering the two checkouts correctly is NOT sufficient on its own: run
# 31335858426 had the correct tooling-before-pinned ORDER (per this test's
# original version) and still failed, because both checkouts targeted the
# SAME directory (no `path:`) -- the second (historical, pinned) checkout
# silently overwrote the first, replacing the CURRENT, hardened
# `.devcontainer/Dockerfile` (fixed aka.ms vcpkg-bootstrap breakage) with
# v7.29.2's historical (broken) copy of the same file, which then failed
# the image build with "vcpkg: command not found" (exit 127). This test
# therefore also asserts the two checkouts use DIFFERENT, non-empty
# `path:` values, and that the "Build Docker Image" step's `-f` Dockerfile
# argument and build-context argument resolve under those two DIFFERENT
# paths respectively (Dockerfile from the tooling path, context from the
# pinned/historical path) -- i.e. that the Docker build context and
# Dockerfile provenance are never the same tree.
#
# This is pure GitHub Actions YAML structure, not executable script logic,
# so it isn't unit-testable by invoking the workflow directly. This test
# instead performs structural assertions on the workflow file itself:
# it locates the build-and-push job's checkout/resolve/build steps by name
# and asserts their relative order, `path:`/`ref:` values, and the build
# step's Dockerfile/context arguments, entirely offline with no YAML
# parser dependency (grep/awk only, matching this repo's existing
# lightweight-bash test conventions). A synthetic "broken" fixture,
# mirroring the pre-fix (same-directory, no `path:`) workflow, is also
# checked to prove this test would have caught the live defect.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW="${REPO}/.github/workflows/build_publish_docker.yml"

fail=0
report() {
  echo "FAIL: $1" >&2
  fail=1
}

# line_of <pattern> -- first matching line number, or empty if absent.
line_of() {
  grep -n -m1 -- "$1" "${WORKFLOW}" | cut -d: -f1
}

tooling_checkout_line="$(line_of '- name: Checkout Repository (workflow tooling)')"
resolve_step_line="$(line_of '- name: Resolve image tags')"
pinned_checkout_line="$(line_of '- name: Checkout Repository (pinned to source_ref)')"
build_step_line="$(line_of '- name: Build Docker Image')"

if [[ -z "${tooling_checkout_line}" ]]; then
  report "could not find the 'Checkout Repository (workflow tooling)' step"
fi
if [[ -z "${resolve_step_line}" ]]; then
  report "could not find the 'Resolve image tags' step"
fi
if [[ -z "${pinned_checkout_line}" ]]; then
  report "could not find the 'Checkout Repository (pinned to source_ref)' step"
fi
if [[ -z "${build_step_line}" ]]; then
  report "could not find the 'Build Docker Image' step"
fi

if [[ "${fail}" -eq 0 ]]; then
  # Ordering: tooling checkout -> resolve -> pinned checkout -> build.
  if ! [[ "${tooling_checkout_line}" -lt "${resolve_step_line}" ]]; then
    report "'workflow tooling' checkout must come BEFORE 'Resolve image tags' (found tooling@${tooling_checkout_line}, resolve@${resolve_step_line})"
  fi
  if ! [[ "${resolve_step_line}" -lt "${pinned_checkout_line}" ]]; then
    report "'Resolve image tags' must come BEFORE the pinned 'source_ref' checkout (found resolve@${resolve_step_line}, pinned@${pinned_checkout_line}) -- otherwise a historical source_ref checkout can shadow the current tooling before the helper script runs"
  fi
  if ! [[ "${pinned_checkout_line}" -lt "${build_step_line}" ]]; then
    report "pinned 'source_ref' checkout must come BEFORE 'Build Docker Image' (found pinned@${pinned_checkout_line}, build@${build_step_line}) -- the Docker build context must come from the immutable historical source, not the tooling checkout"
  fi

  # The tooling checkout must NOT pin `ref:` to inputs.source_ref -- it must
  # float to whatever triggered the run (github.sha by default), so it
  # always has the current merged helper scripts even when recovering an
  # old tag.
  tooling_block="$(sed -n "${tooling_checkout_line},$((resolve_step_line - 1))p" "${WORKFLOW}")"
  if grep -q 'ref:' <<< "${tooling_block}"; then
    report "the 'workflow tooling' checkout step must not pin 'ref:' -- found one, which would reintroduce defect 1 for historical source_ref values"
  fi

  # The pinned checkout MUST pin ref to inputs.source_ref (falling back to
  # github.sha for normal dispatch/schedule runs).
  pinned_block="$(sed -n "${pinned_checkout_line},$((build_step_line - 1))p" "${WORKFLOW}")"
  # shellcheck disable=SC2016 # literal match against the workflow's YAML text, not shell expansion
  if ! grep -q 'ref: \${{ inputs.source_ref || github.sha }}' <<< "${pinned_block}"; then
    report "the 'pinned to source_ref' checkout step must set ref: \${{ inputs.source_ref || github.sha }}"
  fi

  # The build-and-push job must not itself perform another checkout (i.e.
  # there should be exactly two actions/checkout@ uses within THAT job).
  # Scoped to the job block (not the whole file) so this doesn't produce a
  # brittle false negative if an unrelated job elsewhere in the file
  # legitimately adds its own checkout step in the future.
  job_start_line="$(line_of '^  build-and-push:')"
  if [[ -z "${job_start_line}" ]]; then
    report "could not find the 'build-and-push:' job definition"
    job_start_line=1
  fi
  job_end_line="$(awk -v start="${job_start_line}" \
    'NR > start && /^  [A-Za-z0-9_-]+:/ { print NR; exit }' "${WORKFLOW}")"
  if [[ -z "${job_end_line}" ]]; then
    job_end_line="$(wc -l < "${WORKFLOW}")"
  else
    job_end_line=$((job_end_line - 1))
  fi
  job_block="$(sed -n "${job_start_line},${job_end_line}p" "${WORKFLOW}")"
  checkout_count="$(grep -c 'uses: actions/checkout@' <<< "${job_block}")"
  if [[ "${checkout_count}" -ne 2 ]]; then
    report "expected exactly 2 actions/checkout@ steps in the build-and-push job (tooling + pinned), found ${checkout_count}"
  fi

  # --- Path isolation (live defect 1b, run 31335858426 job 93301489901) ---
  # Correct step order is NOT sufficient: both checkouts must also target
  # DIFFERENT directories via `path:`, or the second (historical) checkout
  # silently clobbers the first (current tooling) -- including the
  # CURRENT, hardened .devcontainer/Dockerfile -- before the Docker build
  # ever runs.
  tooling_path="$(grep -m1 -oE 'path: *\S+' <<< "${tooling_block}" | awk '{print $2}')"
  pinned_path="$(grep -m1 -oE 'path: *\S+' <<< "${pinned_block}" | awk '{print $2}')"

  if [[ -z "${tooling_path}" ]]; then
    report "the 'workflow tooling' checkout step must set a non-default 'path:' (e.g. '_tooling') so the pinned historical checkout below cannot silently overwrite it (AmbiqAI/ns-cmsis-nn#228 live recovery run 31335858426 job 93301489901: the historical .devcontainer/Dockerfile shadowed the current hardened one)"
  fi
  if [[ -z "${pinned_path}" ]]; then
    report "the 'pinned to source_ref' checkout step must set a non-default 'path:' (e.g. '_source'), distinct from the tooling checkout's path"
  fi
  if [[ -n "${tooling_path}" && -n "${pinned_path}" && "${tooling_path}" == "${pinned_path}" ]]; then
    report "the tooling and pinned checkouts must use DIFFERENT 'path:' values, found the same value '${tooling_path}' for both -- this is exactly the same-directory clobber that caused live defect 1b"
  fi

  # --- Dockerfile/build-context provenance -----------------------------
  # 'Build Docker Image' must read the Dockerfile from the TOOLING path
  # (current, hardened) and use the PINNED path (immutable historical
  # source) as the build context -- and those two must be different
  # directories, never the same tree, and never the bare repository root
  # (which would silently resolve to whichever checkout happened to run
  # last).
  #
  # Join backslash-continued lines within the step's `run:` block into one
  # logical `docker build ...` invocation, then take: the argument
  # following `-f` (Dockerfile path), and the final bare (non `-f`,
  # non-flag-value) token (the build context).
  build_block_end="$(awk -v start="$((build_step_line + 1))" \
    'NR >= start && /^      - name:/ { print NR; exit }' "${WORKFLOW}")"
  if [[ -z "${build_block_end}" ]]; then
    build_block_end="$(wc -l < "${WORKFLOW}")"
  fi
  build_run_joined="$(sed -n "${build_step_line},$((build_block_end - 1))p" "${WORKFLOW}" \
    | sed -e 's/[[:space:]]*\\$//' \
    | tr '\n' ' ')"
  docker_build_invocation="$(grep -oE 'docker build.*' <<< "${build_run_joined}" | head -n1)"

  dockerfile_arg="$(grep -oE -- '-f +[^ ]+' <<< "${docker_build_invocation}" | head -n1 | awk '{print $2}')"
  context_arg="$(awk '{print $NF}' <<< "${docker_build_invocation}")"

  if [[ -z "${dockerfile_arg}" ]]; then
    report "'Build Docker Image' must pass an explicit '-f <path>/Dockerfile' argument"
  elif [[ -n "${tooling_path}" && "${dockerfile_arg}" != "${tooling_path}"/* ]]; then
    report "'Build Docker Image's -f Dockerfile argument ('${dockerfile_arg}') must be read from the tooling checkout path ('${tooling_path}') -- i.e. the CURRENT, hardened Dockerfile, not a historical copy"
  fi

  if [[ -z "${context_arg}" ]]; then
    report "'Build Docker Image' must pass an explicit build-context directory argument"
  elif [[ "${context_arg}" == "." || "${context_arg}" == "./" ]]; then
    report "'Build Docker Image's build context must not be the bare repository root ('.') -- it must explicitly be the pinned historical source path ('${pinned_path}'), so it can never accidentally resolve to whichever checkout happened to run last"
  elif [[ -n "${pinned_path}" && "${context_arg}" != "${pinned_path}" ]]; then
    report "'Build Docker Image's build-context argument ('${context_arg}') must be the pinned historical source path ('${pinned_path}'), not the tooling path or repository root"
  fi

  if [[ -n "${dockerfile_arg}" && -n "${context_arg}" && "${dockerfile_arg}" == "${context_arg}"* && -n "${tooling_path}" && "${context_arg}" == "${tooling_path}" ]]; then
    report "Dockerfile path and build context must come from two DIFFERENT trees (tooling vs. historical source); found both resolving under '${context_arg}'"
  fi
fi

# --- Fixture: prove this test rejects the pre-fix, same-directory shape ---
# Mirrors the ACTUAL live-failing shape from run 31335858426 job
# 93301489901: correct step order, but both checkouts default to the same
# (repo-root) directory, and the build step reads `.devcontainer/Dockerfile`
# with build context `.` -- i.e. no `path:`/tree separation at all. A test
# that only checked step order (as this file did before AmbiqAI/ns-cmsis-nn#228's
# live-recovery follow-up) would incorrectly PASS this broken fixture.
check_workflow_text() {
  local text="$1" label="$2"
  local t_line r_line p_line b_line t_block p_block b_block t_path p_path dfile ctx
  local local_fail=0

  t_line="$(grep -n -m1 -- '- name: Checkout Repository (workflow tooling)' <<< "${text}" | cut -d: -f1)"
  r_line="$(grep -n -m1 -- '- name: Resolve image tags' <<< "${text}" | cut -d: -f1)"
  p_line="$(grep -n -m1 -- '- name: Checkout Repository (pinned to source_ref)' <<< "${text}" | cut -d: -f1)"
  b_line="$(grep -n -m1 -- '- name: Build Docker Image' <<< "${text}" | cut -d: -f1)"
  [[ -n "${t_line}" && -n "${r_line}" && -n "${p_line}" && -n "${b_line}" ]] || { echo "FIXTURE_ERROR:${label}: missing expected step" >&2; return 2; }

  t_block="$(sed -n "${t_line},$((r_line - 1))p" <<< "${text}")"
  p_block="$(sed -n "${p_line},$((b_line - 1))p" <<< "${text}")"
  b_block="$(sed -n "${b_line},\$p" <<< "${text}")"

  t_path="$(grep -m1 -oE 'path: *\S+' <<< "${t_block}" | awk '{print $2}')"
  p_path="$(grep -m1 -oE 'path: *\S+' <<< "${p_block}" | awk '{print $2}')"
  [[ -n "${t_path}" ]] || local_fail=1
  [[ -n "${p_path}" ]] || local_fail=1
  [[ -n "${t_path}" && -n "${p_path}" && "${t_path}" != "${p_path}" ]] || local_fail=1

  dfile="$(grep -oE -- '-f +[^ ]+' <<< "${b_block}" | head -n1 | awk '{print $2}')"
  ctx="$(printf '%s' "${b_block}" | tr '\n' ' ' | grep -oE 'docker build.*' | head -n1 | awk '{print $NF}')"
  [[ "${ctx}" != "." && "${ctx}" != "./" ]] || local_fail=1
  [[ -n "${t_path}" && "${dfile}" == "${t_path}"/* ]] || local_fail=1

  if [[ "${local_fail}" -eq 0 ]]; then
    echo "FIXTURE_UNEXPECTED_PASS:${label}" >&2
    return 0
  fi
  return 1
}

WORK="${REPO}/build/check_docker_checkout_order_test"
rm -rf "${WORK}"
mkdir -p "${WORK}"
broken_fixture_file="${WORK}/broken_fixture.yml"
cat > "${broken_fixture_file}" <<'EOF'
jobs:
  build-and-push:
    steps:
      - name: Checkout Repository (workflow tooling)
        uses: actions/checkout@v6

      - name: Resolve image tags
        run: bash scripts/ci/resolve_image_tags.sh

      - name: Checkout Repository (pinned to source_ref)
        uses: actions/checkout@v6
        with:
          ref: ${{ inputs.source_ref || github.sha }}

      - name: Build Docker Image
        run: |
          docker build -f .devcontainer/Dockerfile .
  other-job:
    steps: []
EOF
broken_fixture="$(cat "${broken_fixture_file}")"

if check_workflow_text "${broken_fixture}" "pre-fix-same-directory"; then
  report "the same-directory (no path: separation) pre-fix fixture unexpectedly PASSED the path-isolation contract -- this test would not have caught live defect 1b (run 31335858426, job 93301489901)"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "build_publish_docker.yml checkout-order contract FAILED" >&2
  exit 1
fi

echo "build_publish_docker.yml checkout-order contract OK: unpinned tooling checkout runs (and resolves image tags) before the source_ref-pinned checkout, which in turn precedes the Docker build."
