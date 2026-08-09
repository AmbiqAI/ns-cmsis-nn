#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for .github/workflows/build_publish_docker.yml's
# tooling-vs-source checkout separation (AmbiqAI/ns-cmsis-nn#228, live
# recovery run 31333176320, defect 1).
#
# A recovery build for a historical tag (e.g. v7.29.2) must resolve image
# tags using scripts/ci/resolve_image_tags.sh from the CURRENT, merged
# workflow tooling -- that historical tag predates the helper script, so a
# checkout pinned to `source_ref` *before* running it fails with "No such
# file or directory". The Docker build context/Dockerfile, in turn, must
# come entirely from the pinned historical `source_ref` checkout, never
# from the unpinned tooling checkout.
#
# This is pure GitHub Actions YAML step-ordering, not executable script
# logic, so it isn't unit-testable by invoking the workflow directly. This
# test instead performs a structural assertion on the workflow file itself:
# it locates the build-and-push job's checkout/resolve/build steps by name
# and asserts their relative order and `ref:` pinning, entirely offline
# with no YAML parser dependency (grep/awk only, matching this repo's
# existing lightweight-bash test conventions).

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

  # The build step must not itself perform another checkout (i.e. there
  # should be exactly two actions/checkout@ uses in the whole file).
  checkout_count="$(grep -c 'uses: actions/checkout@' "${WORKFLOW}")"
  if [[ "${checkout_count}" -ne 2 ]]; then
    report "expected exactly 2 actions/checkout@ steps (tooling + pinned), found ${checkout_count}"
  fi
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "build_publish_docker.yml checkout-order contract FAILED" >&2
  exit 1
fi

echo "build_publish_docker.yml checkout-order contract OK: unpinned tooling checkout runs (and resolves image tags) before the source_ref-pinned checkout, which in turn precedes the Docker build."
