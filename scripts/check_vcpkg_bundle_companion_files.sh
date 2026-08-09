#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for .devcontainer/Dockerfile's vcpkg-artifacts standalone
# bundle bootstrap (AmbiqAI/ns-cmsis-nn#228, third live recovery run
# 31337539282 job 93305784722, discovered AFTER the two-tree
# tooling/source checkout fix from #231 let publish-ci-image reach the
# actual Docker build).
#
# That run proved the pinned vcpkg-glibc BINARY alone is not sufficient for
# artifacts commands (`vcpkg x-update-registry --all`, `vcpkg activate`):
# those commands shell out to the bundled Node-based vcpkg-artifacts
# tooling described by scripts/vcpkg-tools.json, which -- along with
# vcpkg-artifacts.mjs, .vcpkg-root, and the scripts/ and triplets/ trees --
# only ships in the STANDALONE BUNDLE release asset (PR #229 downloaded
# only vcpkg-glibc). Without it:
#   /opt/vcpkg/scripts/vcpkg-tools.json: No such file or directory
#
# The fix downloads+checksums+extracts vcpkg-standalone-bundle.tar.gz into
# VCPKG_ROOT FIRST, then downloads+checksums+installs the pinned
# vcpkg-glibc binary OVER it, then asserts (at build time, before any
# artifacts command runs) that the bundle's companion files actually
# landed. This test performs structural assertions on the Dockerfile
# itself -- ARG pins present and well-formed, extraction-before-binary
# ordering, and the build-time companion-file assertion existing and
# running strictly between the bootstrap RUN step and the first artifacts
# command (`x-update-registry`) -- entirely offline (grep/awk only,
# matching this repo's existing lightweight-bash test conventions, e.g.
# scripts/check_docker_checkout_order.sh). A synthetic "broken" fixture,
# mirroring the pre-fix (vcpkg-glibc-only, no bundle) Dockerfile shape, is
# also checked to prove this test would have caught the live defect.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DOCKERFILE="${REPO}/.devcontainer/Dockerfile"

fail=0
report() {
  echo "FAIL: $1" >&2
  fail=1
}

# line_of <pattern> -- first matching line number among non-comment lines,
# or empty if absent. Comment lines (e.g. the historical-note prose above
# the actual steps) are excluded so mentioning a filename/command in prose
# doesn't get mistaken for the real step. Uses awk (not grep -n | grep) so
# the pattern is matched against the ORIGINAL line, preserving anchors
# like '^ARG'.
line_of() {
  awk -v pat="$1" '$0 !~ /^[[:space:]]*#/ && $0 ~ pat { print NR; exit }' "${DOCKERFILE}"
}

bundle_sha_arg_line="$(line_of '^[[:space:]]*ARG VCPKG_STANDALONE_BUNDLE_SHA256=')"
bundle_version_arg_line="$(line_of '^[[:space:]]*ARG VCPKG_STANDALONE_BUNDLE_VERSION=')"
bootstrap_run_line="$(line_of 'vcpkg-standalone-bundle\.tar\.gz')"
tool_run_line="$(line_of '/vcpkg-glibc"')"
companion_check_line="$(line_of 'vcpkg-artifacts\.mjs')"
update_registry_line="$(line_of 'x-update-registry --all')"

if [[ -z "${bundle_version_arg_line}" ]]; then
  report "could not find an 'ARG VCPKG_STANDALONE_BUNDLE_VERSION=' pin in ${DOCKERFILE}"
fi
if [[ -z "${bundle_sha_arg_line}" ]]; then
  report "could not find an 'ARG VCPKG_STANDALONE_BUNDLE_SHA256=' pin in ${DOCKERFILE}"
fi
if [[ -z "${bootstrap_run_line}" ]]; then
  report "could not find a step downloading vcpkg-standalone-bundle.tar.gz"
fi
if [[ -z "${tool_run_line}" ]]; then
  report "could not find a step downloading the vcpkg-glibc binary"
fi
if [[ -z "${companion_check_line}" ]]; then
  report "could not find a build-time assertion referencing vcpkg-artifacts.mjs"
fi
if [[ -z "${update_registry_line}" ]]; then
  report "could not find the 'vcpkg x-update-registry --all' step"
fi

if [[ "${fail}" -eq 0 ]]; then
  # The bundle SHA must be a well-formed, non-placeholder 64-hex-char
  # sha256 digest (matches the GitHub release asset's published digest).
  bundle_sha_value="$(sed -n "${bundle_sha_arg_line}p" "${DOCKERFILE}" | grep -oE '[0-9a-f]{64}' || true)"
  if [[ -z "${bundle_sha_value}" ]]; then
    report "VCPKG_STANDALONE_BUNDLE_SHA256 must be a 64-hex-character sha256 digest"
  fi

  # The bundle version must be a non-empty, non-placeholder release tag.
  bundle_version_value="$(sed -n "${bundle_version_arg_line}p" "${DOCKERFILE}" | sed -E 's/^[[:space:]]*ARG VCPKG_STANDALONE_BUNDLE_VERSION=//')"
  if [[ -z "${bundle_version_value}" || "${bundle_version_value}" == "unknown" ]]; then
    report "VCPKG_STANDALONE_BUNDLE_VERSION must be pinned to a real release tag, found '${bundle_version_value}'"
  fi

  # Ordering: bundle extraction must happen BEFORE the vcpkg-glibc binary
  # is installed (per this fix's design -- bundle first, binary on top),
  # both of which must happen BEFORE the build-time companion-file
  # assertion, which itself must happen BEFORE the first artifacts command.
  if ! [[ "${bootstrap_run_line}" -lt "${tool_run_line}" ]]; then
    report "the vcpkg-standalone-bundle.tar.gz extraction must come BEFORE the vcpkg-glibc binary install (found bundle@${bootstrap_run_line}, binary@${tool_run_line})"
  fi
  if ! [[ "${tool_run_line}" -lt "${companion_check_line}" ]]; then
    report "the vcpkg-glibc binary install must come BEFORE the companion-file build-time assertion (found binary@${tool_run_line}, assertion@${companion_check_line})"
  fi
  if ! [[ "${companion_check_line}" -lt "${update_registry_line}" ]]; then
    report "the companion-file build-time assertion must come BEFORE 'vcpkg x-update-registry --all' (found assertion@${companion_check_line}, update-registry@${update_registry_line}) -- otherwise a missing-bundle regression surfaces as an oblique internal vcpkg error instead of a specific, actionable message"
  fi

  # The bootstrap step must actually checksum-verify the downloaded
  # bundle before extracting it (not just download-and-trust).
  bootstrap_block="$(sed -n "${bootstrap_run_line},$((tool_run_line + 3))p" "${DOCKERFILE}")"
  if ! grep -q 'sha256sum -c' <<< "${bootstrap_block}"; then
    report "the vcpkg-standalone-bundle.tar.gz download must be checksum-verified (sha256sum -c) before extraction"
  fi
  if ! grep -q 'tar -x' <<< "${bootstrap_block}"; then
    report "the vcpkg-standalone-bundle.tar.gz download must be extracted (tar -x...) into VCPKG_ROOT"
  fi

  # The companion-file assertion must check for ALL the files this fix
  # actually needs downstream: the bundle marker, the artifacts-tools
  # manifest, and the vcpkg-artifacts entry point. Find the nearest 'RUN'
  # step start at or above companion_check_line, so the block covers the
  # whole assertion step regardless of exact line offsets.
  companion_run_start="$(awk -v end="${companion_check_line}" \
    'NR <= end && /^RUN / { start = NR } END { print start }' "${DOCKERFILE}")"
  if [[ -z "${companion_run_start}" ]]; then
    companion_run_start="${companion_check_line}"
  fi
  companion_block="$(sed -n "${companion_run_start},$((update_registry_line - 1))p" "${DOCKERFILE}")"
  for required in '.vcpkg-root' 'scripts/vcpkg-tools.json' 'vcpkg-artifacts.mjs'; do
    if ! grep -qF -- "${required}" <<< "${companion_block}"; then
      report "the build-time companion-file assertion must check for '${required}'"
    fi
  done
  # triplets/ is required companion content too (a directory, not a file --
  # checked separately since the Dockerfile uses `-d` for it), so assert it
  # explicitly instead of relying on the file-only loop above.
  if ! grep -qF -- 'triplets' <<< "${companion_block}"; then
    report "the build-time companion-file assertion must check for the 'triplets' directory"
  fi
fi

# --- Fixture: prove this test rejects the pre-fix, glibc-binary-only shape ---
# Mirrors the ACTUAL live-failing shape from run 31337539282 job
# 93305784722: only the vcpkg-glibc binary is downloaded, no standalone
# bundle, no companion-file assertion. A test that only checked for the
# vcpkg-glibc download (as this file did before AmbiqAI/ns-cmsis-nn#228's
# third live-recovery follow-up) would incorrectly PASS this broken
# fixture.
check_dockerfile_text() {
  local text="$1" label="$2"
  local b_ver_line b_sha_line boot_line tool_line comp_line reg_line
  local local_fail=0

  b_ver_line="$(grep -n -m1 -- '^ARG VCPKG_STANDALONE_BUNDLE_VERSION=' <<< "${text}" | cut -d: -f1)"
  b_sha_line="$(grep -n -m1 -- '^ARG VCPKG_STANDALONE_BUNDLE_SHA256=' <<< "${text}" | cut -d: -f1)"
  boot_line="$(grep -n -m1 -- 'vcpkg-standalone-bundle\.tar\.gz' <<< "${text}" | cut -d: -f1)"
  tool_line="$(grep -n -m1 -- '/vcpkg-glibc"' <<< "${text}" | cut -d: -f1)"
  comp_line="$(grep -n -m1 -- 'vcpkg-artifacts\.mjs' <<< "${text}" | cut -d: -f1)"
  reg_line="$(grep -n -m1 -- 'x-update-registry --all' <<< "${text}" | cut -d: -f1)"

  [[ -n "${b_ver_line}" && -n "${b_sha_line}" && -n "${boot_line}" && -n "${comp_line}" ]] || local_fail=1
  [[ -n "${tool_line}" && -n "${reg_line}" ]] || { echo "FIXTURE_ERROR:${label}: missing expected step" >&2; return 2; }

  if [[ "${local_fail}" -eq 0 ]]; then
    echo "FIXTURE_UNEXPECTED_PASS:${label}" >&2
    return 0
  fi
  return 1
}

WORK="${REPO}/build/check_vcpkg_bundle_companion_files_test"
rm -rf "${WORK}"
mkdir -p "${WORK}"
broken_fixture_file="${WORK}/broken_fixture.dockerfile"
cat > "${broken_fixture_file}" <<'EOF'
ENV VCPKG_ROOT=/opt/vcpkg \
    VCPKG_DOWNLOADS=/opt/vcpkg/downloads \
    VCPKG_DISABLE_METRICS=1

ARG VCPKG_TOOL_VERSION=2026-07-27
ARG VCPKG_TOOL_SHA256=7e97ef6bcd58f74d079f40d086b801a0222c5d15e4ea0d8d507a538033493d04

RUN set -eux; \
      mkdir -p "${VCPKG_ROOT}" /opt/ns-cmsis-nn; \
      curl -fsSL "https://github.com/microsoft/vcpkg-tool/releases/download/${VCPKG_TOOL_VERSION}/vcpkg-glibc" \
            -o "${VCPKG_ROOT}/vcpkg"; \
      echo "${VCPKG_TOOL_SHA256}  ${VCPKG_ROOT}/vcpkg" | sha256sum -c -; \
      chmod +x "${VCPKG_ROOT}/vcpkg"; \
      echo "${VCPKG_TOOL_VERSION}" > "${VCPKG_ROOT}/vcpkg-version.txt"

ENV PATH="${VCPKG_ROOT}:${PATH}"

RUN vcpkg version

COPY vcpkg-configuration.json /opt/ns-cmsis-nn/vcpkg-configuration.json
RUN set -eux; \
      cd /opt/ns-cmsis-nn; \
      "${VCPKG_ROOT}/vcpkg" x-update-registry --all; \
      "${VCPKG_ROOT}/vcpkg" activate \
            --downloads-root="${VCPKG_DOWNLOADS}" \
            --json="${NS_CMSIS_NN_VCPKG_ENV}"; \
      jq -e '.tools' "${NS_CMSIS_NN_VCPKG_ENV}" >/dev/null; \
      rm -rf "${VCPKG_DOWNLOADS}"
EOF

set +e
check_dockerfile_text "$(cat "${broken_fixture_file}")" "vcpkg-glibc-only-no-bundle" >"${WORK}/fixture_check.out" 2>&1
fixture_rc=$?
set -e

if [[ "${fixture_rc}" -eq 0 ]]; then
  cat "${WORK}/fixture_check.out" >&2
  report "the broken (pre-fix, vcpkg-glibc-only) fixture unexpectedly PASSED -- this test would not have caught the live defect from run 31337539282 job 93305784722"
elif [[ "${fixture_rc}" -eq 2 ]]; then
  cat "${WORK}/fixture_check.out" >&2
  report "broken fixture check errored instead of cleanly failing -- see FIXTURE_ERROR above"
fi

rm -rf "${WORK}"

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "vcpkg-artifacts standalone-bundle companion-files contract OK: the bundle is pinned/checksummed/extracted into VCPKG_ROOT before the vcpkg-glibc binary is installed over it, a build-time assertion confirms .vcpkg-root/scripts/vcpkg-tools.json/vcpkg-artifacts.mjs landed before any artifacts command runs, and the pre-fix vcpkg-glibc-only shape is correctly rejected."
