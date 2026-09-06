#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

usage() {
  cat <<'EOF'
Usage:
  check_clang_format_changed.sh [--fix] [--base <ref>] [--head <ref>]
  check_clang_format_changed.sh <base-ref> [head-ref]

Examples:
  scripts/check_clang_format_changed.sh --base origin/main
  scripts/check_clang_format_changed.sh --fix --base upstream/main --head HEAD
  scripts/check_clang_format_changed.sh origin/main HEAD

Behavior:
  - Mirrors the CI clang-format workflow file selection.
  - Uses CLANG_FORMAT_BIN when set, else the first clang-format on PATH that
    reports the pinned version, else any 18.x with a warning.
  - In check mode, runs --dry-run --Werror.
  - In fix mode, rewrites the changed files in place.
EOF
}

BASE_REF="${BASE_REF:-}"
HEAD_REF="${HEAD_REF:-HEAD}"
FIX=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      FIX=1
      shift
      ;;
    --base)
      [[ $# -ge 2 ]] || { echo "--base requires a value" >&2; usage >&2; exit 2; }
      BASE_REF="$2"
      shift 2
      ;;
    --head)
      [[ $# -ge 2 ]] || { echo "--head requires a value" >&2; usage >&2; exit 2; }
      HEAD_REF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  BASE_REF="${POSITIONAL[0]}"
fi

if [[ ${#POSITIONAL[@]} -gt 1 ]]; then
  HEAD_REF="${POSITIONAL[1]}"
fi

if [[ -z "${BASE_REF}" ]]; then
  echo "Missing base ref." >&2
  usage >&2
  exit 2
fi

command -v git >/dev/null || { echo "git not found" >&2; exit 3; }

git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null
git rev-parse --verify "${HEAD_REF}^{commit}" >/dev/null

# The enforced formatter is clang-format 18 (the .pre-commit-config.yaml pin);
# formatter major versions disagree on committed files, so the version is
# checked, not assumed. Resolution order: CLANG_FORMAT_BIN if set (CI points
# it at the pip-installed 18.1.8), else the first candidate on PATH reporting
# the pinned version, else any candidate reporting the major.
REQUIRED_CLANG_FORMAT_MAJOR=18
# Point releases inside the major disagree too: 18.1.3, which ubuntu-24.04
# ships, still rejects Include/Internal/arm_conv_opt_common.h after 18.1.8 has
# formatted it. The gate stays on the major so a close-enough local build is
# usable, but a mismatch is called out rather than left to look like a real
# finding (see AmbiqAI/ns-cmsis-nn#394).
PINNED_CLANG_FORMAT_VERSION=18.1.8
CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-}"

clang_format_version_of() {
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" --version 2>/dev/null | head -n 1
}

# Repeated after a failure: a 18.1.x that is not the pin produces findings the
# contributor must not act on, and the first line of output has scrolled away
# by then (see AmbiqAI/ns-cmsis-nn#394).
warn_if_not_pinned() {
  [[ "${raw_version}" == *"${PINNED_CLANG_FORMAT_VERSION}"* ]] && return 0
  echo "warning: ${CLANG_FORMAT_BIN} is not the pinned ${PINNED_CLANG_FORMAT_VERSION}; point releases disagree on committed files." >&2
  echo "warning: install it with: python -m pip install clang-format==${PINNED_CLANG_FORMAT_VERSION}" >&2
}

raw_version=""
if [[ -n "${CLANG_FORMAT_BIN}" ]]; then
  raw_version="$(clang_format_version_of "${CLANG_FORMAT_BIN}")" \
    || { echo "CLANG_FORMAT_BIN=${CLANG_FORMAT_BIN} not found on PATH." >&2; exit 3; }
else
  # Selection is by reported version, not by name: ubuntu-24.04 ships 18.1.3 as
  # clang-format-18, which would otherwise outrank a pinned 18.1.8 sitting
  # earlier on PATH under the bare name.
  fallback_bin=""
  fallback_version=""
  for candidate in clang-format "clang-format-${REQUIRED_CLANG_FORMAT_MAJOR}"; do
    candidate_version="$(clang_format_version_of "${candidate}")" || continue
    if [[ "${candidate_version}" == *"${PINNED_CLANG_FORMAT_VERSION}"* ]]; then
      CLANG_FORMAT_BIN="${candidate}"
      raw_version="${candidate_version}"
      break
    fi
    if [[ -z "${fallback_bin}" && "${candidate_version}" =~ version\ ${REQUIRED_CLANG_FORMAT_MAJOR}\. ]]; then
      fallback_bin="${candidate}"
      fallback_version="${candidate_version}"
    fi
  done
  if [[ -z "${CLANG_FORMAT_BIN}" && -n "${fallback_bin}" ]]; then
    CLANG_FORMAT_BIN="${fallback_bin}"
    raw_version="${fallback_version}"
  fi
fi

if [[ -z "${CLANG_FORMAT_BIN}" ]]; then
  echo "no clang-format ${REQUIRED_CLANG_FORMAT_MAJOR}.x found. Install the pinned build (python -m pip install clang-format==${PINNED_CLANG_FORMAT_VERSION}) or set CLANG_FORMAT_BIN." >&2
  exit 3
fi

found_major=""
if [[ "${raw_version}" =~ version\ ([0-9]+)\. ]]; then
  found_major="${BASH_REMATCH[1]}"
fi
if [[ "${found_major}" != "${REQUIRED_CLANG_FORMAT_MAJOR}" ]]; then
  echo "clang-format major ${found_major:-unknown} found at $(command -v "${CLANG_FORMAT_BIN}"); this repo enforces ${REQUIRED_CLANG_FORMAT_MAJOR}.x." >&2
  echo "Install it with: python -m pip install clang-format==${PINNED_CLANG_FORMAT_VERSION}, then point the script at that copy explicitly:" >&2
  echo "  CLANG_FORMAT_BIN=\"\$(python -c \"import sys,os;print(os.path.dirname(sys.executable))\")/clang-format\" $0 ..." >&2
  exit 3
fi

echo "Using formatter: $(command -v "${CLANG_FORMAT_BIN}") (${raw_version})"
warn_if_not_pinned

# Byte-identical to Arm upstream and rejected by the enforced formatter;
# reformatting them would conflict on every sync, so this gate and the
# pre-commit exclude skip the same two files (see AmbiqAI/ns-cmsis-nn#394).
unformatted_upstream='^(Include/Internal/arm_conv1x1_opt_common\.h|Include/Internal/arm_depthwise_conv_opt_common\.h)$'

changed_files=()
while IFS= read -r file; do
  [[ -n "${file}" ]] || continue
  changed_files+=("${file}")
done < <(
  git diff --name-only --diff-filter=d "${BASE_REF}" "${HEAD_REF}" -- \
    Include \
    Source \
    Tests/UnitTest/Corstone-300 |
  grep -E '\.(c|cc|cpp|h|hpp)$' |
  grep -Ev "${unformatted_upstream}" || true
)

if [[ ${#changed_files[@]} -eq 0 ]]; then
  echo "No changed C/C++ files found between ${BASE_REF} and ${HEAD_REF}."
  exit 0
fi

echo "Checking files changed between ${BASE_REF} and ${HEAD_REF}:"
printf '  %s\n' "${changed_files[@]}"

if [[ ${FIX} -eq 1 ]]; then
  echo
  echo "Applying clang-format in place."
  "${CLANG_FORMAT_BIN}" -i "${changed_files[@]}"
else
  echo
  echo "Running clang-format dry-run check."
  if ! "${CLANG_FORMAT_BIN}" --dry-run --Werror "${changed_files[@]}"; then
    warn_if_not_pinned
    exit 1
  fi
fi

echo
echo "clang-format check completed successfully."
