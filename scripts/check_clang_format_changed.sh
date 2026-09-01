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
  - Uses CLANG_FORMAT_BIN when set, else clang-format-18 when available, else clang-format.
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

# CLANG_FORMAT_BIN pins the formatter explicitly (CI sets it to the pip-installed
# clang-format matching .pre-commit-config.yaml). Without it the probe below picks
# whatever the runner image ships, and formatter major versions disagree on
# existing files -- ubuntu-24.04 puts clang-format-18 on PATH and 18 rejects four
# committed files that 13 through 16 accept.
if [[ -n "${CLANG_FORMAT_BIN:-}" ]]; then
  command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1 || { echo "CLANG_FORMAT_BIN=${CLANG_FORMAT_BIN} not found on PATH." >&2; exit 3; }
elif command -v clang-format-18 >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format-18"
elif command -v clang-format >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format"
else
  echo "no clang-format found. Set CLANG_FORMAT_BIN or put clang-format on PATH." >&2
  exit 3
fi

changed_files=()
while IFS= read -r file; do
  [[ -n "${file}" ]] || continue
  changed_files+=("${file}")
done < <(
  git diff --name-only --diff-filter=d "${BASE_REF}" "${HEAD_REF}" -- \
    Include \
    Source \
    Tests/UnitTest/Corstone-300 |
  grep -E '\.(c|cc|cpp|h|hpp)$' || true
)

if [[ ${#changed_files[@]} -eq 0 ]]; then
  echo "No changed C/C++ files found between ${BASE_REF} and ${HEAD_REF}."
  exit 0
fi

echo "Using formatter: ${CLANG_FORMAT_BIN}"
echo "Checking files changed between ${BASE_REF} and ${HEAD_REF}:"
printf '  %s\n' "${changed_files[@]}"

if [[ ${FIX} -eq 1 ]]; then
  echo
  echo "Applying clang-format in place."
  "${CLANG_FORMAT_BIN}" -i "${changed_files[@]}"
else
  echo
  echo "Running clang-format dry-run check."
  "${CLANG_FORMAT_BIN}" --dry-run --Werror "${changed_files[@]}"
fi

echo
echo "clang-format check completed successfully."
