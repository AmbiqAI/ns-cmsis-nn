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
  - Uses CLANG_FORMAT_BIN when set, else clang-format-16, else clang-format; requires major 16.
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

# The enforced formatter is clang-format 16 (the .pre-commit-config.yaml pin);
# formatter major versions disagree on committed files (the 18.1.3 that
# ubuntu-24.04 ships rejects four that 13 through 16 accept; newer 18.x point
# releases flag more), so the version is checked, not assumed. Resolution
# order: CLANG_FORMAT_BIN if set (CI points it at the pip-installed 16.0.6),
# else clang-format-16, else a bare clang-format that reports major 16.
REQUIRED_CLANG_FORMAT_MAJOR=16
if [[ -n "${CLANG_FORMAT_BIN:-}" ]]; then
  command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1 || { echo "CLANG_FORMAT_BIN=${CLANG_FORMAT_BIN} not found on PATH." >&2; exit 3; }
elif command -v "clang-format-${REQUIRED_CLANG_FORMAT_MAJOR}" >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format-${REQUIRED_CLANG_FORMAT_MAJOR}"
elif command -v clang-format >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format"
else
  echo "no clang-format found. Install clang-format ${REQUIRED_CLANG_FORMAT_MAJOR} (pip install clang-format==16.0.6) or set CLANG_FORMAT_BIN." >&2
  exit 3
fi
raw_version="$("${CLANG_FORMAT_BIN}" --version 2>/dev/null | head -n 1 || true)"
found_major=""
if [[ "${raw_version}" =~ version\ ([0-9]+)\. ]]; then
  found_major="${BASH_REMATCH[1]}"
fi
if [[ "${found_major}" != "${REQUIRED_CLANG_FORMAT_MAJOR}" ]]; then
  echo "clang-format major ${found_major:-unknown} found at $(command -v "${CLANG_FORMAT_BIN}"); this repo enforces ${REQUIRED_CLANG_FORMAT_MAJOR}.x." >&2
  echo "Install it with: python -m pip install clang-format==16.0.6, then point the script at that copy explicitly:" >&2
  echo "  CLANG_FORMAT_BIN=\"\$(python -c \"import sys,os;print(os.path.dirname(sys.executable))\")/clang-format\" $0 ..." >&2
  exit 3
fi

echo "Using formatter: $(command -v "${CLANG_FORMAT_BIN}") (${raw_version})"

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
