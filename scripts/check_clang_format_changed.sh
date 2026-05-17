#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

set -euo pipefail

BASE_REF="${1:-${BASE_REF:-}}"
HEAD_REF="${2:-${HEAD_REF:-HEAD}}"

if [[ -z "${BASE_REF}" ]]; then
  echo "usage: $0 <base-ref> [head-ref]" >&2
  echo "example: $0 origin/main HEAD" >&2
  exit 2
fi

command -v git >/dev/null || { echo "git not found" >&2; exit 3; }

git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null
git rev-parse --verify "${HEAD_REF}^{commit}" >/dev/null

changed_files="$(git diff --name-only --diff-filter=ACMR "${BASE_REF}...${HEAD_REF}" -- \
  'Source/*.c' \
  'Source/**/*.c' \
  'Include/*.h' \
  'Include/**/*.h')"

if [[ -z "${changed_files}" ]]; then
  echo "No Source/ or Include/ C/H files changed between ${BASE_REF} and ${HEAD_REF}."
  exit 0
fi

command -v pre-commit >/dev/null || { echo "pre-commit not found" >&2; exit 3; }

echo "Checking clang-format on changed Source/ and Include/ files:"
printf '%s\n' "${changed_files}"

pre-commit run clang-format \
  --from-ref "${BASE_REF}" \
  --to-ref "${HEAD_REF}" \
  --show-diff-on-failure
