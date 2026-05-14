#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Verify every C/header/CMake source under Source/ and Include/ carries an
# SPDX-License-Identifier header and that the identifier is one of the two we
# allow (Apache-2.0 for upstream-derived files, LicenseRef-Ambiq-Apollo-SDK for
# Ambiq-authored files).
#
# Exits non-zero on any violation. Intended to run in CI and locally.
#
# Usage:
#   scripts/check_license_headers.sh           # check whole tree
#   scripts/check_license_headers.sh Source    # check a subset

set -euo pipefail

# Allowed SPDX-License-Identifier values.
ALLOWED='^(Apache-2\.0|LicenseRef-Ambiq-Apollo-SDK)$'

# Default scope if no args provided.
if [[ $# -eq 0 ]]; then
  set -- Source Include
fi

missing=()
disallowed=()
checked=0

while IFS= read -r -d '' f; do
  checked=$((checked + 1))
  spdx=$(grep -m1 -oE 'SPDX-License-Identifier:[[:space:]]+[^[:space:]*]+' "$f" | sed -E 's/SPDX-License-Identifier:[[:space:]]+//' || true)
  if [[ -z "$spdx" ]]; then
    missing+=("$f")
    continue
  fi
  if ! [[ "$spdx" =~ $ALLOWED ]]; then
    disallowed+=("$f -> $spdx")
  fi
done < <(find "$@" -type f \( -name '*.c' -o -name '*.h' -o -name '*.cmake' -o -name 'CMakeLists.txt' \) -print0)

status=0
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: ${#missing[@]} file(s) missing SPDX-License-Identifier header:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  status=1
fi
if (( ${#disallowed[@]} > 0 )); then
  echo "ERROR: ${#disallowed[@]} file(s) with disallowed SPDX-License-Identifier:" >&2
  printf '  %s\n' "${disallowed[@]}" >&2
  echo "Allowed identifiers: Apache-2.0, LicenseRef-Ambiq-Apollo-SDK" >&2
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "OK: $checked file(s) carry an allowed SPDX-License-Identifier."
fi
exit $status
