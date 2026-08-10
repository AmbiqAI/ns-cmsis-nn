#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
manifest="${repo}/ci/tools/manifest.json"
dockerfile="${repo}/.devcontainer/Dockerfile"

jq -e '
  .schema_version == 1
  and .platform == "linux-x86_64"
  and ([.tools[].id] | unique | length) == (.tools | length)
  and ([.tools[].id] | sort) == ([
    "arm-gnu", "armclang", "cmake", "cmsis-toolbox",
    "corstone-300-fvp", "llvm-embedded", "ninja"
  ] | sort)
  and all(.tools[];
    (.url | startswith("https://"))
    and (.sha256 | test("^[0-9a-f]{64}$"))
    and (.license | length > 0))
' "${manifest}" >/dev/null

for forbidden in vcpkg VCPKG vcpkg-configuration.json; do
  if grep -q "${forbidden}" "${dockerfile}"; then
    echo "retired dependency remains in Dockerfile: ${forbidden}" >&2
    exit 1
  fi
done

grep -q 'scripts/install_ci_tools.sh' "${dockerfile}"
grep -q 'ci/tools/manifest.json' "${dockerfile}"
grep -Eq '^FROM python:3\.10-bookworm@sha256:[0-9a-f]{64} ' "${dockerfile}"
grep -q "archive contains an unsafe path" "${repo}/scripts/install_ci_tools.sh"

echo "CI tool manifest contract OK: exact HTTPS/checksum/license records replace vcpkg artifacts."
