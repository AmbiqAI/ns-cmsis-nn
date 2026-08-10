#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
manifest="${repo}/ci/tools/manifest.json"
dockerfile="${repo}/.devcontainer/Dockerfile"
validator="${repo}/scripts/ci/validate_ci_tool_manifest.jq"

jq -e -f "${validator}" "${manifest}" >/dev/null
jq -e '
  ([.tools[].id] | sort) == ([
    "arm-gnu", "armclang", "cmake", "cmsis-toolbox",
    "corstone-300-fvp", "llvm-embedded", "ninja"
  ] | sort)
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

invalid_manifest="$(mktemp)"
trap 'rm -f "${invalid_manifest}"' EXIT
jq '.tools[1].id = .tools[0].id | .tools[2].probe = "../../escape"' \
  "${manifest}" > "${invalid_manifest}"
if jq -e -f "${validator}" "${invalid_manifest}" >/dev/null; then
  echo "manifest validator accepted duplicate IDs or an unsafe probe path" >&2
  exit 1
fi

echo "CI tool manifest contract OK: exact HTTPS/checksum/license records replace vcpkg artifacts."
