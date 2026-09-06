#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
manifest="${repo}/ci/tools/manifest.json"
dockerfile="${repo}/.devcontainer/Dockerfile"
validator="${repo}/scripts/ci/validate_ci_tool_manifest.jq"
workflow="${repo}/.github/workflows/clang-format.yml"
check_script="${repo}/scripts/check_clang_format_changed.sh"

if ! jq -e -f "${validator}" "${manifest}" >/dev/null; then
  echo "ci/tools/manifest.json failed schema validation against ${validator}" >&2
  exit 1
fi
jq -e '
  ([.tools[].id] | sort) == ([
    "arm-gnu", "armclang", "cmake", "cmsis-toolbox",
    "corstone-300-fvp", "llvm-embedded", "ninja"
  ] | sort)
' "${manifest}" >/dev/null

jq -e '([.python_tools[].id] | sort) == ["clang-format"]' "${manifest}" >/dev/null

# The formatter pin is a one-way door: once files carry 18 output, 16 rejects
# them, so every surface that names the version has to move together
# (see AmbiqAI/ns-cmsis-nn#394).
clang_format_version="$(jq -r \
  '.python_tools[] | select(.id == "clang-format") | .version' "${manifest}")"
clang_format_major="${clang_format_version%%.*}"

grep -q "rev: v${clang_format_version}$" "${repo}/.pre-commit-config.yaml" \
  || { echo "pre-commit clang-format rev does not match manifest ${clang_format_version}" >&2; exit 1; }
grep -q "clang-format==${clang_format_version}$" "${workflow}" \
  || { echo "clang-format workflow pin does not match manifest ${clang_format_version}" >&2; exit 1; }
grep -q "^REQUIRED_CLANG_FORMAT_MAJOR=${clang_format_major}$" "${check_script}" \
  || { echo "REQUIRED_CLANG_FORMAT_MAJOR does not match manifest major ${clang_format_major}" >&2; exit 1; }
grep -q "^PINNED_CLANG_FORMAT_VERSION=${clang_format_version}$" "${check_script}" \
  || { echo "PINNED_CLANG_FORMAT_VERSION does not match manifest ${clang_format_version}" >&2; exit 1; }
grep -q "clang-format==${clang_format_version}" "${repo}/docs/contributing.md" \
  || { echo "docs/contributing.md does not name manifest ${clang_format_version}" >&2; exit 1; }
grep -q "clang-format==${clang_format_version}" "${repo}/AGENTS.md" \
  || { echo "AGENTS.md does not name manifest ${clang_format_version}" >&2; exit 1; }
# A stray pin left over from a prior bump reads as current and would not be
# caught by the "names the manifest version" checks above, which only require
# the right pin to appear somewhere, not that it is the only one.
for doc in "${repo}/docs/contributing.md" "${repo}/AGENTS.md"; do
  stale="$(grep -oE 'clang-format==[0-9][0-9.]*' "${doc}" | sort -u | grep -vFx "clang-format==${clang_format_version}" || true)"
  if [[ -n "${stale}" ]]; then
    echo "${doc} carries a clang-format pin that disagrees with manifest ${clang_format_version}: ${stale}" >&2
    exit 1
  fi
done
# The image must take the wheel from the manifest, not a literal pip pin.
grep -q "python_tools" "${dockerfile}" \
  || { echo "Dockerfile does not install clang-format from the manifest" >&2; exit 1; }
if grep -q "pip install .*clang-format==" "${dockerfile}"; then
  echo "Dockerfile still carries a literal clang-format pip pin" >&2
  exit 1
fi

# The pre-commit hook and the changed-files gate must skip the same upstream
# files. If they drift, one gate reformats bytes the other refuses to touch, and
# the resulting PR cannot go green (see AmbiqAI/ns-cmsis-nn#394).
precommit_skipped="$(awk '
  /^  - repo: .*mirrors-clang-format/ { hook = 1; next }
  /^  - repo:/ { hook = 0; block = 0 }
  hook && /^ *exclude: *\|/ { block = 1; next }
  block { print }
' "${repo}/.pre-commit-config.yaml" |
  sed -e 's/[[:space:]]//g' -e 's/\\//g' -e 's/^|//' -e 's/\$$//' |
  grep -E '\.(c|h)$' | sort -u)"
gate_skipped="$(grep -E "^unformatted_upstream=" "${check_script}" |
  sed -e "s/^unformatted_upstream='\^(//" -e "s/)\\\$'\$//" |
  tr '|' '\n' |
  sed -e 's/[[:space:]]//g' -e 's/\\//g' |
  grep -E '\.(c|h)$' | sort -u)"
if [[ -z "${precommit_skipped}" || -z "${gate_skipped}" ]]; then
  echo "could not read the clang-format skip list from .pre-commit-config.yaml or ${check_script}" >&2
  exit 1
fi
if [[ "${precommit_skipped}" != "${gate_skipped}" ]]; then
  echo "clang-format skip lists disagree between .pre-commit-config.yaml and check_clang_format_changed.sh:" >&2
  diff <(echo "${precommit_skipped}") <(echo "${gate_skipped}") >&2 || true
  exit 1
fi

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
echo "clang-format ${clang_format_version} is consistent across the manifest, pre-commit, CI, the dev container and the docs."
