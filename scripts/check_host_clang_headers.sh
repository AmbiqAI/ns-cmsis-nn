#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Compile every public header on its own with a clang that targets the host,
# in C and C++, under each float configuration. Host tools (helia-aot's native
# contract tests) build kernels this way, and a clang with no ACLE sees none
# of the Arm-only branches the embedded builds take: CLZ once expanded to an
# undeclared __clz there, and nothing in CI compiled the headers with a
# non-Arm clang. Syntax only, so no Arm toolchain or link step is needed.
#
# Usage: scripts/check_host_clang_headers.sh [clang]   (default: clang)

set -euo pipefail

clang_bin="${1:-clang}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

"${clang_bin}" --version | head -n 1

# arm_nnsupportfunctions_flt.h is a fragment that arm_nnsupportfunctions.h
# includes after defining the macros it uses, so it is compiled through that
# header rather than on its own.
headers=()
for path in "${root}"/Include/*.h; do
  name="$(basename "${path}")"
  [[ "${name}" == arm_nnsupportfunctions_flt.h ]] && continue
  headers+=("${name}")
done
((${#headers[@]} > 0)) || { echo "no headers found under ${root}/Include" >&2; exit 1; }

configs=("F32=1 F16=1" "F32=0 F16=1" "F32=1 F16=0")
failures=0
for cfg in "${configs[@]}"; do
  defines=()
  for kv in ${cfg}; do
    defines+=("-DARM_NN_ENABLE_${kv}")
  done
  for lang in c c++; do
    if [[ "${lang}" == c ]]; then std=-std=c11; else std=-std=c++17; fi
    for header in "${headers[@]}"; do
      printf '#include "%s"\n' "${header}" > "${work}/tu"
      if ! "${clang_bin}" --target=x86_64-linux-gnu -x "${lang}" "${std}" -fsyntax-only -Wall -Wextra -Werror \
          "${defines[@]}" -I"${root}/Include" "${work}/tu" 2> "${work}/err"; then
        echo "FAIL ${header} (${lang}, ${cfg})"
        sed 's/^/    /' "${work}/err"
        failures=$((failures + 1))
      fi
    done
  done
done

checked=$((${#headers[@]} * ${#configs[@]} * 2))
if ((failures > 0)); then
  echo "${failures} of ${checked} host-clang header compiles failed" >&2
  exit 1
fi
echo "${checked} host-clang header compiles passed"
