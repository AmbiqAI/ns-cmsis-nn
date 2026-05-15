#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# NSX install contract test.
#
# Drives a real configure -> build -> install for the source-mode NSX
# integration, then inspects the install tree to verify:
#
#   1. nsxTargets.cmake (and the noconfig variant) are present at the
#      expected path — the EXPORT set is actually installed.
#   2. The exported include-dirs and IMPORTED_LOCATION use ${_IMPORT_PREFIX}
#      relocatable paths — no absolute build-tree path leaks (regression
#      guard for #159).
#   3. Every public header from Include/ landed at the right install path.
#   4. The static archive landed at the right install path.
#
# Runs on ubuntu-latest x86_64 with stock GCC — no cross-compiler required,
# the NSX module only needs a C compiler to produce the static library.
#

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-${REPO}/build/nsx_install_test}"
BUILD="${WORK}/build"
PREFIX="${WORK}/stage"
WRAPPER="${WORK}/wrapper"

rm -rf "${WORK}"
mkdir -p "${WRAPPER}"

# ----------------------------------------------------------------------------
# Tiny wrapper project that supplies NSX_BOARD_FLAGS_TARGET and registers
# install(EXPORT nsxTargets) — mirroring what an NSX board does in real life.
# ----------------------------------------------------------------------------
cat >"${WRAPPER}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.15)
project(nsx_install_test C)

# Stub for the board's compile-flags carrier.
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
install(TARGETS _nsx_board_flags EXPORT nsxTargets)

add_subdirectory("${REPO}/nsx" "\${CMAKE_BINARY_DIR}/_nsx")

# An NSX board owns the EXPORT-set install. Mimic it here so we can
# inspect the generated nsxTargets.cmake.
install(EXPORT nsxTargets
        DESTINATION lib/cmake/nsx
        FILE nsxTargets.cmake)
EOF

echo "== configure =="
cmake -S "${WRAPPER}" -B "${BUILD}" >/dev/null

echo "== build =="
cmake --build "${BUILD}" -j"$(nproc)" >/dev/null

echo "== install =="
cmake --install "${BUILD}" --prefix "${PREFIX}" >/dev/null

# ----------------------------------------------------------------------------
# Assertions.
# ----------------------------------------------------------------------------
fail=0
report() { printf '  - %s\n' "$1" >&2; fail=1; }

TARGETS="${PREFIX}/lib/cmake/nsx/nsxTargets.cmake"
NOCFG="${PREFIX}/lib/cmake/nsx/nsxTargets-noconfig.cmake"
LIB="${PREFIX}/lib/libnsx_cmsis_nn.a"
INCDIR="${PREFIX}/include/nsx/modules/nsx-cmsis-nn/Include"

# (1) Expected files present.
for f in "${TARGETS}" "${NOCFG}" "${LIB}" \
         "${INCDIR}/arm_nnfunctions.h" \
         "${INCDIR}/arm_nn_types.h" \
         "${INCDIR}/arm_nn_tables.h" \
         "${INCDIR}/arm_nn_math_types.h" \
         "${INCDIR}/arm_nnsupportfunctions.h" \
         "${INCDIR}/Internal/arm_nn_compiler.h"; do
  [[ -f "${f}" ]] || report "missing installed artefact: ${f#${PREFIX}/}"
done

# (2) Targets file must use ${_IMPORT_PREFIX} — no absolute build-tree paths.
if [[ -f "${TARGETS}" ]]; then
  # Any line that names an absolute path and is NOT wrapped in ${_IMPORT_PREFIX}
  # is a leak. Search for the build dir specifically.
  if grep -F "${BUILD}" "${TARGETS}" "${NOCFG}" 2>/dev/null; then
    report "build-tree path leaked into installed nsxTargets (regression #159)"
  fi
  # Sanity: relocatable prefix actually used.
  grep -q '\${_IMPORT_PREFIX}/include/nsx/modules/nsx-cmsis-nn/Include' "${TARGETS}" \
    || report "INTERFACE_INCLUDE_DIRECTORIES does not use \${_IMPORT_PREFIX}"
  grep -q '\${_IMPORT_PREFIX}/lib/libnsx_cmsis_nn.a' "${NOCFG}" \
    || report "IMPORTED_LOCATION does not use \${_IMPORT_PREFIX}"
  # The repo Include dir must NOT leak either.
  if grep -F "${REPO}/Include" "${TARGETS}" "${NOCFG}" 2>/dev/null; then
    report "source-tree Include path leaked into installed nsxTargets"
  fi
fi

# (3) Drive a tiny downstream consumer: find_package against the staged prefix
#     and assert it pulls in the cmsis_nn target.
CONS="${WORK}/consumer"
mkdir -p "${CONS}"
cat >"${CONS}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.15)
project(nsx_consumer NONE)
# nsxTargets.cmake is not a Config file but its include() is what NSX board
# would do. Mimic that.
include("${CMAKE_FIND_PACKAGE_NAME_OR_PREFIX}/lib/cmake/nsx/nsxTargets.cmake")
if(NOT TARGET cmsis_nn)
  message(FATAL_ERROR "cmsis_nn target not imported from nsxTargets.cmake")
endif()
get_target_property(_inc cmsis_nn INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "consumer sees INTERFACE_INCLUDE_DIRECTORIES=${_inc}")
EOF
if ! cmake -S "${CONS}" -B "${CONS}/build" \
        -DCMAKE_FIND_PACKAGE_NAME_OR_PREFIX="${PREFIX}" >/dev/null; then
  report "downstream consumer cannot import cmsis_nn from installed nsxTargets"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "NSX install contract FAILED" >&2
  exit 1
fi

echo "NSX install contract OK: ${TARGETS#${PREFIX}/} relocatable, headers + libnsx_cmsis_nn.a installed, consumer import works."
