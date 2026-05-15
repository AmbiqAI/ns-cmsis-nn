#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# NSX prebuilt real-link smoke (#175).
#
# Wires a real, published `libns-cmsis-nn-<cpu>-<version>.a` into
# nsx/CMakeLists.txt's prebuilt branch and cross-links a tiny consumer
# TU against `nsx::cmsis_nn`. This closes the loop that the existing
# configure-time wiring test (cmake/tests/nsx_wiring) deliberately
# leaves open — that test exercises only CMake plumbing against a
# zero-byte placeholder, which catches wiring drift but not ABI/arch
# mismatch or missing symbols.
#
# Designed to run on a CI runner with `arm-none-eabi-gcc` on PATH; the
# downloaded .a comes from `gh release download` in the workflow job.
# Pair with cmake/toolchain/arm-none-eabi-gcc.cmake:
#
#   cmake -S cmake/tests/nsx_prebuilt_real_link \
#         -B build/nsx_prebuilt_real_link \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/arm-none-eabi-gcc.cmake \
#         -DNS_CMSIS_NN_TARGET_CPU=cortex-m4 \
#         -DNSX_CMSIS_NN_LIB=<path>/libns-cmsis-nn-cortex-m4-X.Y.Z.a
#   cmake --build build/nsx_prebuilt_real_link
#
cmake_minimum_required(VERSION 3.15)

# Override the toolchain's try_compile target-type (it defaults to
# STATIC_LIBRARY for the staticlib pipeline). We need executables to
# link end-to-end here so the smoke ELF actually validates the .a.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY CACHE INTERNAL "")

project(ns_cmsis_nn_nsx_prebuilt_real_link C)

if(NOT DEFINED NSX_CMSIS_NN_LIB)
  message(FATAL_ERROR
    "NSX_CMSIS_NN_LIB must point at the downloaded prebuilt .a.")
endif()
if(NOT EXISTS "${NSX_CMSIS_NN_LIB}")
  message(FATAL_ERROR "NSX_CMSIS_NN_LIB does not exist: ${NSX_CMSIS_NN_LIB}")
endif()

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# The NSX module expects a board-supplied INTERFACE target carrying
# arch flags via NSX_BOARD_FLAGS_TARGET. Stub it with the same -mcpu
# the toolchain set in CMAKE_C_FLAGS_INIT, so the consumer TU compiles
# with the ABI matching the prebuilt archive.
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)

add_subdirectory("${_repo_root}/nsx" nsx)

# Bare-metal link without a board's startup/linker glue:
#   -nostartfiles + --specs=nosys.specs + custom entry symbol.
# Same pattern scripts/smoke/smoke_staticlib.sh uses.
add_executable(nsx_prebuilt_real_link_smoke
  "${_repo_root}/scripts/smoke/staticlib_smoke.c")
target_link_libraries(nsx_prebuilt_real_link_smoke
  PRIVATE nsx::cmsis_nn _nsx_board_flags)
target_link_options(nsx_prebuilt_real_link_smoke PRIVATE
  -nostartfiles
  --specs=nosys.specs
  -Wl,--gc-sections
  -Wl,--entry=ns_cmsis_nn_smoke_refs
  # Force the archive in so the linker keeps the smoke-anchor symbols
  # even though no startup .o references them.
  -Wl,--whole-archive
  $<TARGET_PROPERTY:nsx_cmsis_nn_prebuilt,IMPORTED_LOCATION>
  -Wl,--no-whole-archive)
