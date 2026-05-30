#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# NSX-integration wiring test for nsx/CMakeLists.txt.
#
# Runs entirely at CMake configure time (no NSX board, no Apollo toolchain).
# Stubs the one thing the NSX module requires from its host environment —
# NSX_BOARD_FLAGS_TARGET — then add_subdirectory()'s the real nsx module
# and asserts on the resulting target graph.
#
# Parameterised by -DCASE=<case>, where <case> is one of:
#   source_all      (default) — 221 sources attached, alias present.
#   source_subset             — NSX_CMSIS_NN_GROUPS=activation;convolution,
#                               expect 62 sources.
#   inline_asm                — NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM=ON,
#                               expect CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
#                               on the target's COMPILE_DEFINITIONS.
#   prebuilt                  — NSX_CMSIS_NN_LIB=<dummy.a>, expect the
#                               IMPORTED prebuilt target + INTERFACE wrapper
#                               + alias.
#
# Invoke as:
#
#     cmake -S cmake/tests/nsx_wiring -B build/nsx_wiring -DCASE=source_all
#
cmake_minimum_required(VERSION 3.15)
project(ns_cmsis_nn_nsx_wiring C)

set(CASE "source_all" CACHE STRING "Which wiring case to exercise.")
set_property(CACHE CASE PROPERTY STRINGS
  source_all source_subset inline_asm prebuilt)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# ----------------------------------------------------------------------------
# Failure collector.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Required host plumbing: the NSX board provides this; we stub it.
# ----------------------------------------------------------------------------
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)

# ----------------------------------------------------------------------------
# Per-case configuration. Must be set BEFORE add_subdirectory() because the
# nsx CMakeLists reads them at configure time.
# ----------------------------------------------------------------------------
if(CASE STREQUAL "source_all")
  set(_expected_sources 221)

elseif(CASE STREQUAL "source_subset")
  # activation (18) + convolution (44) = 62. If the SSoT counts drift, the
  # ssot_contract test fires first — keep these two numbers in sync with
  # cmake/tests/ssot_contract/CMakeLists.txt::EXPECTED_COUNT_*.
  set(NSX_CMSIS_NN_GROUPS "activation;convolution" CACHE STRING "" FORCE)
  set(_expected_sources 62)

elseif(CASE STREQUAL "inline_asm")
  set(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM ON CACHE BOOL "" FORCE)
  set(_expected_sources 221)
  set(_expect_inline_asm_define TRUE)

elseif(CASE STREQUAL "prebuilt")
  set(_fake_lib "${CMAKE_BINARY_DIR}/_fake_prebuilt.a")
  file(TOUCH "${_fake_lib}")
  set(NSX_CMSIS_NN_LIB "${_fake_lib}" CACHE FILEPATH "" FORCE)
  set(_expect_prebuilt TRUE)

else()
  message(FATAL_ERROR "Unknown CASE='${CASE}'. Valid: source_all, source_subset, inline_asm, prebuilt.")
endif()

# ----------------------------------------------------------------------------
# Drive the real NSX module.
# ----------------------------------------------------------------------------
add_subdirectory("${_repo_root}/nsx" "${CMAKE_BINARY_DIR}/_nsx")

# ----------------------------------------------------------------------------
# Common assertions: target + alias must exist in every mode.
# ----------------------------------------------------------------------------
if(NOT TARGET nsx_cmsis_nn)
  _fail("nsx/CMakeLists.txt did not create the nsx_cmsis_nn target")
endif()
if(NOT TARGET nsx::cmsis_nn)
  _fail("nsx::cmsis_nn alias was not created")
endif()

if(TARGET nsx_cmsis_nn)
  get_target_property(_export_name nsx_cmsis_nn EXPORT_NAME)
  if(NOT _export_name STREQUAL "cmsis_nn")
    _fail("nsx_cmsis_nn EXPORT_NAME='${_export_name}', expected 'cmsis_nn' (consumer find_package contract).")
  endif()
endif()

# ----------------------------------------------------------------------------
# Mode-specific assertions.
# ----------------------------------------------------------------------------
if(_expect_prebuilt)
  # nsx_cmsis_nn must be an INTERFACE wrapper that links the IMPORTED prebuilt.
  if(TARGET nsx_cmsis_nn)
    get_target_property(_type nsx_cmsis_nn TYPE)
    if(NOT _type STREQUAL "INTERFACE_LIBRARY")
      _fail("prebuilt mode: nsx_cmsis_nn TYPE='${_type}', expected INTERFACE_LIBRARY")
    endif()
  endif()
  if(NOT TARGET nsx_cmsis_nn_prebuilt)
    _fail("prebuilt mode: nsx_cmsis_nn_prebuilt IMPORTED target missing")
  else()
    get_target_property(_imp nsx_cmsis_nn_prebuilt IMPORTED)
    if(NOT _imp)
      _fail("prebuilt mode: nsx_cmsis_nn_prebuilt is not IMPORTED")
    endif()
    get_target_property(_loc nsx_cmsis_nn_prebuilt IMPORTED_LOCATION)
    if(NOT _loc STREQUAL "${_fake_lib}")
      _fail("prebuilt mode: IMPORTED_LOCATION='${_loc}', expected '${_fake_lib}'")
    endif()
  endif()
else()
  # Source modes: nsx_cmsis_nn is STATIC with N sources attached.
  if(TARGET nsx_cmsis_nn)
    get_target_property(_type nsx_cmsis_nn TYPE)
    if(NOT _type STREQUAL "STATIC_LIBRARY")
      _fail("source mode: nsx_cmsis_nn TYPE='${_type}', expected STATIC_LIBRARY")
    endif()
    get_target_property(_srcs nsx_cmsis_nn SOURCES)
    if(NOT _srcs)
      _fail("source mode: no sources attached to nsx_cmsis_nn")
    else()
      list(LENGTH _srcs _nsrc)
      if(NOT _nsrc EQUAL _expected_sources)
        _fail("source mode (${CASE}): attached ${_nsrc} sources; expected ${_expected_sources}")
      endif()
    endif()
  endif()
endif()

if(_expect_inline_asm_define)
  if(TARGET nsx_cmsis_nn)
    get_target_property(_defs nsx_cmsis_nn COMPILE_DEFINITIONS)
    list(FIND _defs "CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY" _idx)
    if(_idx EQUAL -1)
      _fail("inline_asm mode: CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY missing from COMPILE_DEFINITIONS; got: ${_defs}")
    endif()
  endif()
endif()

# Include-dir wiring: the generator-expression split is what fixed #159 —
# BUILD_INTERFACE for in-tree consumers, INSTALL_INTERFACE for the
# installed package. Both must be present in every mode.
if(TARGET nsx_cmsis_nn)
  get_target_property(_incs nsx_cmsis_nn INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _incs)
    _fail("nsx_cmsis_nn has no INTERFACE_INCLUDE_DIRECTORIES")
  else()
    string(FIND "${_incs}" "BUILD_INTERFACE:" _idx)
    if(_idx EQUAL -1)
      _fail("BUILD_INTERFACE generator expression missing on Include/; got: ${_incs}")
    endif()
    string(FIND "${_incs}" "INSTALL_INTERFACE:" _idx)
    if(_idx EQUAL -1)
      _fail("INSTALL_INTERFACE generator expression missing on Include/ — relocatable install will break (regression #159); got: ${_incs}")
    endif()
  endif()
endif()

# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------
if(_failures)
  message(FATAL_ERROR "NSX wiring [${CASE}] FAILED:${_failures}")
endif()

message(STATUS "NSX wiring [${CASE}] OK.")
