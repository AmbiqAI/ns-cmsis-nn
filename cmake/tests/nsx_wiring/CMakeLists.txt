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
#   source_all      (default) — baseline int source set attached, alias present.
#   source_subset             — NSX_CMSIS_NN_GROUPS=activation;convolution,
#                               expect SSoT-derived source count.
#   inline_asm                — NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM=ON,
#                               expect CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
#                               on the target's COMPILE_DEFINITIONS.
#   float_f32                 — NSX_CMSIS_NN_ENABLE_F32=ON; expect _f32 sources
#                               + ARM_NN_ENABLE_F32=1 and ARM_NN_ENABLE_F16=0.
#   float_f16                 — NSX_CMSIS_NN_ENABLE_F16=ON; expect _f16 sources
#                               + ARM_NN_ENABLE_F32=0 and ARM_NN_ENABLE_F16=1.
#   float_both                — both float toggles ON; expect both _f32/_f16
#                               sources and both compile definitions =1.
#   prebuilt                  — NSX_CMSIS_NN_LIB=<dummy.a>, expect the
#                               IMPORTED prebuilt target + INTERFACE wrapper
#                               + alias.
#   prebuilt_manifest_ok      — NSX_CMSIS_NN_LIB with a sibling manifest.json
#                               reporting f32=true, NSX_CMSIS_NN_ENABLE_F32=ON;
#                               expect configure success + ARM_NN_ENABLE_F32=1.
#   prebuilt_manifest_reject  — same, but manifest reports f32=false; expect
#                               a configure-time FATAL_ERROR (via a child
#                               cmake probe), not a silent link-time failure.
#   prebuilt_manifest_legacy  — manifest.json has no "features" block
#                               (pre-existing schema_version 1 package);
#                               requesting F32 must FATAL_ERROR per the
#                               documented conservative integer-only fallback.
#
# Invoke as:
#
#     cmake -S cmake/tests/nsx_wiring -B build/nsx_wiring -DCASE=source_all
#
# string(JSON ...) (used by the prebuilt_manifest_* cases) requires CMake >= 3.19.
cmake_minimum_required(VERSION 3.19)
project(ns_cmsis_nn_nsx_wiring C)

set(CASE "source_all" CACHE STRING "Which wiring case to exercise.")
set_property(CACHE CASE PROPERTY STRINGS
  source_all source_subset inline_asm float_f32 float_f16 float_both prebuilt
  prebuilt_manifest_ok prebuilt_manifest_reject prebuilt_manifest_legacy)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${_repo_root}/cmake/ns_cmsis_nn.cmake")

function(_compute_expected_source_count out_var groups enable_f32 enable_f16)
  set(ARM_NN_ENABLE_F32 ${enable_f32})
  set(ARM_NN_ENABLE_F16 ${enable_f16})

  if(groups STREQUAL "ALL")
    ns_cmsis_nn_groups(_expected_groups)
  else()
    set(_expected_groups ${groups})
  endif()

  set(_expected_sources)
  foreach(_g IN LISTS _expected_groups)
    ns_cmsis_nn_group_sources(${_g} _group_sources)
    list(APPEND _expected_sources ${_group_sources})
  endforeach()

  if(_expected_sources)
    list(REMOVE_DUPLICATES _expected_sources)
  endif()
  list(LENGTH _expected_sources _expected_count)
  set(${out_var} ${_expected_count} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Failure collector.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# prebuilt_manifest_reject / prebuilt_manifest_legacy: nsx/CMakeLists.txt's
# manifest validation raises a configure-time FATAL_ERROR, which would abort
# this whole test script if triggered directly via add_subdirectory(). Drive
# it via a child cmake invocation instead (same pattern as
# cmake/tests/find_package's cpu_mismatch case) and assert on the non-zero
# exit + diagnostic message.
# ----------------------------------------------------------------------------
if(CASE STREQUAL "prebuilt_manifest_reject" OR CASE STREQUAL "prebuilt_manifest_legacy")
  set(_fake_lib "${CMAKE_BINARY_DIR}/_fake_prebuilt_reject.a")
  file(TOUCH "${_fake_lib}")

  if(CASE STREQUAL "prebuilt_manifest_reject")
    # Archive explicitly reports it does NOT have f32.
    file(WRITE "${CMAKE_BINARY_DIR}/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": false, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")
  else()
    # Pre-existing schema_version 1 package: no "features" block at all.
    file(WRITE "${CMAKE_BINARY_DIR}/manifest.json" "{
  \"schema_version\": 1
}
")
  endif()

  set(_probe "${CMAKE_BINARY_DIR}/_manifest_probe")
  file(REMOVE_RECURSE "${_probe}")
  file(MAKE_DIRECTORY "${_probe}")
  file(WRITE "${_probe}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
set(NSX_CMSIS_NN_LIB \"${_fake_lib}\" CACHE FILEPATH \"\" FORCE)
set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL \"\" FORCE)
add_subdirectory(\"${_repo_root}/nsx\" nsx)
")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err)
  if(_rc EQUAL 0)
    _fail("${CASE}: probe unexpectedly succeeded (expected FATAL_ERROR)\n${_out}\n${_err}")
  else()
    string(FIND "${_err}" "NSX_CMSIS_NN_ENABLE_F32 was requested" _msg_idx)
    if(_msg_idx EQUAL -1)
      _fail("${CASE}: probe failed but without the expected diagnostic; stderr:\n${_err}")
    endif()
  endif()

  if(_failures)
    message(FATAL_ERROR "NSX wiring [${CASE}] FAILED:${_failures}")
  endif()
  message(STATUS "NSX wiring [${CASE}] OK.")
  return()
endif()

# ----------------------------------------------------------------------------
# Required host plumbing: the NSX board provides this; we stub it.
# ----------------------------------------------------------------------------
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)

# ----------------------------------------------------------------------------
# Per-case configuration. Must be set BEFORE add_subdirectory() because the
# nsx CMakeLists reads them at configure time.
# ----------------------------------------------------------------------------
# Baseline defaults to avoid cache leakage when the same build directory is
# reconfigured for multiple CASE values.
set(NSX_CMSIS_NN_GROUPS "ALL" CACHE STRING "" FORCE)
set(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM OFF CACHE BOOL "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F32 OFF CACHE BOOL "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F16 OFF CACHE BOOL "" FORCE)
unset(NSX_CMSIS_NN_LIB CACHE)

if(CASE STREQUAL "source_all")
  _compute_expected_source_count(_expected_sources "ALL" 0 0)

elseif(CASE STREQUAL "source_subset")
  set(NSX_CMSIS_NN_GROUPS "activation;convolution" CACHE STRING "" FORCE)
  _compute_expected_source_count(_expected_sources "activation;convolution" 0 0)

elseif(CASE STREQUAL "inline_asm")
  set(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 0 0)
  set(_expect_inline_asm_define TRUE)

elseif(CASE STREQUAL "float_f32")
  set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 1 0)
  set(_expect_f32_define 1)
  set(_expect_f16_define 0)
  set(_expect_f32_sources TRUE)

elseif(CASE STREQUAL "float_f16")
  set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 0 1)
  set(_expect_f32_define 0)
  set(_expect_f16_define 1)
  set(_expect_f16_sources TRUE)

elseif(CASE STREQUAL "float_both")
  set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 1 1)
  set(_expect_f32_define 1)
  set(_expect_f16_define 1)
  set(_expect_f32_sources TRUE)
  set(_expect_f16_sources TRUE)

elseif(CASE STREQUAL "prebuilt")
  set(_fake_lib "${CMAKE_BINARY_DIR}/_fake_prebuilt.a")
  file(TOUCH "${_fake_lib}")
  set(NSX_CMSIS_NN_LIB "${_fake_lib}" CACHE FILEPATH "" FORCE)
  set(_expect_prebuilt TRUE)

elseif(CASE STREQUAL "prebuilt_manifest_ok")
  set(_fake_lib "${CMAKE_BINARY_DIR}/_fake_prebuilt_ok.a")
  file(TOUCH "${_fake_lib}")
  # manifest.json sits next to the archive — matches the tarball's
  # "lib/libns-cmsis-nn.a" + root "manifest.json" layout auto-discovery rule.
  file(WRITE "${CMAKE_BINARY_DIR}/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": true, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")
  set(NSX_CMSIS_NN_LIB "${_fake_lib}" CACHE FILEPATH "" FORCE)
  set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  set(_expect_prebuilt TRUE)
  set(_expect_f32_define 1)
  set(_expect_f16_define 0)

else()
  message(FATAL_ERROR "Unknown CASE='${CASE}'. Valid: source_all, source_subset, inline_asm, float_f32, float_f16, float_both, prebuilt, prebuilt_manifest_ok, prebuilt_manifest_reject, prebuilt_manifest_legacy.")
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

if(DEFINED _expect_f32_define)
  if(TARGET nsx_cmsis_nn)
    get_target_property(_defs nsx_cmsis_nn COMPILE_DEFINITIONS)
    list(FIND _defs "ARM_NN_ENABLE_F32=${_expect_f32_define}" _idx)
    if(_idx EQUAL -1)
      _fail("${CASE} mode: expected ARM_NN_ENABLE_F32=${_expect_f32_define} in COMPILE_DEFINITIONS; got: ${_defs}")
    endif()
  endif()
endif()

if(DEFINED _expect_f16_define)
  if(TARGET nsx_cmsis_nn)
    get_target_property(_defs nsx_cmsis_nn COMPILE_DEFINITIONS)
    list(FIND _defs "ARM_NN_ENABLE_F16=${_expect_f16_define}" _idx)
    if(_idx EQUAL -1)
      _fail("${CASE} mode: expected ARM_NN_ENABLE_F16=${_expect_f16_define} in COMPILE_DEFINITIONS; got: ${_defs}")
    endif()
  endif()
endif()

if(_expect_f32_sources)
  if(TARGET nsx_cmsis_nn)
    get_target_property(_srcs nsx_cmsis_nn SOURCES)
    set(_f32_srcs ${_srcs})
    list(FILTER _f32_srcs INCLUDE REGEX ".*(_f32\\.c)$")
    list(LENGTH _f32_srcs _n_f32_srcs)
    if(NOT _n_f32_srcs GREATER 0)
      _fail("${CASE} mode: no _f32.c sources selected")
    endif()
  endif()
endif()

if(_expect_f16_sources)
  if(TARGET nsx_cmsis_nn)
    get_target_property(_srcs nsx_cmsis_nn SOURCES)
    set(_f16_srcs ${_srcs})
    list(FILTER _f16_srcs INCLUDE REGEX ".*(_f16\\.c)$")
    list(LENGTH _f16_srcs _n_f16_srcs)
    if(NOT _n_f16_srcs GREATER 0)
      _fail("${CASE} mode: no _f16.c sources selected")
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
