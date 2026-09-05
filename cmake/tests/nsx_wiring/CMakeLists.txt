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
#   float_arm_nn_only         - ARM_NN_ENABLE_F16=ON alone, both NSX float
#                               switches at their defaults; the ARM_NN
#                               spelling is the request, so expect _f16
#                               sources + ARM_NN_ENABLE_F16=1 published.
#   float_conflict            - NSX_CMSIS_NN_ENABLE_F32=ON (away from its
#                               default) while the cache carries
#                               ARM_NN_ENABLE_F32=OFF; two explicit requests
#                               disagree, so expect a configure-time
#                               FATAL_ERROR (via a child cmake probe).
#                               See AmbiqAI/ns-cmsis-nn#420.
#   float_conflict_plain      - same contradiction, but ARM_NN_ENABLE_F16 is a
#                               plain (non-cache) variable in the calling
#                               scope; expect the same FATAL_ERROR.
#   prebuilt                  — NSX_CMSIS_NN_LIB=<dummy.a>, expect the
#                               IMPORTED prebuilt target + INTERFACE wrapper
#                               + alias.
#   prebuilt_manifest_ok      — NSX_CMSIS_NN_LIB with a sibling manifest.json
#                               reporting f32=true, NSX_CMSIS_NN_ENABLE_F32=ON;
#                               expect configure success + ARM_NN_ENABLE_F32=1.
#   prebuilt_manifest_reject  — same, but manifest reports f32=false; expect
#                               a configure-time FATAL_ERROR (via a child
#                               cmake probe) naming NSX_CMSIS_NN_ENABLE_F32 as
#                               the spelling the request came from, not a
#                               silent link-time failure, and expect the -U it
#                               names to clear the rejection.
#   prebuilt_manifest_reject_arm_nn
#                             - manifest reports f16=false and the request
#                               arrives only as ARM_NN_ENABLE_F16=ON, with the
#                               NSX switches at their defaults; validation
#                               reads the resolved request, so expect the same
#                               manifest FATAL_ERROR, naming ARM_NN_ENABLE_F16
#                               and recovering under -UARM_NN_ENABLE_F16.
#                               See AmbiqAI/ns-cmsis-nn#420.
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
  source_all source_subset inline_asm float_f32 float_f16 float_both
  float_arm_nn_only float_conflict float_conflict_plain prebuilt
  prebuilt_manifest_ok prebuilt_manifest_reject prebuilt_manifest_reject_arm_nn
  prebuilt_manifest_legacy)

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
if(CASE STREQUAL "prebuilt_manifest_reject"
   OR CASE STREQUAL "prebuilt_manifest_reject_arm_nn"
   OR CASE STREQUAL "prebuilt_manifest_legacy")
  set(_fake_lib "${CMAKE_BINARY_DIR}/_fake_prebuilt_reject.a")
  file(TOUCH "${_fake_lib}")

  if(CASE STREQUAL "prebuilt_manifest_legacy")
    # Pre-existing schema_version 1 package: no "features" block at all.
    file(WRITE "${CMAKE_BINARY_DIR}/manifest.json" "{
  \"schema_version\": 1
}
")
  else()
    # Archive explicitly reports it has neither float width.
    file(WRITE "${CMAKE_BINARY_DIR}/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": false, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")
  endif()

  # The request arrives on the child cmake command line, not as a set() inside
  # the probe, so the -U recovery the diagnostic names is actually reachable.
  if(CASE STREQUAL "prebuilt_manifest_reject_arm_nn")
    # The NSX switches stay at their defaults, so the ARM_NN spelling is the
    # whole request: the manifest check must name that spelling, not the NSX
    # one. See AmbiqAI/ns-cmsis-nn#420.
    set(_probe_args "-DARM_NN_ENABLE_F16=ON")
    set(_probe_needles
        "float16 requested via ARM_NN_ENABLE_F16"
        "-UARM_NN_ENABLE_F16")
    set(_probe_recovery "-UARM_NN_ENABLE_F16")
  else()
    set(_probe_args "-DNSX_CMSIS_NN_ENABLE_F32=ON")
    set(_probe_needles
        "float32 requested via NSX_CMSIS_NN_ENABLE_F32"
        "-UNSX_CMSIS_NN_ENABLE_F32")
    set(_probe_recovery "-UNSX_CMSIS_NN_ENABLE_F32")
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
add_subdirectory(\"${_repo_root}/nsx\" nsx)
")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b" ${_probe_args}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err)
  if(_rc EQUAL 0)
    _fail("${CASE}: probe unexpectedly succeeded (expected FATAL_ERROR)\n${_out}\n${_err}")
  else()
    foreach(_needle IN LISTS _probe_needles)
      string(FIND "${_err}" "${_needle}" _msg_idx)
      if(_msg_idx EQUAL -1)
        _fail("${CASE}: diagnostic never mentions ${_needle}; stderr:\n${_err}")
      endif()
    endforeach()
  endif()

  # A rejection the user cannot undo is not a diagnostic. Re-run the same build
  # directory with the -U the message names and expect a clean configure.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b" ${_probe_recovery}
    RESULT_VARIABLE _rc_recover
    OUTPUT_VARIABLE _out_recover
    ERROR_VARIABLE  _err_recover)
  if(NOT _rc_recover EQUAL 0)
    _fail("${CASE}: ${_probe_recovery} did not clear the rejection\n${_out_recover}\n${_err_recover}")
  endif()

  if(_failures)
    message(FATAL_ERROR "NSX wiring [${CASE}] FAILED:${_failures}")
  endif()
  message(STATUS "NSX wiring [${CASE}] OK.")
  return()
endif()

# ----------------------------------------------------------------------------
# float_conflict / float_conflict_plain: two explicit requests that disagree
# are a configure-time FATAL_ERROR, which would abort this script if triggered
# directly. Drive it via a child cmake invocation, as above. The NSX switch has
# to be away from its default for the disagreement to count; at its default the
# ARM_NN_ENABLE_* value is simply the request (float_arm_nn_only).
# ----------------------------------------------------------------------------
if(CASE STREQUAL "float_conflict" OR CASE STREQUAL "float_conflict_plain")
  set(_probe "${CMAKE_BINARY_DIR}/_conflict_probe")
  file(REMOVE_RECURSE "${_probe}")
  file(MAKE_DIRECTORY "${_probe}")
  if(CASE STREQUAL "float_conflict")
    set(_probe_request
"set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL \"\" FORCE)
set(ARM_NN_ENABLE_F32 OFF CACHE BOOL \"\" FORCE)")
    set(_probe_names ARM_NN_ENABLE_F32 NSX_CMSIS_NN_ENABLE_F32 -UARM_NN_ENABLE_F32)
  else()
    # Plain directory-scope variable: what a consumer gets from a set() above
    # add_subdirectory() rather than from -D.
    set(_probe_request
"set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL \"\" FORCE)
set(ARM_NN_ENABLE_F16 OFF)")
    set(_probe_names ARM_NN_ENABLE_F16 NSX_CMSIS_NN_ENABLE_F16 -UNSX_CMSIS_NN_ENABLE_F16)
  endif()
  file(WRITE "${_probe}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
${_probe_request}
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
    string(FIND "${_err}" "contradictory" _msg_idx)
    if(_msg_idx EQUAL -1)
      _fail("${CASE}: probe failed but without the expected diagnostic; stderr:\n${_err}")
    endif()
    # The diagnostic has to name both spellings and the way out of the wedge.
    foreach(_needle IN LISTS _probe_names)
      string(FIND "${_err}" "${_needle}" _idx)
      if(_idx EQUAL -1)
        _fail("${CASE}: diagnostic never mentions ${_needle}; stderr:\n${_err}")
      endif()
    endforeach()
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
# What an earlier CASE published, and the bookkeeping that tells a stale
# published value from a fresh request, must not leak into this one.
foreach(_leftover F32 F16)
  unset(ARM_NN_ENABLE_${_leftover} CACHE)
  unset(_NS_CMSIS_NN_PUBLISHED_ARM_NN_ENABLE_${_leftover} CACHE)
endforeach()

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

elseif(CASE STREQUAL "float_arm_nn_only")
  # Both NSX switches stay at their defaults, so the ARM_NN spelling is the
  # whole request. See AmbiqAI/ns-cmsis-nn#420.
  set(ARM_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 0 1)
  set(_expect_f32_define 0)
  set(_expect_f16_define 1)
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
  message(FATAL_ERROR "Unknown CASE='${CASE}'. Valid: source_all, source_subset, inline_asm, float_f32, float_f16, float_both, float_arm_nn_only, float_conflict, float_conflict_plain, prebuilt, prebuilt_manifest_ok, prebuilt_manifest_reject, prebuilt_manifest_reject_arm_nn, prebuilt_manifest_legacy.")
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

# The target's compile definitions are the compile-time truth; the cache entry
# is the configure-time mirror consumers read before any target of ours is in
# scope. Both are asserted, in both modes. See AmbiqAI/ns-cmsis-nn#420.
foreach(_width F32 F16)
  if(DEFINED _expect_${_width}_define)
    set(_want ${_expect_${_width}_define})
    if(TARGET nsx_cmsis_nn)
      if(_expect_prebuilt)
        get_target_property(_defs nsx_cmsis_nn INTERFACE_COMPILE_DEFINITIONS)
      else()
        get_target_property(_defs nsx_cmsis_nn COMPILE_DEFINITIONS)
      endif()
      list(FIND _defs "ARM_NN_ENABLE_${_width}=${_want}" _idx)
      if(_idx EQUAL -1)
        _fail("${CASE} mode: expected ARM_NN_ENABLE_${_width}=${_want} in COMPILE_DEFINITIONS; got: ${_defs}")
      endif()
    endif()
    if(NOT DEFINED CACHE{ARM_NN_ENABLE_${_width}})
      _fail("${CASE} mode: ARM_NN_ENABLE_${_width} was not published to the cache")
    elseif(NOT "$CACHE{ARM_NN_ENABLE_${_width}}" EQUAL "${_want}")
      _fail("${CASE} mode: cache ARM_NN_ENABLE_${_width}='$CACHE{ARM_NN_ENABLE_${_width}}', expected ${_want}")
    endif()
  endif()
endforeach()

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
