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
#   float_f32                 - ARM_NN_ENABLE_F32=ON; expect _f32 sources
#                               + ARM_NN_ENABLE_F32=1 and ARM_NN_ENABLE_F16=0.
#   float_f16                 - ARM_NN_ENABLE_F16=ON; expect _f16 sources
#                               + ARM_NN_ENABLE_F32=0 and ARM_NN_ENABLE_F16=1.
#   float_both                — both float toggles ON; expect both _f32/_f16
#                               sources and both compile definitions =1.
#   float_renamed_switch      - NSX_CMSIS_NN_ENABLE_F16=ON, the removed
#                               spelling; expect a configure-time FATAL_ERROR
#                               naming the switch and ARM_NN_ENABLE_F32/F16 as
#                               its replacement. Then both removed names at
#                               once, as a build directory configured against
#                               the predecessor carries them: expect one error
#                               naming both plus a recovery that configures
#                               cleanly and lands the request it names (child
#                               cmake probes).
#                               See AmbiqAI/ns-cmsis-nn#420.
#   float_query               - ns_cmsis_nn_float_support() answers from the
#                               library target in source mode, in prebuilt
#                               mode and through the find_package config
#                               template, and refuses a call that omits a
#                               width. Four child cmake probes.
#   prebuilt                  — NSX_CMSIS_NN_LIB=<dummy.a>, expect the
#                               IMPORTED prebuilt target + INTERFACE wrapper
#                               + alias.
#   prebuilt_manifest_ok      — NSX_CMSIS_NN_LIB with a sibling manifest.json
#                               reporting f32=true, ARM_NN_ENABLE_F32=ON;
#                               expect configure success + ARM_NN_ENABLE_F32=1.
#   prebuilt_manifest_reject  — same, but manifest reports f32=false; expect
#                               a configure-time FATAL_ERROR (via a child
#                               cmake probe) naming ARM_NN_ENABLE_F32, not a
#                               silent link-time failure, and expect the -U it
#                               names to clear the rejection.
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
  float_renamed_switch float_query prebuilt
  prebuilt_manifest_ok prebuilt_manifest_reject
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
  set(_probe_args "-DARM_NN_ENABLE_F32=ON")
  set(_probe_needles
      "float32 requested via ARM_NN_ENABLE_F32"
      "-UARM_NN_ENABLE_F32")
  set(_probe_recovery "-UARM_NN_ENABLE_F32")

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
# float_renamed_switch: NSX_CMSIS_NN_ENABLE_F32/F16 are gone, and the configure
# error is the migration aid. Driven via a child cmake invocation because the
# FATAL_ERROR would otherwise abort this script.
# See AmbiqAI/ns-cmsis-nn#420.
# ----------------------------------------------------------------------------
if(CASE STREQUAL "float_renamed_switch")
  set(_probe "${CMAKE_BINARY_DIR}/_renamed_probe")
  file(REMOVE_RECURSE "${_probe}")
  file(MAKE_DIRECTORY "${_probe}")
  # The probe asserts on the request that survives the recovery, so a recovery
  # that merely configures without carrying the float request over still fails.
  file(WRITE "${_probe}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
add_subdirectory(\"${_repo_root}/nsx\" nsx)
ns_cmsis_nn_float_support(F32 _got_f32 F16 _got_f16)
if(NOT \"\${_got_f32}\" STREQUAL \"OFF\" OR NOT \"\${_got_f16}\" STREQUAL \"ON\")
  message(FATAL_ERROR
    \"after recovery: F32='\${_got_f32}' F16='\${_got_f16}', expected OFF/ON\")
endif()
")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b"
            -DNSX_CMSIS_NN_ENABLE_F16=ON
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err)
  if(_rc EQUAL 0)
    _fail("${CASE}: probe unexpectedly succeeded (expected FATAL_ERROR)\n${_out}\n${_err}")
  else()
    foreach(_needle "NSX_CMSIS_NN_ENABLE_F16 was renamed to"
                    "ARM_NN_ENABLE_F32/F16")
      string(FIND "${_err}" "${_needle}" _idx)
      if(_idx EQUAL -1)
        _fail("${CASE}: diagnostic never mentions ${_needle}; stderr:\n${_err}")
      endif()
    endforeach()
  endif()

  # A build directory configured against the predecessor holds a cache entry
  # for both removed names, since it declared both. One error has to name both,
  # or the recovery it prints only gets the user as far as the next error.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b"
            -DNSX_CMSIS_NN_ENABLE_F32=OFF -DNSX_CMSIS_NN_ENABLE_F16=ON
    RESULT_VARIABLE _rc_both
    OUTPUT_VARIABLE _out_both
    ERROR_VARIABLE  _err_both)
  if(_rc_both EQUAL 0)
    _fail("${CASE}: both-name probe unexpectedly succeeded\n${_out_both}\n${_err_both}")
  else()
    # Needles stay short: CMake hard-wraps a diagnostic, so a longer span of
    # the recovery line would only match by accident of terminal width.
    foreach(_needle "NSX_CMSIS_NN_ENABLE_F32 was renamed to"
                    "NSX_CMSIS_NN_ENABLE_F16 was renamed to"
                    "-UNSX_CMSIS_NN_ENABLE_F32"
                    "-UNSX_CMSIS_NN_ENABLE_F16"
                    "-DARM_NN_ENABLE_F16=ON")
      string(FIND "${_err_both}" "${_needle}" _idx)
      if(_idx EQUAL -1)
        _fail("${CASE}: diagnostic never mentions ${_needle}; stderr:\n${_err_both}")
      endif()
    endforeach()
  endif()

  # The recovery the message prints, applied to the same build directory.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b"
            -UNSX_CMSIS_NN_ENABLE_F32 -UNSX_CMSIS_NN_ENABLE_F16
            -DARM_NN_ENABLE_F16=ON
    RESULT_VARIABLE _rc_recover
    OUTPUT_VARIABLE _out_recover
    ERROR_VARIABLE  _err_recover)
  if(NOT _rc_recover EQUAL 0)
    _fail("${CASE}: the printed recovery did not clear the migration error\n${_out_recover}\n${_err_recover}")
  endif()

  if(_failures)
    message(FATAL_ERROR "NSX wiring [${CASE}] FAILED:${_failures}")
  endif()
  message(STATUS "NSX wiring [${CASE}] OK.")
  return()
endif()

# ----------------------------------------------------------------------------
# float_query: ns_cmsis_nn_float_support() is what a consumer calls instead of
# reading the cache, so it has to answer from the library target on each way of
# getting one, and refuse a call that omits a width. Each probe asserts on the
# answer itself, so a wrong pair fails the child configure rather than only
# showing up in this script's output.
# See AmbiqAI/ns-cmsis-nn#420.
# ----------------------------------------------------------------------------
if(CASE STREQUAL "float_query")
  set(_probe "${CMAKE_BINARY_DIR}/_query_probe")
  file(REMOVE_RECURSE "${_probe}")

  # The assertion every probe ends with: the query has to return this exact
  # pair, and ON/OFF rather than whatever the target's definitions spell.
  set(_query_assert
"ns_cmsis_nn_float_support(F32 _got_f32 F16 _got_f16)
if(NOT \"\${_got_f32}\" STREQUAL \"@WANT_F32@\" OR NOT \"\${_got_f16}\" STREQUAL \"@WANT_F16@\")
  message(FATAL_ERROR
    \"ns_cmsis_nn_float_support returned F32='\${_got_f32}' F16='\${_got_f16}', \"
    \"expected F32='@WANT_F32@' F16='@WANT_F16@'\")
endif()
")

  # Source mode: the switch is the input and the static target carries it.
  string(REPLACE "@WANT_F32@" "OFF" _assert_source "${_query_assert}")
  string(REPLACE "@WANT_F16@" "ON"  _assert_source "${_assert_source}")
  file(MAKE_DIRECTORY "${_probe}/source")
  file(WRITE "${_probe}/source/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
add_subdirectory(\"${_repo_root}/nsx\" nsx)
${_assert_source}")

  # Prebuilt mode: the answer comes from the INTERFACE wrapper, and the
  # manifest next to the archive says what the archive really has.
  file(MAKE_DIRECTORY "${_probe}/prebuilt/pkg/lib")
  file(TOUCH "${_probe}/prebuilt/pkg/lib/libns-cmsis-nn.a")
  file(WRITE "${_probe}/prebuilt/pkg/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": true, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")
  string(REPLACE "@WANT_F32@" "ON"  _assert_prebuilt "${_query_assert}")
  string(REPLACE "@WANT_F16@" "OFF" _assert_prebuilt "${_assert_prebuilt}")
  file(WRITE "${_probe}/prebuilt/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
add_library(_nsx_board_flags INTERFACE)
set(NSX_BOARD_FLAGS_TARGET _nsx_board_flags)
set(NSX_CMSIS_NN_LIB \"${_probe}/prebuilt/pkg/lib/libns-cmsis-nn.a\" CACHE FILEPATH \"\" FORCE)
add_subdirectory(\"${_repo_root}/nsx\" nsx)
${_assert_prebuilt}")

  # find_package mode: the config template defines its own copy of the query,
  # reading the imported target it just created. The package is synthesized
  # from the template rather than unpacked from a tarball, so this leg does not
  # need GNU tar and runs on any host.
  set(_pkg "${_probe}/find_package/pkg")
  file(MAKE_DIRECTORY "${_pkg}/lib" "${_pkg}/include" "${_pkg}/cmake")
  file(TOUCH "${_pkg}/lib/libns-cmsis-nn.a")
  set(NS_CMSIS_NN_VERSION "0.0.0-wiring")
  set(NS_CMSIS_NN_TARGET_CPU "")
  set(NS_CMSIS_NN_ARCH_FLAGS "")
  set(NS_CMSIS_NN_TOOLCHAIN "")
  set(NS_CMSIS_NN_TOOLCHAIN_COMPILER "")
  set(NS_CMSIS_NN_HAS_F32 1)
  set(NS_CMSIS_NN_HAS_F16 0)
  set(NS_CMSIS_NN_HAS_REQUANTIZE_INLINE_ASM 0)
  configure_file("${_repo_root}/cmake/templates/ns-cmsis-nn-config.cmake.in"
                 "${_pkg}/cmake/ns-cmsis-nn-config.cmake" @ONLY)
  string(REPLACE "@WANT_F32@" "ON"  _assert_pkg "${_query_assert}")
  string(REPLACE "@WANT_F16@" "OFF" _assert_pkg "${_assert_pkg}")
  file(WRITE "${_probe}/find_package/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
find_package(ns-cmsis-nn REQUIRED PATHS \"${_pkg}\" NO_DEFAULT_PATH)
${_assert_pkg}")

  foreach(_leg source prebuilt find_package)
    if(_leg STREQUAL "source")
      set(_leg_args "-DARM_NN_ENABLE_F16=ON")
    elseif(_leg STREQUAL "prebuilt")
      set(_leg_args "-DARM_NN_ENABLE_F32=ON")
    else()
      # Nothing to ask for: the archive's own capabilities decide here.
      set(_leg_args "")
    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -S "${_probe}/${_leg}" -B "${_probe}/${_leg}/_b"
              ${_leg_args}
      RESULT_VARIABLE _rc
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE  _err)
    if(NOT _rc EQUAL 0)
      _fail("${CASE}: ${_leg} leg failed\n${_out}\n${_err}")
    endif()
  endforeach()

  # A call that omits a width would leave the caller's variable untouched, so
  # it aborts instead.
  file(MAKE_DIRECTORY "${_probe}/omitted")
  file(WRITE "${_probe}/omitted/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.19)
project(probe C)
include(\"${_repo_root}/cmake/ns_cmsis_nn_float_support.cmake\")
ns_cmsis_nn_float_support(F32 _got_f32)
")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}/omitted" -B "${_probe}/omitted/_b"
    RESULT_VARIABLE _rc_omitted
    OUTPUT_VARIABLE _out_omitted
    ERROR_VARIABLE  _err_omitted)
  if(_rc_omitted EQUAL 0)
    _fail("${CASE}: omitting F16 configured cleanly (expected FATAL_ERROR)\n${_out_omitted}")
  else()
    foreach(_needle "ns_cmsis_nn_float_support" "F16 is required")
      string(FIND "${_err_omitted}" "${_needle}" _idx)
      if(_idx EQUAL -1)
        _fail("${CASE}: diagnostic never mentions ${_needle}; stderr:\n${_err_omitted}")
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
set(ARM_NN_ENABLE_F32 OFF CACHE BOOL "" FORCE)
set(ARM_NN_ENABLE_F16 OFF CACHE BOOL "" FORCE)
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
  set(ARM_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 1 0)
  set(_expect_f32_define 1)
  set(_expect_f16_define 0)
  set(_expect_f32_sources TRUE)

elseif(CASE STREQUAL "float_f16")
  set(ARM_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
  _compute_expected_source_count(_expected_sources "ALL" 0 1)
  set(_expect_f32_define 0)
  set(_expect_f16_define 1)
  set(_expect_f16_sources TRUE)

elseif(CASE STREQUAL "float_both")
  set(ARM_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  set(ARM_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
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
  set(ARM_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
  set(_expect_prebuilt TRUE)
  set(_expect_f32_define 1)
  set(_expect_f16_define 0)

else()
  message(FATAL_ERROR "Unknown CASE='${CASE}'. Valid: source_all, source_subset, inline_asm, float_f32, float_f16, float_both, float_renamed_switch, float_query, prebuilt, prebuilt_manifest_ok, prebuilt_manifest_reject, prebuilt_manifest_legacy.")
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

# The target's compile definitions are the compile-time truth, and
# ns_cmsis_nn_float_support() is what a consumer asks instead of reading the
# cache, so both are asserted in both modes. CMake variable names are
# case-sensitive and the per-case expectations are lowercase, so the width is
# lowered here rather than spelled a second time: a guard whose name no case
# sets reads as "this case has no float expectation" and asserts nothing.
# See AmbiqAI/ns-cmsis-nn#420.
if(TARGET nsx_cmsis_nn
   AND (DEFINED _expect_f32_define OR DEFINED _expect_f16_define))
  ns_cmsis_nn_float_support(F32 _query_f32 F16 _query_f16)
endif()
foreach(_width F32 F16)
  string(TOLOWER "${_width}" _w)
  if(DEFINED _expect_${_w}_define)
    set(_want ${_expect_${_w}_define})
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
      if(_want EQUAL 1)
        set(_want_query ON)
      else()
        set(_want_query OFF)
      endif()
      if(NOT "${_query_${_w}}" STREQUAL "${_want_query}")
        _fail("${CASE} mode: ns_cmsis_nn_float_support reported ${_width}='${_query_${_w}}', expected ${_want_query}")
      endif()
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
