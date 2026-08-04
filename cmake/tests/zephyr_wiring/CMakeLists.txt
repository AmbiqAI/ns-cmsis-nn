#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Zephyr-integration wiring test for zephyr/CMakeLists.txt.
#
# Runs entirely during CMake configure (no Zephyr SDK, no cross-compiler, no
# west, no toolchain). It stubs every zephyr_* macro the real
# zephyr/CMakeLists.txt calls so we can record exactly what sources, include
# paths, and compile definitions would be handed to a real Zephyr build.
#
# Invoke as:
#
#     cmake -S cmake/tests/zephyr_wiring -B build/zephyr_wiring
#
# A green configure proves:
#   - the Kconfig -> SSoT-group translation table is complete (every one of
#     the 23 group ids is referenced) and uses the exact group spellings the
#     SSoT exposes;
#   - turning every CONFIG_NS_CMSIS_NN_<knob>=y on resolves to all 225
#     sources via ns_cmsis_nn_attach();
#   - Include/ is exposed via zephyr_include_directories();
#   - CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY=y propagates as the
#     CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY define;
#   - the module is guarded on CONFIG_NS_CMSIS_NN.
#
cmake_minimum_required(VERSION 3.20)
# C is enabled so add_library(STATIC) (used internally by the stubs) works.
project(ns_cmsis_nn_zephyr_wiring C)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${_repo_root}/cmake/ns_cmsis_nn.cmake")

function(_compute_expected_source_count out_var enable_f32 enable_f16)
  set(ARM_NN_ENABLE_F32 ${enable_f32})
  set(ARM_NN_ENABLE_F16 ${enable_f16})

  ns_cmsis_nn_groups(_expected_groups)
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
# Test harness: collect failures, report them all at once.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Zephyr build-system stubs. Each macro records its arguments into a
# well-known cache so the assertions below can inspect them.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")
set(STUB_LIBRARY_COUNTER 0)

macro(zephyr_library)
  math(EXPR STUB_LIBRARY_COUNTER "${STUB_LIBRARY_COUNTER} + 1")
  set(STUB_LIBRARY_NAME "zephyr_stub_lib_${STUB_LIBRARY_COUNTER}")
  add_library(${STUB_LIBRARY_NAME} STATIC)
  # Mirror Zephyr's real contract: subsequent zephyr_library_* calls operate
  # on this library, exposed as ZEPHYR_CURRENT_LIBRARY.
  set(ZEPHYR_CURRENT_LIBRARY "${STUB_LIBRARY_NAME}")
endmacro()

macro(zephyr_library_compile_options)
  list(APPEND STUB_COMPILE_OPTIONS ${ARGN})
endmacro()

macro(zephyr_include_directories)
  list(APPEND STUB_GLOBAL_INCLUDE_DIRS ${ARGN})
endmacro()

macro(zephyr_library_include_directories)
  list(APPEND STUB_LIBRARY_INCLUDE_DIRS ${ARGN})
endmacro()

macro(zephyr_compile_definitions)
  list(APPEND STUB_COMPILE_DEFINITIONS ${ARGN})
endmacro()

macro(zephyr_link_libraries)
  list(APPEND STUB_APP_LINK_LIBRARIES ${ARGN})
endmacro()

# ----------------------------------------------------------------------------
# Stub the Zephyr environment that zephyr/CMakeLists.txt references.
# ----------------------------------------------------------------------------
set(ZEPHYR_CURRENT_MODULE_DIR "${_repo_root}")
# The real Zephyr build sets ZEPHYR_CMSIS_MODULE_DIR to the cmsis module root.
# We don't care where it points for wiring purposes — only that the macro
# captures it. Use a sentinel so the assertions can identify it.
set(ZEPHYR_CMSIS_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/__fake_cmsis_module__")

# ----------------------------------------------------------------------------
# Case 1: CONFIG_NS_CMSIS_NN not set — module must be a no-op.
# ----------------------------------------------------------------------------
include("${_repo_root}/zephyr/CMakeLists.txt")
if(STUB_LIBRARY_NAME)
  _fail("guard regression: zephyr/CMakeLists.txt created a library even with CONFIG_NS_CMSIS_NN unset")
endif()

# ----------------------------------------------------------------------------
# Case 2: every CONFIG_NS_CMSIS_NN_<knob>=y. The renamed knobs only — the
# deprecated aliases are Kconfig-level select sugar and never appear as
# CONFIG_NS_CMSIS_NN_<DEPRECATED>=y in a real build (Kconfig rewrites them).
# ----------------------------------------------------------------------------
set(_renamed_knobs
    ACTIVATION
    BASICMATHSNN
    COMPARISON
    CONCATENATION
    CONVOLUTION
    FULLYCONNECTED
    GATHER
    LSTM
    NNSUPPORT
    PAD
    POOLING
    QUANTIZATION
    RESHAPE
    SOFTMAX
    STRIDEDSLICE
    SVDF
    TRANSPOSE
    TILE
    BROADCAST
    SCATTER
    SELECT
    REVERSESEQUENCE
    DYNAMICUPDATESLICE
)

set(CONFIG_NS_CMSIS_NN 1)
set(CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY 1)
foreach(_k IN LISTS _renamed_knobs)
  set(CONFIG_NS_CMSIS_NN_${_k} 1)
endforeach()

include("${_repo_root}/zephyr/CMakeLists.txt")

# ----------------------------------------------------------------------------
# Assertions on the captured wiring.
# ----------------------------------------------------------------------------
_compute_expected_source_count(_expected_default 0 0)

# Library was created.
if(NOT STUB_LIBRARY_NAME)
  _fail("zephyr_library() was never called; wiring not reached")
endif()

# All 225 sources attached.
if(TARGET ${STUB_LIBRARY_NAME})
  get_target_property(_srcs ${STUB_LIBRARY_NAME} SOURCES)
  if(NOT _srcs)
    _fail("no sources attached to the Zephyr library")
  else()
    list(LENGTH _srcs _nsrc)
    if(NOT _nsrc EQUAL _expected_default)
      _fail("Zephyr wiring attached ${_nsrc} sources; expected ${_expected_default} (SSoT ALL).\n      A Kconfig->group mapping typo will show up here.")
    endif()
  endif()
endif()

# Public Include/ exposed globally (consumers include arm_nn*.h directly).
list(FIND STUB_GLOBAL_INCLUDE_DIRS "${_repo_root}/Include" _inc_idx)
if(_inc_idx EQUAL -1)
  _fail("Include/ was not exposed via zephyr_include_directories(); got: ${STUB_GLOBAL_INCLUDE_DIRS}")
endif()

# CMSIS Core/Include forwarded to the library.
list(FIND STUB_LIBRARY_INCLUDE_DIRS "${ZEPHYR_CMSIS_MODULE_DIR}/CMSIS/Core/Include" _cmsis_idx)
if(_cmsis_idx EQUAL -1)
  _fail("CMSIS Core/Include not forwarded; got: ${STUB_LIBRARY_INCLUDE_DIRS}")
endif()

# Inline-asm requantize define propagated globally.
list(FIND STUB_COMPILE_DEFINITIONS "CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY" _def_idx)
if(_def_idx EQUAL -1)
  _fail("CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY did not propagate as CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY")
endif()

# Float feature toggles default to OFF in this case.
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F32=0" _f32_off_idx)
if(_f32_off_idx EQUAL -1)
  _fail("default float wiring: ARM_NN_ENABLE_F32=0 compile definition missing")
endif()
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F16=0" _f16_off_idx)
if(_f16_off_idx EQUAL -1)
  _fail("default float wiring: ARM_NN_ENABLE_F16=0 compile definition missing")
endif()

set(_srcs_all_groups_default "${_srcs}")

# ----------------------------------------------------------------------------
# Translation-table coverage: every renamed Kconfig knob in this test must
# also appear in the real zephyr/CMakeLists.txt foreach() pair list. Read
# the file and grep for each knob to catch any typo (e.g. "FULLYCONNETED").
# ----------------------------------------------------------------------------
file(READ "${_repo_root}/zephyr/CMakeLists.txt" _zcml)
foreach(_k IN LISTS _renamed_knobs)
  string(FIND "${_zcml}" "\"${_k};" _hit)
  if(_hit EQUAL -1)
    _fail("translation table is missing Kconfig knob '${_k}' (or its spelling drifted)")
  endif()
endforeach()

# Coverage cross-check: the translation table must reference EVERY group the
# SSoT exposes. Pull the group list and verify each lowercase id appears as
# the value half of a "KNOB;group" pair.
ns_cmsis_nn_groups(_all_groups)
foreach(_g IN LISTS _all_groups)
  string(FIND "${_zcml}" ";${_g}\"" _hit)
  if(_hit EQUAL -1)
    _fail("SSoT group '${_g}' has no Kconfig knob in the Zephyr translation table")
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Case 2b: float32 enabled.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")

set(CONFIG_NS_CMSIS_NN_ENABLE_F32 1)
unset(CONFIG_NS_CMSIS_NN_ENABLE_F16)

include("${_repo_root}/zephyr/CMakeLists.txt")
_compute_expected_source_count(_expected_f32 1 0)

if(NOT STUB_LIBRARY_NAME)
  _fail("float32 case: zephyr_library() was not called")
else()
  get_target_property(_srcs_f32 ${STUB_LIBRARY_NAME} SOURCES)
  if(NOT _srcs_f32)
    _fail("float32 case: no sources attached to Zephyr library")
  else()
    list(LENGTH _srcs_f32 _nsrc_f32)
    if(NOT _nsrc_f32 EQUAL _expected_f32)
      _fail("float32 case: expected ${_expected_f32} sources with float enabled, got ${_nsrc_f32}")
    endif()
    list(FILTER _srcs_f32 INCLUDE REGEX ".*(_f32\\.c)$")
    list(LENGTH _srcs_f32 _n_f32_files)
    if(NOT _n_f32_files GREATER 0)
      _fail("float32 case: no _f32.c sources selected")
    endif()
  endif()
endif()

list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F32=1" _f32_on_idx)
if(_f32_on_idx EQUAL -1)
  _fail("float32 case: ARM_NN_ENABLE_F32=1 compile definition missing")
endif()
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F16=0" _f16_still_off_idx)
if(_f16_still_off_idx EQUAL -1)
  _fail("float32 case: ARM_NN_ENABLE_F16=0 compile definition missing")
endif()

# ----------------------------------------------------------------------------
# Case 2c: float16 enabled.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")

unset(CONFIG_NS_CMSIS_NN_ENABLE_F32)
set(CONFIG_NS_CMSIS_NN_ENABLE_F16 1)

include("${_repo_root}/zephyr/CMakeLists.txt")
_compute_expected_source_count(_expected_f16 0 1)

if(NOT STUB_LIBRARY_NAME)
  _fail("float16 case: zephyr_library() was not called")
else()
  get_target_property(_srcs_f16 ${STUB_LIBRARY_NAME} SOURCES)
  if(NOT _srcs_f16)
    _fail("float16 case: no sources attached to Zephyr library")
  else()
    list(LENGTH _srcs_f16 _nsrc_f16)
    if(NOT _nsrc_f16 EQUAL _expected_f16)
      _fail("float16 case: expected ${_expected_f16} sources with float enabled, got ${_nsrc_f16}")
    endif()
    list(FILTER _srcs_f16 INCLUDE REGEX ".*(_f16\\.c)$")
    list(LENGTH _srcs_f16 _n_f16_files)
    if(NOT _n_f16_files GREATER 0)
      _fail("float16 case: no _f16.c sources selected")
    endif()
  endif()
endif()

list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F32=0" _f32_still_off_idx)
if(_f32_still_off_idx EQUAL -1)
  _fail("float16 case: ARM_NN_ENABLE_F32=0 compile definition missing")
endif()
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F16=1" _f16_on_idx)
if(_f16_on_idx EQUAL -1)
  _fail("float16 case: ARM_NN_ENABLE_F16=1 compile definition missing")
endif()

# ----------------------------------------------------------------------------
# Case 2d: both float32 and float16 enabled.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")

set(CONFIG_NS_CMSIS_NN_ENABLE_F32 1)
set(CONFIG_NS_CMSIS_NN_ENABLE_F16 1)

include("${_repo_root}/zephyr/CMakeLists.txt")
_compute_expected_source_count(_expected_both 1 1)

if(NOT STUB_LIBRARY_NAME)
  _fail("float32+float16 case: zephyr_library() was not called")
else()
  get_target_property(_srcs_both ${STUB_LIBRARY_NAME} SOURCES)
  if(NOT _srcs_both)
    _fail("float32+float16 case: no sources attached to Zephyr library")
  else()
    list(LENGTH _srcs_both _nsrc_both)
    if(NOT _nsrc_both EQUAL _expected_both)
      _fail("float32+float16 case: expected ${_expected_both} sources with float enabled, got ${_nsrc_both}")
    endif()

    set(_srcs_both_f32 ${_srcs_both})
    list(FILTER _srcs_both_f32 INCLUDE REGEX ".*(_f32\\.c)$")
    list(LENGTH _srcs_both_f32 _n_both_f32_files)
    if(NOT _n_both_f32_files GREATER 0)
      _fail("float32+float16 case: no _f32.c sources selected")
    endif()

    set(_srcs_both_f16 ${_srcs_both})
    list(FILTER _srcs_both_f16 INCLUDE REGEX ".*(_f16\\.c)$")
    list(LENGTH _srcs_both_f16 _n_both_f16_files)
    if(NOT _n_both_f16_files GREATER 0)
      _fail("float32+float16 case: no _f16.c sources selected")
    endif()
  endif()
endif()

list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F32=1" _f32_both_idx)
if(_f32_both_idx EQUAL -1)
  _fail("float32+float16 case: ARM_NN_ENABLE_F32=1 compile definition missing")
endif()
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F16=1" _f16_both_idx)
if(_f16_both_idx EQUAL -1)
  _fail("float32+float16 case: ARM_NN_ENABLE_F16=1 compile definition missing")
endif()

# ----------------------------------------------------------------------------
# Case 3: prebuilt mode. CONFIG_NS_CMSIS_NN_USE_PREBUILT=y diverts the
# module away from the source path entirely: no zephyr_library(), no source
# attachment, no SSoT-group translation. Instead, an IMPORTED static lib
# pointing at the staged SDK tarball is created and linked at app level.
# ----------------------------------------------------------------------------

# Reset stub state.
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")
foreach(_k IN LISTS _renamed_knobs)
  unset(CONFIG_NS_CMSIS_NN_${_k})
endforeach()
unset(CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY)
unset(CONFIG_NS_CMSIS_NN_ENABLE_F32)
unset(CONFIG_NS_CMSIS_NN_ENABLE_F16)

# Stage a fake SDK layout (lib/libns-cmsis-nn.a + include/).
set(_fake_sdk "${CMAKE_BINARY_DIR}/_fake_sdk")
file(MAKE_DIRECTORY "${_fake_sdk}/lib" "${_fake_sdk}/include")
file(WRITE "${_fake_sdk}/lib/libns-cmsis-nn.a" "fake archive\n")
file(WRITE "${_fake_sdk}/include/arm_nnfunctions.h" "/* fake */\n")

set(CONFIG_NS_CMSIS_NN 1)
set(CONFIG_NS_CMSIS_NN_USE_PREBUILT 1)
set(CONFIG_NS_CMSIS_NN_PREBUILT_PATH "${_fake_sdk}")

include("${_repo_root}/zephyr/CMakeLists.txt")

# In prebuilt mode the module must NOT call zephyr_library(); attaching
# sources to a module-level STATIC alongside an IMPORTED library would be
# strictly wasteful.
if(STUB_LIBRARY_NAME)
  _fail("prebuilt mode: zephyr_library() was called (expected none); STUB_LIBRARY_NAME='${STUB_LIBRARY_NAME}'")
endif()
if(NOT TARGET ns_cmsis_nn_prebuilt)
  _fail("prebuilt mode: ns_cmsis_nn_prebuilt IMPORTED target was not created")
else()
  get_target_property(_imp ns_cmsis_nn_prebuilt IMPORTED)
  if(NOT _imp)
    _fail("prebuilt mode: ns_cmsis_nn_prebuilt is not IMPORTED")
  endif()
  get_target_property(_loc ns_cmsis_nn_prebuilt IMPORTED_LOCATION)
  if(NOT _loc STREQUAL "${_fake_sdk}/lib/libns-cmsis-nn.a")
    _fail("prebuilt mode: IMPORTED_LOCATION='${_loc}', expected '${_fake_sdk}/lib/libns-cmsis-nn.a'")
  endif()
  get_target_property(_incs ns_cmsis_nn_prebuilt INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _incs STREQUAL "${_fake_sdk}/include")
    _fail("prebuilt mode: INTERFACE_INCLUDE_DIRECTORIES='${_incs}', expected '${_fake_sdk}/include'")
  endif()
endif()
list(FIND STUB_APP_LINK_LIBRARIES "ns_cmsis_nn_prebuilt" _link_idx)
if(_link_idx EQUAL -1)
  _fail("prebuilt mode: ns_cmsis_nn_prebuilt not added to app via zephyr_link_libraries(); got: ${STUB_APP_LINK_LIBRARIES}")
endif()
list(FIND STUB_GLOBAL_INCLUDE_DIRS "${_fake_sdk}/include" _inc_idx)
if(_inc_idx EQUAL -1)
  _fail("prebuilt mode: ${_fake_sdk}/include not exposed globally; got: ${STUB_GLOBAL_INCLUDE_DIRS}")
endif()

# ----------------------------------------------------------------------------
# Case 3b: prebuilt mode with a manifest.json reporting f32=true. Requesting
# CONFIG_NS_CMSIS_NN_ENABLE_F32=y must succeed and forward ARM_NN_ENABLE_F32=1.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")
set(STUB_APP_LINK_LIBRARIES "")

file(WRITE "${_fake_sdk}/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": true, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")
set(CONFIG_NS_CMSIS_NN_ENABLE_F32 1)
unset(CONFIG_NS_CMSIS_NN_ENABLE_F16)

include("${_repo_root}/zephyr/CMakeLists.txt")

list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F32=1" _prebuilt_f32_idx)
if(_prebuilt_f32_idx EQUAL -1)
  _fail("prebuilt manifest-ok case: ARM_NN_ENABLE_F32=1 missing from compile definitions; got: ${STUB_COMPILE_DEFINITIONS}")
endif()
list(FIND STUB_COMPILE_DEFINITIONS "ARM_NN_ENABLE_F16=0" _prebuilt_f16_idx)
if(_prebuilt_f16_idx EQUAL -1)
  _fail("prebuilt manifest-ok case: ARM_NN_ENABLE_F16=0 missing from compile definitions; got: ${STUB_COMPILE_DEFINITIONS}")
endif()

# ----------------------------------------------------------------------------
# Case 3c: prebuilt mode, manifest reports f32=false. Requesting
# CONFIG_NS_CMSIS_NN_ENABLE_F32=y must FATAL_ERROR at configure time rather
# than silently producing an undefined-reference link error later. Driven
# via a child cmake invocation (the FATAL_ERROR would otherwise abort this
# whole test script).
# ----------------------------------------------------------------------------
file(WRITE "${_fake_sdk}/manifest.json" "{
  \"schema_version\": 2,
  \"features\": { \"integer\": true, \"f32\": false, \"f16\": false, \"requantize_inline_asm\": false, \"verified\": true }
}
")

set(_zephyr_probe "${CMAKE_BINARY_DIR}/_zephyr_manifest_probe")
file(REMOVE_RECURSE "${_zephyr_probe}")
file(MAKE_DIRECTORY "${_zephyr_probe}")
file(WRITE "${_zephyr_probe}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.20)
project(zephyr_manifest_reject_probe C)
macro(zephyr_library)
endmacro()
macro(zephyr_library_compile_options)
endmacro()
macro(zephyr_include_directories)
endmacro()
macro(zephyr_library_include_directories)
endmacro()
macro(zephyr_compile_definitions)
endmacro()
macro(zephyr_link_libraries)
endmacro()
set(ZEPHYR_CURRENT_MODULE_DIR \"${_repo_root}\")
set(ZEPHYR_CMSIS_MODULE_DIR \"${CMAKE_CURRENT_BINARY_DIR}/__fake_cmsis_module__\")
set(CONFIG_NS_CMSIS_NN 1)
set(CONFIG_NS_CMSIS_NN_USE_PREBUILT 1)
set(CONFIG_NS_CMSIS_NN_PREBUILT_PATH \"${_fake_sdk}\")
set(CONFIG_NS_CMSIS_NN_ENABLE_F32 1)
include(\"${_repo_root}/zephyr/CMakeLists.txt\")
")
execute_process(
  COMMAND ${CMAKE_COMMAND} -S "${_zephyr_probe}" -B "${_zephyr_probe}/_b"
  RESULT_VARIABLE _zephyr_probe_rc
  OUTPUT_VARIABLE _zephyr_probe_out
  ERROR_VARIABLE  _zephyr_probe_err)
if(_zephyr_probe_rc EQUAL 0)
  _fail("prebuilt manifest-reject case: probe unexpectedly succeeded (expected FATAL_ERROR)\n${_zephyr_probe_out}\n${_zephyr_probe_err}")
else()
  string(FIND "${_zephyr_probe_err}" "CONFIG_NS_CMSIS_NN_ENABLE_F32=y but the prebuilt SDK's manifest" _zephyr_probe_msg_idx)
  if(_zephyr_probe_msg_idx EQUAL -1)
    _fail("prebuilt manifest-reject case: probe failed but without the expected diagnostic; stderr:\n${_zephyr_probe_err}")
  endif()
endif()

# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------
if(_failures)
  message(FATAL_ERROR "Zephyr wiring test FAILED:${_failures}")
endif()

message(STATUS "Zephyr wiring OK: 23 knobs translated, baseline source set attached, float source gates validated, inline-asm define propagated, prebuilt mode imports SDK tarball, prebuilt manifest validation accepts/rejects float requests.")
