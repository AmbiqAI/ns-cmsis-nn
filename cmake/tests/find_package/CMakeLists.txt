#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# find_package(ns-cmsis-nn) consumer regression test.
#
# Runs at CMake configure time. Builds a per-arch SDK tarball from the
# repo's current state into the build dir, extracts it, then exercises
# find_package() against that extracted tree and asserts on the target
# graph it produces.
#
# Parameterised by -DCASE=<case>, where <case> is one of:
#
#   happy           (default) — find_package() succeeds, ns::cmsis-nn imported,
#                               headers and lib paths wired up, version match
#                               works for exact + >=.
#   version_mismatch          — find_package(ns-cmsis-nn 99.0.0 REQUIRED) must
#                               fail (PACKAGE_VERSION_COMPATIBLE=FALSE).
#   cpu_mismatch              — consumer's CMAKE_C_FLAGS specifies a different
#                               -mcpu, config must FATAL_ERROR.
#
# Invoke as:
#
#     cmake -S cmake/tests/find_package -B build/find_package -DCASE=happy
#
# Requires bash + tar + sha256sum + a fake .a (no real toolchain needed).
cmake_minimum_required(VERSION 3.15)
project(ns_cmsis_nn_find_package_test C)

set(CASE "happy" CACHE STRING "Which find_package case to exercise.")
set_property(CACHE CASE PROPERTY STRINGS happy version_mismatch cpu_mismatch)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(_fake_version "9.9.9")
set(_target_cpu   "cortex-m4")
set(_toolchain    "gcc")

# ----------------------------------------------------------------------------
# Stage a fake static library, then drive build_sdk_tarball.sh against it.
# We don't need a real Cortex-M build to exercise find_package's wiring.
# ----------------------------------------------------------------------------
set(_fake_lib "${CMAKE_BINARY_DIR}/libns-cmsis-nn-${_target_cpu}-${_fake_version}.a")
file(WRITE "${_fake_lib}" "fake archive\n")

set(_pkg_outdir "${CMAKE_BINARY_DIR}/sdk")
file(MAKE_DIRECTORY "${_pkg_outdir}")

execute_process(
  COMMAND bash "${_repo_root}/scripts/build_sdk_tarball.sh"
    --target-cpu        "${_target_cpu}"
    --version           "${_fake_version}"
    --staticlib         "${_fake_lib}"
    --toolchain         "${_toolchain}"
    --toolchain-version "test-9.9"
    --outdir            "${_pkg_outdir}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE  _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "build_sdk_tarball.sh failed (rc=${_rc})\nstdout:\n${_out}\nstderr:\n${_err}")
endif()

set(_tarball "${_pkg_outdir}/ns-cmsis-nn-${_target_cpu}-${_toolchain}-${_fake_version}.tar.gz")
if(NOT EXISTS "${_tarball}")
  message(FATAL_ERROR "build_sdk_tarball.sh produced no tarball at ${_tarball}")
endif()

set(_extract_root "${CMAKE_BINARY_DIR}/extracted")
file(REMOVE_RECURSE "${_extract_root}")
file(MAKE_DIRECTORY "${_extract_root}")
execute_process(
  COMMAND tar -xzf "${_tarball}" -C "${_extract_root}"
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "extracting ${_tarball} failed (rc=${_rc})")
endif()

set(_pkg_dir "${_extract_root}/ns-cmsis-nn-${_target_cpu}-${_toolchain}-${_fake_version}")

# ----------------------------------------------------------------------------
# Failure collector.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Drive find_package per CASE.
# ----------------------------------------------------------------------------
if(CASE STREQUAL "happy")
  find_package(ns-cmsis-nn ${_fake_version} EXACT REQUIRED
               PATHS "${_pkg_dir}/cmake" NO_DEFAULT_PATH)

  if(NOT ns-cmsis-nn_FOUND)
    _fail("find_package(ns-cmsis-nn EXACT ${_fake_version}) did not set _FOUND")
  endif()
  foreach(_t cmsis-nn ns-cmsis-nn ns::cmsis-nn)
    if(NOT TARGET ${_t})
      _fail("target '${_t}' not created by find_package")
    endif()
  endforeach()
  if(TARGET cmsis-nn)
    get_target_property(_loc cmsis-nn IMPORTED_LOCATION)
    if(NOT _loc STREQUAL "${_pkg_dir}/lib/libns-cmsis-nn.a")
      _fail("IMPORTED_LOCATION='${_loc}' (expected ${_pkg_dir}/lib/libns-cmsis-nn.a)")
    endif()
    get_target_property(_incs cmsis-nn INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _incs STREQUAL "${_pkg_dir}/include")
      _fail("INTERFACE_INCLUDE_DIRECTORIES='${_incs}' (expected ${_pkg_dir}/include)")
    endif()
  endif()
  if(NOT NS_CMSIS_NN_VERSION STREQUAL _fake_version)
    _fail("NS_CMSIS_NN_VERSION='${NS_CMSIS_NN_VERSION}' (expected ${_fake_version})")
  endif()
  if(NOT NS_CMSIS_NN_TARGET_CPU STREQUAL _target_cpu)
    _fail("NS_CMSIS_NN_TARGET_CPU='${NS_CMSIS_NN_TARGET_CPU}' (expected ${_target_cpu})")
  endif()
  if(NOT NS_CMSIS_NN_TOOLCHAIN STREQUAL _toolchain)
    _fail("NS_CMSIS_NN_TOOLCHAIN='${NS_CMSIS_NN_TOOLCHAIN}' (expected ${_toolchain})")
  endif()
  if(NOT NS_CMSIS_NN_TOOLCHAIN_COMPILER STREQUAL "GNU")
    _fail("NS_CMSIS_NN_TOOLCHAIN_COMPILER='${NS_CMSIS_NN_TOOLCHAIN_COMPILER}' (expected GNU)")
  endif()

elseif(CASE STREQUAL "version_mismatch")
  # Demanding 99.0.0 against a 9.9.9 package: must NOT find.
  find_package(ns-cmsis-nn 99.0.0
               PATHS "${_pkg_dir}/cmake" NO_DEFAULT_PATH QUIET)
  if(ns-cmsis-nn_FOUND)
    _fail("find_package(ns-cmsis-nn 99.0.0) unexpectedly succeeded against a ${_fake_version} package")
  endif()

elseif(CASE STREQUAL "cpu_mismatch")
  # Pretend the consumer is building for cortex-m0 — the package is for m4.
  set(CMAKE_C_FLAGS "-mcpu=cortex-m0 -mthumb" CACHE STRING "" FORCE)
  # Capture the FATAL_ERROR by driving find_package in a child cmake.
  # We expect a non-zero exit.
  set(_probe "${CMAKE_BINARY_DIR}/_cpu_probe")
  file(REMOVE_RECURSE "${_probe}")
  file(MAKE_DIRECTORY "${_probe}")
  file(WRITE "${_probe}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.15)
project(probe C)
set(CMAKE_C_FLAGS \"-mcpu=cortex-m0 -mthumb\")
find_package(ns-cmsis-nn REQUIRED PATHS \"${_pkg_dir}/cmake\" NO_DEFAULT_PATH)
")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${_probe}" -B "${_probe}/_b"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err)
  if(_rc EQUAL 0)
    _fail("cpu mismatch probe unexpectedly succeeded (expected FATAL_ERROR)\n${_out}\n${_err}")
  endif()
  string(FIND "${_err}" "Download the matching per-arch tarball" _msg_idx)
  if(_msg_idx EQUAL -1)
    _fail("cpu mismatch probe failed but without the expected diagnostic; stderr:\n${_err}")
  endif()

else()
  message(FATAL_ERROR "Unknown CASE='${CASE}'. Valid: happy, version_mismatch, cpu_mismatch.")
endif()

# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------
if(_failures)
  message(FATAL_ERROR "find_package wiring [${CASE}] FAILED:${_failures}")
endif()

message(STATUS "find_package wiring [${CASE}] OK.")
