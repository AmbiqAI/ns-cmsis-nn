#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# SSoT contract test for cmake/ns_cmsis_nn.cmake.
#
# Runs entirely during CMake configure (no compiler required, no project
# language enabled). Invoke as:
#
#     cmake -S cmake/tests/ssot_contract -B build/ssot_contract
#
# A green configure means the contract holds. Any drift is reported as a
# FATAL_ERROR with the offending group(s) and counts.
#
# This locks the SSoT against silent drift from upstream Arm syncs by
# asserting:
#
#   1. ns_cmsis_nn_groups() returns the expected ordered list of group ids.
#   2. Each group resolves to the expected number of source files.
#   3. The GROUPS ALL DTYPES ALL total matches the baseline.
#   4. Every source path the SSoT returns exists on disk.
#   5. ns_cmsis_nn_attach() is callable and produces the same source set on a
#      throw-away INTERFACE target.
#   6. DTYPES filtering removes at least one file when narrowed.
#
# When an upstream sync legitimately adds, removes, or renames sources, bump
# the EXPECTED_* values below in the same PR that updates ns_cmsis_nn.cmake.
# The error message will tell you exactly which group drifted and by how much.
#

cmake_minimum_required(VERSION 3.15)
# C is enabled so add_library(STATIC) works for the attach() smoke; no
# sources are actually compiled (the targets are configure-time probes).
project(ns_cmsis_nn_ssot_contract C)

# ----------------------------------------------------------------------------
# Locate and include the SSoT module relative to this CMakeLists.
# ----------------------------------------------------------------------------
get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${_repo_root}/cmake/ns_cmsis_nn.cmake")

# ----------------------------------------------------------------------------
# Expected group ordering. Order matters — consumers iterate in this order.
# ----------------------------------------------------------------------------
set(EXPECTED_GROUPS
    activation
    basicmath
    comparison
    concatenation
    convolution
    fullyconnected
    gather
    lstm
    pad
    pooling
    quantization
    reshape
    softmax
    stridedslice
    svd
    transpose
    tile
    broadcast
    scatter
    select
    reversesequence
    dynamicupdateslice
    nnsupport
)

# ----------------------------------------------------------------------------
# Expected per-group source counts (DTYPES = ALL). Sum must equal
# EXPECTED_TOTAL below.
# ----------------------------------------------------------------------------
set(EXPECTED_COUNT_activation     21)
set(EXPECTED_COUNT_basicmath      46)
set(EXPECTED_COUNT_comparison      2)
set(EXPECTED_COUNT_concatenation   9)
set(EXPECTED_COUNT_convolution    44)
set(EXPECTED_COUNT_fullyconnected 13)
set(EXPECTED_COUNT_gather          4)
set(EXPECTED_COUNT_lstm            2)
set(EXPECTED_COUNT_pad             4)
set(EXPECTED_COUNT_pooling          6)
set(EXPECTED_COUNT_quantization     6)
set(EXPECTED_COUNT_reshape        11)
set(EXPECTED_COUNT_softmax         4)
set(EXPECTED_COUNT_stridedslice    3)
set(EXPECTED_COUNT_svd             3)
set(EXPECTED_COUNT_transpose       2)
set(EXPECTED_COUNT_tile            2)
set(EXPECTED_COUNT_broadcast       2)
set(EXPECTED_COUNT_scatter         2)
set(EXPECTED_COUNT_select          4)
set(EXPECTED_COUNT_reversesequence 2)
set(EXPECTED_COUNT_dynamicupdateslice 2)
set(EXPECTED_COUNT_nnsupport      30)

set(EXPECTED_TOTAL 224)

# ----------------------------------------------------------------------------
# Test harness: collect failures, report them all at once.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# 1. Group ordering.
# ----------------------------------------------------------------------------
ns_cmsis_nn_groups(actual_groups)
if(NOT "${actual_groups}" STREQUAL "${EXPECTED_GROUPS}")
  _fail("ns_cmsis_nn_groups() drifted.\n      expected: ${EXPECTED_GROUPS}\n      actual:   ${actual_groups}")
endif()

# ----------------------------------------------------------------------------
# 2. Per-group source counts + 4. file existence.
# ----------------------------------------------------------------------------
set(_running_total 0)
foreach(g IN LISTS EXPECTED_GROUPS)
  ns_cmsis_nn_group_sources("${g}" _src)
  list(LENGTH _src _n)
  math(EXPR _running_total "${_running_total} + ${_n}")

  set(_expected "${EXPECTED_COUNT_${g}}")
  if(NOT _n EQUAL _expected)
    _fail("group '${g}': expected ${_expected} sources, got ${_n}.\n      Bump EXPECTED_COUNT_${g} if this drift is intentional.")
  endif()

  foreach(f IN LISTS _src)
    if(NOT EXISTS "${f}")
      _fail("group '${g}': SSoT returned non-existent path: ${f}")
    endif()
  endforeach()
endforeach()

# ----------------------------------------------------------------------------
# 3. GROUPS ALL DTYPES ALL total.
# ----------------------------------------------------------------------------
if(NOT _running_total EQUAL EXPECTED_TOTAL)
  _fail("GROUPS ALL total: expected ${EXPECTED_TOTAL}, got ${_running_total}.\n      Bump EXPECTED_TOTAL if this drift is intentional.")
endif()

# ----------------------------------------------------------------------------
# 5. ns_cmsis_nn_attach() smoke: throw-away STATIC target must end up with
#    the same source set as the sum of per-group queries. The SSoT calls
#    target_sources(... PRIVATE ...), which only populates SOURCES on a
#    non-INTERFACE target. No build is performed.
# ----------------------------------------------------------------------------
add_library(_ssot_probe STATIC)
ns_cmsis_nn_attach(_ssot_probe
                   GROUPS ALL
                   INCLUDE_DIRS_VISIBILITY PUBLIC)
get_target_property(_attached_sources _ssot_probe SOURCES)
if(NOT _attached_sources)
  _fail("ns_cmsis_nn_attach(GROUPS ALL) attached no sources.")
else()
  list(LENGTH _attached_sources _na)
  if(NOT _na EQUAL EXPECTED_TOTAL)
    _fail("ns_cmsis_nn_attach attached ${_na} sources; per-group sum is ${EXPECTED_TOTAL}.")
  endif()
endif()

# ----------------------------------------------------------------------------
# 6. DTYPES filter sanity: restricting to s8 must produce strictly fewer
#    sources than ALL (otherwise the filter is silently no-op).
# ----------------------------------------------------------------------------
add_library(_ssot_probe_s8 STATIC)
ns_cmsis_nn_attach(_ssot_probe_s8
                   GROUPS ALL
                   DTYPES s8
                   INCLUDE_DIRS_VISIBILITY PUBLIC)
get_target_property(_s8_sources _ssot_probe_s8 SOURCES)
list(LENGTH _s8_sources _ns8)
if(NOT _ns8 LESS EXPECTED_TOTAL)
  _fail("DTYPES=s8 filter is a no-op: ${_ns8} sources >= ALL total ${EXPECTED_TOTAL}.")
endif()
if(_ns8 LESS 1)
  _fail("DTYPES=s8 filter dropped everything: ${_ns8} sources.")
endif()

# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------
if(_failures)
  message(FATAL_ERROR "SSoT contract test FAILED:${_failures}")
endif()

message(STATUS "SSoT contract OK: 23 groups, ${EXPECTED_TOTAL} sources (ALL), ${_ns8} sources (DTYPES=s8).")
