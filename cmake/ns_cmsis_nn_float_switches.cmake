#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Single source of truth for the float feature switches.
#
# Every entry point into this repository takes the float request under its own
# option spelling and hands it to ns_cmsis_nn_publish_float_switches():
#
#   top-level CMakeLists.txt   ARM_NN_ENABLE_F32 / ARM_NN_ENABLE_F16
#   nsx/CMakeLists.txt         NSX_CMSIS_NN_ENABLE_F32 / _F16
#   zephyr/CMakeLists.txt      CONFIG_NS_CMSIS_NN_ENABLE_F32 / _F16
#
# The function publishes the effective values under the one pair of names the
# kernels, the public headers, and the downstream Ambiq consumers all read.
#
#   ns_cmsis_nn_publish_float_switches(F32 <value> F16 <value>
#                                      [REQUESTED_BY <text>])
#
# It writes each value to two places:
#
#   - the CMake cache, as BOOL 0/1, for consumers that add this repository as
#     a subdirectory or a module and read the switches before any target of
#     ours is in scope (heliaRT's nsx module, heliaAOT's generated module);
#   - the calling directory scope, as 0/1, for the caller's own
#     target_compile_definitions() and for ns_cmsis_nn_group_sources().
#
# The library target's compile definitions remain the compile-time truth; the
# cache entry is a configure-time mirror of the same decision, so the two
# cannot drift.
#
# Setting both spellings to contradicting values is a FATAL_ERROR rather than
# a precedence rule: only the caller knows which one it meant, and silently
# picking a winner is what produced the mismatch in the first place. A cache
# entry this function wrote on an earlier configure is stale, not a
# contradiction, and is overwritten without complaint.
#
# See AmbiqAI/ns-cmsis-nn#420.
#

if(DEFINED NS_CMSIS_NN_FLOAT_SWITCHES_INCLUDED)
  return()
endif()
set(NS_CMSIS_NN_FLOAT_SWITCHES_INCLUDED TRUE)

function(ns_cmsis_nn_publish_float_switches)
  cmake_parse_arguments(NSF "" "F32;F16;REQUESTED_BY" "" ${ARGN})
  if(NSF_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "ns_cmsis_nn_publish_float_switches: unexpected arguments: "
      "${NSF_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT NSF_REQUESTED_BY)
    set(NSF_REQUESTED_BY "this build")
  endif()

  foreach(_width F32 F16)
    set(_name "ARM_NN_ENABLE_${_width}")
    set(_sentinel "_NS_CMSIS_NN_PUBLISHED_${_name}")
    string(REGEX REPLACE "^F" "float" _pretty "${_width}")

    if(NSF_${_width})
      set(_want 1)
    else()
      set(_want 0)
    endif()

    set(_doc "Enable the CMSIS-NN ${_pretty} extensions.")
    if(DEFINED CACHE{${_name}})
      if($CACHE{${_name}})
        set(_cached 1)
      else()
        set(_cached 0)
      endif()
      if(NOT "${_cached}" EQUAL "${_want}"
         AND NOT "$CACHE{${_sentinel}}" STREQUAL "${_cached}")
        message(FATAL_ERROR
          "ns-cmsis-nn: contradictory float switches.\n"
          "  ${NSF_REQUESTED_BY} resolves ${_name} to ${_want}, but the cache "
          "already carries ${_name}=$CACHE{${_name}} from somewhere else.\n"
          "  On this entry path ${_name} is published output, not input: set "
          "the entry point's own switch and let it publish, or set ${_name} "
          "alone and drop the other spelling.")
      endif()
      get_property(_doc CACHE "${_name}" PROPERTY HELPSTRING)
    endif()

    set(${_name} ${_want} CACHE BOOL "${_doc}" FORCE)
    set(${_sentinel} ${_want} CACHE INTERNAL
      "Effective ${_name} last published by ns-cmsis-nn.")
    set(${_name} ${_want} PARENT_SCOPE)
  endforeach()
endfunction()
