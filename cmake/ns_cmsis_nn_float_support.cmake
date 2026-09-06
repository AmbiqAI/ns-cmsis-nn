#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# The query consumers use to learn which float widths the ns-cmsis-nn library
# in scope was actually built with.
#
#   ns_cmsis_nn_float_support(F32 <out_var> F16 <out_var> [TARGET <target>])
#
# Both width keywords are required and each names a variable set to ON or OFF
# in the caller's scope. The answer comes from the library target's compile
# definitions, which is where the build wrote the decision, so it is the same
# answer in a source build, against a prebuilt archive, and after
# find_package(). Reading ARM_NN_ENABLE_F32/F16 out of the cache instead only
# works on the paths that take it as an input.
#
# TARGET names the library to answer for. Without it the target is resolved
# from what is in scope: NS_CMSIS_NN_FLOAT_TARGET (a global property the Zephyr
# module sets, since its library target name is chosen by Zephyr), cmsis-nn,
# nsx_cmsis_nn, ns_cmsis_nn_prebuilt. A project can hold more than one of them
# at once, a standalone cmsis-nn next to the NSX module's nsx_cmsis_nn for
# instance, and they need not agree on the float widths, so that is a
# FATAL_ERROR naming the candidates rather than an answer from whichever came
# first. No target at all is a FATAL_ERROR too, rather than a silent OFF/OFF.
#
# Kept in its own module so entry points that compile nothing can offer the
# query without pulling in the source layout, and mirrored in
# cmake/templates/ns-cmsis-nn-config.cmake.in for find_package() consumers.
#
# See AmbiqAI/ns-cmsis-nn#420.
#

if(DEFINED NS_CMSIS_NN_FLOAT_SUPPORT_INCLUDED)
  return()
endif()
set(NS_CMSIS_NN_FLOAT_SUPPORT_INCLUDED TRUE)

# Mirrored verbatim in cmake/templates/ns-cmsis-nn-config.cmake.in; a change
# here belongs there too.
function(ns_cmsis_nn_float_support)
  cmake_parse_arguments(PARSE_ARGV 0 NSFS "" "F32;F16;TARGET" "")
  if(NSFS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "ns_cmsis_nn_float_support: unexpected arguments: "
      "${NSFS_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_width F32 F16)
    if(NOT NSFS_${_width})
      message(FATAL_ERROR
        "ns_cmsis_nn_float_support: ${_width} is required. Pass "
        "${_width} <out_var>; an omitted width would leave the caller's "
        "variable at whatever it already held.")
    endif()
  endforeach()

  if(NSFS_TARGET)
    if(NOT TARGET ${NSFS_TARGET})
      message(FATAL_ERROR
        "ns_cmsis_nn_float_support: TARGET ${NSFS_TARGET} is not a target in "
        "this scope.")
    endif()
    set(_target ${NSFS_TARGET})
  else()
    set(_candidates "")
    get_property(_declared GLOBAL PROPERTY NS_CMSIS_NN_FLOAT_TARGET)
    if(_declared AND TARGET ${_declared})
      list(APPEND _candidates ${_declared})
    endif()
    foreach(_candidate cmsis-nn nsx_cmsis_nn ns_cmsis_nn_prebuilt)
      if(TARGET ${_candidate})
        list(APPEND _candidates ${_candidate})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _candidates)
    list(LENGTH _candidates _candidate_count)
    if(_candidate_count EQUAL 0)
      message(FATAL_ERROR
        "ns_cmsis_nn_float_support: no ns-cmsis-nn library target is in scope. "
        "Call it after add_subdirectory(), after the Zephyr module has run, or "
        "after find_package(ns-cmsis-nn).")
    elseif(_candidate_count GREATER 1)
      string(REPLACE ";" ", " _named "${_candidates}")
      message(FATAL_ERROR
        "ns_cmsis_nn_float_support: ${_candidate_count} ns-cmsis-nn library "
        "targets are in scope (${_named}) and they can report different float "
        "widths, so there is no one answer. Pass TARGET <target> to name the "
        "one this caller links against.")
    endif()
    set(_target ${_candidates})
  endif()

  set(_defs "")
  get_target_property(_iface_defs ${_target} INTERFACE_COMPILE_DEFINITIONS)
  if(_iface_defs)
    list(APPEND _defs ${_iface_defs})
  endif()
  # An INTERFACE library has no COMPILE_DEFINITIONS to read, and asking for it
  # is an error on the CMake versions the standalone build still supports.
  get_target_property(_type ${_target} TYPE)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_own_defs ${_target} COMPILE_DEFINITIONS)
    if(_own_defs)
      list(APPEND _defs ${_own_defs})
    endif()
  endif()

  foreach(_width F32 F16)
    list(FIND _defs "ARM_NN_ENABLE_${_width}=1" _idx)
    if(_idx EQUAL -1)
      set(${NSFS_${_width}} OFF PARENT_SCOPE)
    else()
      set(${NSFS_${_width}} ON PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
