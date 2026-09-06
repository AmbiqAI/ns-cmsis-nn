#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# The query consumers use to learn which float widths the ns-cmsis-nn library
# in scope was actually built with.
#
#   ns_cmsis_nn_float_support(F32 <out_var> F16 <out_var>)
#
# Both keywords are required and each names a variable set to ON or OFF in the
# caller's scope. The answer comes from the library target's compile
# definitions, which is where the build wrote the decision, so it is the same
# answer in a source build, against a prebuilt archive, and after
# find_package(). Reading ARM_NN_ENABLE_F32/F16 out of the cache instead only
# works on the paths that take it as an input.
#
# The target is the first of NS_CMSIS_NN_FLOAT_TARGET (a global property the
# Zephyr module sets, since its library target name is chosen by Zephyr) and
# then cmsis-nn, nsx_cmsis_nn, ns_cmsis_nn_prebuilt. Calling this before any of
# them exists is a FATAL_ERROR rather than a silent OFF/OFF.
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

function(ns_cmsis_nn_float_support)
  cmake_parse_arguments(PARSE_ARGV 0 NSFS "" "F32;F16" "")
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

  get_property(_target GLOBAL PROPERTY NS_CMSIS_NN_FLOAT_TARGET)
  if(NOT _target)
    foreach(_candidate cmsis-nn nsx_cmsis_nn ns_cmsis_nn_prebuilt)
      if(TARGET ${_candidate})
        set(_target ${_candidate})
        break()
      endif()
    endforeach()
  endif()
  if(NOT _target)
    message(FATAL_ERROR
      "ns_cmsis_nn_float_support: no ns-cmsis-nn library target is in scope. "
      "Call it after add_subdirectory(), after the Zephyr module has run, or "
      "after find_package(ns-cmsis-nn).")
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
