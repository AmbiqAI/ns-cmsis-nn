# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# ns-cmsis-nn prebuilt-library consumer helper.
#
# Mirrors the NSX `NSX_CMSIS_NN_LIB` prebuilt-mode path: lets a consumer
# point at one of the per-arch `.a` files we publish on each GitHub
# Release and link against it as an IMPORTED target without checking
# out the cmsis-nn source tree.
#
# Usage:
#
#   include(<path>/cmake/ns-cmsis-nn-prebuilt.cmake)
#
#   ns_cmsis_nn_import_prebuilt(
#     LIBRARY      /path/to/libns-cmsis-nn-cortex-m4-7.24.1.a
#     INCLUDE_DIRS /path/to/ns-cmsis-nn/Include
#   )
#
#   target_link_libraries(my_app PRIVATE ns::cmsis-nn)
#
# After the call, the same target names exposed by the source build
# (`ns::cmsis-nn`, `ns-cmsis-nn`, `cmsis-nn`) all resolve to the
# imported archive, so consumer code is build-mode agnostic.

function(ns_cmsis_nn_import_prebuilt)
  cmake_parse_arguments(NS_PB
    ""                       # options
    "LIBRARY"                # one-value
    "INCLUDE_DIRS"           # multi-value
    ${ARGN})

  if(NS_PB_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "ns_cmsis_nn_import_prebuilt: unexpected arguments: "
      "${NS_PB_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT NS_PB_LIBRARY)
    message(FATAL_ERROR
      "ns_cmsis_nn_import_prebuilt(LIBRARY ...) is required.")
  endif()

  if(NOT EXISTS "${NS_PB_LIBRARY}")
    message(FATAL_ERROR
      "ns_cmsis_nn_import_prebuilt: LIBRARY not found: ${NS_PB_LIBRARY}")
  endif()

  if(TARGET cmsis-nn)
    message(FATAL_ERROR
      "ns_cmsis_nn_import_prebuilt: target 'cmsis-nn' already defined. "
      "Don't mix prebuilt and source consumption in the same build.")
  endif()

  add_library(cmsis-nn STATIC IMPORTED GLOBAL)
  set_target_properties(cmsis-nn PROPERTIES
    IMPORTED_LOCATION "${NS_PB_LIBRARY}")

  if(NS_PB_INCLUDE_DIRS)
    foreach(_d IN LISTS NS_PB_INCLUDE_DIRS)
      if(NOT EXISTS "${_d}")
        message(FATAL_ERROR
          "ns_cmsis_nn_import_prebuilt: INCLUDE_DIRS path not found: ${_d}")
      endif()
    endforeach()
    set_target_properties(cmsis-nn PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NS_PB_INCLUDE_DIRS}")
  endif()

  add_library(ns-cmsis-nn  ALIAS cmsis-nn)
  add_library(ns::cmsis-nn ALIAS cmsis-nn)
endfunction()
