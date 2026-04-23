# Copyright (c) 2025 Ambiq
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_NS_CMSIS_NN)
  set(NS_CMSIS_NN_DIR ${ZEPHYR_CURRENT_MODULE_DIR})

  include(${NS_CMSIS_NN_DIR}/cmake/ns-cmsis-nn-select-variant.cmake)

  ns_cmsis_nn_resolve_variant(
    PACKAGE_ROOT "${NS_CMSIS_NN_DIR}"
    OUT_VAR NS_CMSIS_NN_LIBRARY
    TOOLCHAIN_OUT NS_CMSIS_NN_RESOLVED_TOOLCHAIN
    TARGET_CPU_OUT NS_CMSIS_NN_RESOLVED_TARGET_CPU
    ARCH_OUT NS_CMSIS_NN_RESOLVED_ARCH
  )

  zephyr_library()
  zephyr_include_directories(${NS_CMSIS_NN_DIR}/include)

  if(NOT TARGET ns_cmsis_nn_prebuilt)
    add_library(ns_cmsis_nn_prebuilt STATIC IMPORTED GLOBAL)
    set_target_properties(ns_cmsis_nn_prebuilt PROPERTIES
      IMPORTED_LOCATION "${NS_CMSIS_NN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NS_CMSIS_NN_DIR}/include"
    )
  endif()

  target_link_libraries(zephyr_interface INTERFACE ns_cmsis_nn_prebuilt)
  zephyr_link_libraries(ns_cmsis_nn_prebuilt)

  message(STATUS
    "ns-cmsis-nn: linking ${NS_CMSIS_NN_RESOLVED_ARCH} binary for ${NS_CMSIS_NN_RESOLVED_TOOLCHAIN} "
    "(${NS_CMSIS_NN_RESOLVED_TARGET_CPU})"
  )
endif()
