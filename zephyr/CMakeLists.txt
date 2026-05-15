# Copyright (c) 2025 Ambiq
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_NS_CMSIS_NN)

  # Single source of truth for the per-group source lists, shared with the
  # standalone CMake build and the NSX module.
  include(${ZEPHYR_CURRENT_MODULE_DIR}/cmake/ns_cmsis_nn.cmake)

  zephyr_library()
  zephyr_library_compile_options($<TARGET_PROPERTY:compiler,optimization_fast>)

  zephyr_include_directories(${ZEPHYR_CURRENT_MODULE_DIR}/Include)
  zephyr_library_include_directories(${ZEPHYR_CMSIS_MODULE_DIR}/CMSIS/Core/Include)

  # Translate the existing CONFIG_NS_CMSIS_NN_* knobs into SSoT group names.
  # (Kconfig renames, if any, will be done in #151 — keep the symbols here.)
  set(_ns_groups)
  foreach(_pair
            "ACTIVATION;activation"
            "BASICMATH;basicmath"
            "COMPARISON;comparison"
            "CONCATENATION;concatenation"
            "CONVOLUTION;convolution"
            "FULLYCONNECTED;fullyconnected"
            "GATHER;gather"
            "LSTM;lstm"
            "NNSUPPORT;nnsupport"
            "PAD;pad"
            "POOLING;pooling"
            "QUANTIZATION;quantization"
            "RESHAPE;reshape"
            "SOFTMAX;softmax"
            "STRIDED_SLICE;stridedslice"
            "SVD;svd"
            "TRANSPOSE;transpose")
    list(GET _pair 0 _kconfig)
    list(GET _pair 1 _group)
    if(CONFIG_NS_CMSIS_NN_${_kconfig})
      list(APPEND _ns_groups ${_group})
    endif()
  endforeach()

  if(_ns_groups)
    ns_cmsis_nn_attach(${ZEPHYR_CURRENT_LIBRARY}
                       GROUPS ${_ns_groups}
                       INCLUDE_DIRS_VISIBILITY PUBLIC)
  endif()

  # Inline-assembly requantize must be visible at every translation unit that
  # includes arm_nnsupportfunctions.h (it gates inline functions in the public
  # header), so propagate it globally rather than only to this library.
  if(CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY)
    zephyr_compile_definitions(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
  endif()

endif()
