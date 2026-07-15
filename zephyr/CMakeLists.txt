# Copyright (c) 2025 Ambiq
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_NS_CMSIS_NN)

  if(CONFIG_NS_CMSIS_NN_USE_PREBUILT)

    # ----------------------------------------------------------------------
    # Prebuilt mode: import a pre-compiled libns-cmsis-nn.a from a per-arch
    # SDK tarball already extracted at CONFIG_NS_CMSIS_NN_PREBUILT_PATH.
    # The application is responsible for ensuring the tarball's target_cpu
    # matches the board's Cortex-M variant — the .a's ABI must agree with
    # what Zephyr passes to the linker.
    # ----------------------------------------------------------------------

    set(_ns_sdk "${CONFIG_NS_CMSIS_NN_PREBUILT_PATH}")
    if(NOT _ns_sdk)
      message(FATAL_ERROR
        "CONFIG_NS_CMSIS_NN_USE_PREBUILT=y but "
        "CONFIG_NS_CMSIS_NN_PREBUILT_PATH is empty.")
    endif()
    if(NOT EXISTS "${_ns_sdk}/lib/libns-cmsis-nn.a")
      message(FATAL_ERROR
        "Prebuilt ns-cmsis-nn SDK at '${_ns_sdk}' is missing "
        "lib/libns-cmsis-nn.a — check CONFIG_NS_CMSIS_NN_PREBUILT_PATH.")
    endif()
    if(NOT IS_DIRECTORY "${_ns_sdk}/include")
      message(FATAL_ERROR
        "Prebuilt ns-cmsis-nn SDK at '${_ns_sdk}' is missing include/.")
    endif()

    if(NOT TARGET ns_cmsis_nn_prebuilt)
      add_library(ns_cmsis_nn_prebuilt STATIC IMPORTED GLOBAL)
      set_target_properties(ns_cmsis_nn_prebuilt PROPERTIES
        IMPORTED_LOCATION             "${_ns_sdk}/lib/libns-cmsis-nn.a"
        INTERFACE_INCLUDE_DIRECTORIES "${_ns_sdk}/include")
    endif()

    # Headers must be globally visible — application code includes
    # arm_nnfunctions.h directly, same as in source mode.
    zephyr_include_directories("${_ns_sdk}/include")
    zephyr_link_libraries(ns_cmsis_nn_prebuilt)

    # Inline-asm requantize is baked into the prebuilt's binary; the
    # define is irrelevant at consumer-side, so we deliberately do not
    # forward CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY here.

    # Unlike the requantize flag above, ARM_NN_ENABLE_F32/F16 gate the float
    # *declarations* in the public headers (Include/Internal/arm_nn_config.h),
    # not just internal implementation choices baked into the archive. They
    # must therefore be forwarded to consumer translation units even in
    # prebuilt mode, or the float API surface stays invisible and any direct
    # use of the float headers fails to compile.
    #
    # When the SDK tarball at CONFIG_NS_CMSIS_NN_PREBUILT_PATH ships a
    # manifest.json (schema_version >= 2 with a "features" block), the
    # requested CONFIG_NS_CMSIS_NN_ENABLE_F32/F16 toggles are validated
    # against it and a FATAL_ERROR is raised for any request the archive
    # doesn't actually implement — this turns a predictable
    # undefined-reference link error into a clear configure-time
    # diagnostic. Older manifests without a "features" block (or no
    # manifest.json at all) are treated conservatively as integer-only,
    # per this project's documented compatibility fallback.
    set(_ns_manifest "${_ns_sdk}/manifest.json")
    set(_ns_has_f32 FALSE)
    set(_ns_has_f16 FALSE)
    if(EXISTS "${_ns_manifest}")
      file(READ "${_ns_manifest}" _ns_manifest_json)
      string(JSON _ns_schema_version ERROR_VARIABLE _ns_json_err
             GET "${_ns_manifest_json}" schema_version)
      string(JSON _ns_features ERROR_VARIABLE _ns_json_err
             GET "${_ns_manifest_json}" features)
      if(_ns_features STREQUAL "NOTFOUND" OR NOT _ns_schema_version OR _ns_schema_version LESS 2)
        message(STATUS
          "ns-cmsis-nn: ${_ns_manifest} has no schema_version>=2 'features' "
          "block; treating prebuilt archive as integer-only.")
      else()
        string(JSON _ns_has_f32 GET "${_ns_manifest_json}" features f32)
        string(JSON _ns_has_f16 GET "${_ns_manifest_json}" features f16)
        message(STATUS
          "ns-cmsis-nn: validated prebuilt archive at '${_ns_sdk}' against "
          "${_ns_manifest} (f32=${_ns_has_f32} f16=${_ns_has_f16})")
      endif()
    else()
      message(STATUS
        "ns-cmsis-nn: no manifest.json found at '${_ns_sdk}'; "
        "CONFIG_NS_CMSIS_NN_ENABLE_F32/F16 requests are UNVERIFIED against "
        "archive contents and treated conservatively as integer-only for "
        "validation purposes (the flags are still forwarded as requested).")
    endif()

    if(CONFIG_NS_CMSIS_NN_ENABLE_F32 AND EXISTS "${_ns_manifest}" AND NOT _ns_has_f32)
      message(FATAL_ERROR
        "CONFIG_NS_CMSIS_NN_ENABLE_F32=y but the prebuilt SDK's manifest "
        "(${_ns_manifest}) does not report f32 support. Use an SDK tarball "
        "built with F32 enabled, or disable CONFIG_NS_CMSIS_NN_ENABLE_F32.")
    endif()
    if(CONFIG_NS_CMSIS_NN_ENABLE_F16 AND EXISTS "${_ns_manifest}" AND NOT _ns_has_f16)
      message(FATAL_ERROR
        "CONFIG_NS_CMSIS_NN_ENABLE_F16=y but the prebuilt SDK's manifest "
        "(${_ns_manifest}) does not report f16 support. Use an SDK tarball "
        "built with F16 enabled, or disable CONFIG_NS_CMSIS_NN_ENABLE_F16.")
    endif()

    set(_ns_f32 0)
    set(_ns_f16 0)
    if(CONFIG_NS_CMSIS_NN_ENABLE_F32)
      set(_ns_f32 1)
    endif()
    if(CONFIG_NS_CMSIS_NN_ENABLE_F16)
      set(_ns_f16 1)
    endif()
    zephyr_compile_definitions(
      ARM_NN_ENABLE_F32=${_ns_f32}
      ARM_NN_ENABLE_F16=${_ns_f16}
    )

  else()

  # Single source of truth for the per-group source lists, shared with the
  # standalone CMake build and the NSX module.
  include(${ZEPHYR_CURRENT_MODULE_DIR}/cmake/ns_cmsis_nn.cmake)

  zephyr_library()
  zephyr_library_compile_options($<TARGET_PROPERTY:compiler,optimization_fast>)

  zephyr_include_directories(${ZEPHYR_CURRENT_MODULE_DIR}/Include)
  zephyr_library_include_directories(${ZEPHYR_CMSIS_MODULE_DIR}/CMSIS/Core/Include)

  # Bridge Zephyr Kconfig float toggles into the SSoT float gates.
  # These variables are consumed by ns_cmsis_nn_group_sources() when it resolves
  # per-group source lists.
  set(ARM_NN_ENABLE_F32 0)
  set(ARM_NN_ENABLE_F16 0)
  if(CONFIG_NS_CMSIS_NN_ENABLE_F32)
    set(ARM_NN_ENABLE_F32 1)
  endif()
  if(CONFIG_NS_CMSIS_NN_ENABLE_F16)
    set(ARM_NN_ENABLE_F16 1)
  endif()

  # Keep float feature guards visible in all translation units that include
  # CMSIS-NN public/internal headers.
  zephyr_compile_definitions(
    ARM_NN_ENABLE_F32=${ARM_NN_ENABLE_F32}
    ARM_NN_ENABLE_F16=${ARM_NN_ENABLE_F16}
  )

  # Translate the existing CONFIG_NS_CMSIS_NN_* knobs into SSoT group names.
  # Kconfig symbol names are aligned with SSoT group ids (#151); deprecated
  # spellings (BASICMATH / STRIDED_SLICE / SVD) `select` into the new ones.
  set(_ns_groups)
  foreach(_pair
            "ACTIVATION;activation"
            "BASICMATHSNN;basicmath"
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
            "STRIDEDSLICE;stridedslice"
            "SVDF;svd"
            "TRANSPOSE;transpose"
            "TILE;tile"
            "BROADCAST;broadcast"
            "SCATTER;scatter"
            "SELECT;select"
            "REVERSESEQUENCE;reversesequence"
            "DYNAMICUPDATESLICE;dynamicupdateslice")
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

  endif()  # CONFIG_NS_CMSIS_NN_USE_PREBUILT

endif()
