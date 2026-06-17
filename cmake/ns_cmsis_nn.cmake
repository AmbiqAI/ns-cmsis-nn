#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Single source of truth (SSoT) for the ns-cmsis-nn source layout.
#
# This module is consumed by:
#   - the standalone CMake build (top-level CMakeLists.txt)
#   - the Zephyr module (zephyr/CMakeLists.txt)
#   - the NSX module    (nsx/CMakeLists.txt)
#
# It exposes three public entry points:
#
#   ns_cmsis_nn_groups(<out_var>)
#       Returns the list of all known operator group ids.
#
#   ns_cmsis_nn_group_sources(<group> <out_var>)
#       Returns the absolute paths of all source files belonging to <group>,
#       using the same selection rules as the legacy upstream build.
#
#   ns_cmsis_nn_attach(<target>
#                      [GROUPS  <g...> | ALL]
#                      [DTYPES  <d...> | ALL]
#                      [INCLUDE_DIRS_VISIBILITY  PUBLIC | PRIVATE | INTERFACE])
#       Adds the resolved source set and the public Include/ directory to
#       <target>. <target> must already exist (created by the consumer).
#
# Notes on selection:
#   - Each group has an explicit subdirectory under Source/, an explicit list
#     of glob patterns, and an optional list of explicitly-named extras.
#     The combined set is byte-for-byte equivalent to the legacy
#     Source/<Subdir>/CMakeLists.txt rules.
#   - When DTYPES is restricted, files whose basename carries one of the
#     recognized dtype tags (_s4 / _s8 / _s16 / _s32 / _s64 / _q7 / _q15)
#     are kept only if their tag is in <d...>. Files with no recognized
#     dtype tag (e.g. arm_nntables.c) are always kept. DTYPES=ALL (the
#     default) is a no-op.
#   - Upstream's Source/<Subdir>/CMakeLists.txt files are intentionally left
#     in place but unused by SSoT consumers, to minimize friction when
#     syncing with ARM-software/CMSIS-NN.
#

if(DEFINED NS_CMSIS_NN_CMAKE_INCLUDED)
  return()
endif()
set(NS_CMSIS_NN_CMAKE_INCLUDED TRUE)

# Absolute path to the repository root (parent of this cmake/ directory).
get_filename_component(NS_CMSIS_NN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Canonical, ordered list of operator group ids.
set(_NS_CMSIS_NN_GROUPS
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
    nnsupport
)

# All known dtype tags; used by the DTYPES filter.
set(_NS_CMSIS_NN_DTYPES s4 s8 s16 s32 s64 q7 q15 f16 f32)

# ----------------------------------------------------------------------------
# Internal: per-group definition.
#
# _ns_cmsis_nn_group_def(<group> <out_subdir> <out_patterns> <out_extras>)
# ----------------------------------------------------------------------------
function(_ns_cmsis_nn_group_def group out_subdir out_patterns out_extras)
  set(subdir "")
  set(patterns "")
  set(extras "")

  if(group STREQUAL "activation")
    set(subdir   "ActivationFunctions")
    set(patterns "*_s8*.c" "*_s16*.c")
    set(extras   "arm_relu_q7.c" "arm_relu6_q7.c" "arm_relu_q15.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_nn_activation_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_nn_activation_f16.c")
    endif()
  elseif(group STREQUAL "basicmath")
    set(subdir   "BasicMathFunctions")
    set(patterns "*_s8*.c" "*_s16*.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_elementwise_add_f32.c"
                         "arm_elementwise_mul_f32.c"
                         "arm_minimum_f32.c"
                         "arm_maximum_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_elementwise_add_f16.c"
                         "arm_elementwise_mul_f16.c"
                         "arm_minimum_f16.c"
                         "arm_maximum_f16.c")
    endif()
  elseif(group STREQUAL "comparison")
    set(subdir   "ComparisonFunctions")
    set(patterns "*.c")
  elseif(group STREQUAL "concatenation")
    set(subdir   "ConcatenationFunctions")
    set(patterns "*_s8*.c" "*_s16*.c" "*_s32*.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_concatenation_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_concatenation_f16.c")
    endif()
  elseif(group STREQUAL "convolution")
    set(subdir   "ConvolutionFunctions")
    set(patterns "*_s4*.c" "*_s8*.c" "*_s16*.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_convolve_f32.c"
                         "arm_convolve_1_x_n_f32.c"
                         "arm_convolve_1x1_f32.c"
                         "arm_depthwise_conv_f32.c"
                         "arm_transpose_conv_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_convolve_f16.c"
                         "arm_convolve_1_x_n_f16.c"
                         "arm_convolve_1x1_f16.c"
                         "arm_depthwise_conv_f16.c"
                         "arm_transpose_conv_f16.c")
    endif()
  elseif(group STREQUAL "fullyconnected")
    set(subdir   "FullyConnectedFunctions")
    set(patterns "*_s4.c" "*_s8.c" "*_s16*.c" "*_s64.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_batch_matmul_f32.c"
                         "arm_fully_connected_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_batch_matmul_f16.c"
                         "arm_fully_connected_f16.c")
    endif()
  elseif(group STREQUAL "gather")
    set(subdir   "GatherFunctions")
    set(patterns "*_*.c")
  elseif(group STREQUAL "lstm")
    set(subdir   "LSTMFunctions")
    set(patterns "*_s8.c" "*_s16.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_lstm_unidirectional_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_lstm_unidirectional_f16.c")
    endif()
  elseif(group STREQUAL "nnsupport")
    set(subdir   "NNSupportFunctions")
    set(patterns "*_s4*.c" "*_s8*.c" "*_s16*.c" "*_s32*.c")
    set(extras   "arm_nntables.c"
                 "arm_q7_to_q15_with_offset.c"
                 "arm_s8_to_s16_unordered_with_offset.c")
    if(ARM_NN_ENABLE_F32 OR ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_nntables_flt.c")
    endif()
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_batch_norm_f32.c"
                         "arm_get_buffer_size_f32.c"
                         "arm_nn_conv1d_k3_f32.c"
                         "arm_nn_conv1d_k3_packed_f32.c"
                         "arm_nn_conv1d_k5_f32.c"
                         "arm_nn_conv1d_k5_packed_f32.c"
                         "arm_nn_depthwise_conv1d_k3_f32.c"
                         "arm_nn_depthwise_conv3x3_f32.c"
                         "arm_nn_depthwise_conv_nt_t_f32.c"
                         "arm_nn_maxpool1d_f32.c"
                         "arm_nn_pack_conv_patch_f32.c"
                         "arm_nn_lstm_step_f32.c"
                         "arm_nn_mat_mult_nt_t_f32.c"
                         "arm_nn_mat_mult_nt_n_packed_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_batch_norm_f16.c"
                         "arm_get_buffer_size_f16.c"
                         "arm_nn_conv1d_k3_f16.c"
                         "arm_nn_conv1d_k3_packed_f16.c"
                         "arm_nn_conv1d_k5_f16.c"
                         "arm_nn_conv1d_k5_packed_f16.c"
                         "arm_nn_depthwise_conv1d_k3_f16.c"
                         "arm_nn_depthwise_conv2x5_f16.c"
                         "arm_nn_depthwise_conv3x3_f16.c"
                         "arm_nn_depthwise_conv_nt_t_f16.c"
                         "arm_nn_pack_conv_patch_f16.c"
                         "arm_nn_lstm_step_f16.c"
                         "arm_nn_maxpool1d_f16.c"
                         "arm_nn_mat_mult_nt_t_f16.c"
                         "arm_nn_mat_mult_nt_n_packed_f16.c")
    endif()
  elseif(group STREQUAL "pad")
    set(subdir   "PadFunctions")
    set(patterns "*_s8.c" "*_s16.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_pad_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_pad_f16.c")
    endif()
  elseif(group STREQUAL "pooling")
    set(subdir   "PoolingFunctions")
    set(patterns "*_s8.c" "*_s16.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_avg_pool_f32.c"
                         "arm_max_pool_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_avg_pool_f16.c"
                         "arm_max_pool_f16.c")
    endif()
  elseif(group STREQUAL "quantization")
    set(subdir   "QuantizationFunctions")
    set(patterns "*_*.c")
  elseif(group STREQUAL "reshape")
    set(subdir   "ReshapeFunctions")
    set(patterns "arm_reshape_s8.c"
                 "arm_batch_to_space_nd_s8.c"
                 "arm_batch_to_space_nd_s16.c"
                 "arm_depth_to_space_s8.c"
                 "arm_depth_to_space_s16.c"
                 "arm_resize_nearest_neighbor_s8.c"
                 "arm_resize_nearest_neighbor_s16.c"
                 "arm_space_to_batch_nd_s8.c"
                 "arm_space_to_batch_nd_s16.c"
                 "arm_space_to_depth_s8.c"
                 "arm_space_to_depth_s16.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_reshape_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_reshape_f16.c")
    endif()
  elseif(group STREQUAL "softmax")
    set(subdir   "SoftmaxFunctions")
    set(patterns "*_s8.c")
    set(extras   "arm_softmax_s8_s16.c"
                 "arm_softmax_s16.c"
                 "arm_nn_softmax_common_s8.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_softmax_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f16.c")
    endif()
  elseif(group STREQUAL "stridedslice")
    set(subdir   "StridedSliceFunctions")
    set(patterns "*_*.c")
  elseif(group STREQUAL "svd")
    set(subdir   "SVDFunctions")
    set(patterns "*_s8.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_svdf_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_svdf_f16.c")
    endif()
  elseif(group STREQUAL "transpose")
    set(subdir   "TransposeFunctions")
    set(patterns "*_s8.c" "*_s16.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_transpose_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_transpose_f16.c")
    endif()
  else()
    message(FATAL_ERROR "ns_cmsis_nn: unknown group '${group}'. Valid groups: ${_NS_CMSIS_NN_GROUPS}")
  endif()

  set(${out_subdir}   "${subdir}"   PARENT_SCOPE)
  set(${out_patterns} "${patterns}" PARENT_SCOPE)
  set(${out_extras}   "${extras}"   PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Internal: filter a list of source paths by allowed dtype tags.
#
# _ns_cmsis_nn_filter_dtypes(<allowed_dtypes> <out_var> <files...>)
#
# A file is kept when either:
#   - its basename does not contain any recognized dtype tag, or
#   - its basename contains a dtype tag that is in <allowed_dtypes>.
# ----------------------------------------------------------------------------
function(_ns_cmsis_nn_filter_dtypes allowed_dtypes out_var)
  set(kept "")
  foreach(file IN LISTS ARGN)
    get_filename_component(base "${file}" NAME)
    set(tag "")
    foreach(dt IN LISTS _NS_CMSIS_NN_DTYPES)
      if(base MATCHES "_${dt}([._]|$)")
        set(tag "${dt}")
        break()
      endif()
    endforeach()
    if(tag STREQUAL "")
      list(APPEND kept "${file}")
    else()
      list(FIND allowed_dtypes "${tag}" _idx)
      if(NOT _idx EQUAL -1)
        list(APPEND kept "${file}")
      endif()
    endif()
  endforeach()
  set(${out_var} "${kept}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Public: list of all known operator group ids.
# ----------------------------------------------------------------------------
function(ns_cmsis_nn_groups out_var)
  set(${out_var} "${_NS_CMSIS_NN_GROUPS}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Public: absolute source paths for <group> (no dtype filtering).
# ----------------------------------------------------------------------------
function(ns_cmsis_nn_group_sources group out_var)
  _ns_cmsis_nn_group_def("${group}" subdir patterns extras)

  set(dir "${NS_CMSIS_NN_ROOT}/Source/${subdir}")
  set(files "")

  foreach(pat IN LISTS patterns)
    file(GLOB matches "${dir}/${pat}")
    list(APPEND files ${matches})
  endforeach()

  foreach(extra IN LISTS extras)
    list(APPEND files "${dir}/${extra}")
  endforeach()

  if(files)
    list(REMOVE_DUPLICATES files)
    list(SORT files)
  endif()

  set(${out_var} "${files}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Public: attach the resolved source set and Include/ to <target>.
# ----------------------------------------------------------------------------
function(ns_cmsis_nn_attach target)
  set(options "")
  set(one_value INCLUDE_DIRS_VISIBILITY)
  set(multi_value GROUPS DTYPES)
  cmake_parse_arguments(NSA "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NSA_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "ns_cmsis_nn_attach: unrecognized argument(s): ${NSA_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT TARGET ${target})
    message(FATAL_ERROR "ns_cmsis_nn_attach: target '${target}' does not exist")
  endif()

  # Resolve GROUPS (default = ALL).
  if(NOT NSA_GROUPS)
    set(NSA_GROUPS ALL)
  endif()
  list(LENGTH NSA_GROUPS _ng)
  if(_ng EQUAL 1 AND NSA_GROUPS STREQUAL "ALL")
    set(groups "${_NS_CMSIS_NN_GROUPS}")
  else()
    set(groups "${NSA_GROUPS}")
    foreach(g IN LISTS groups)
      list(FIND _NS_CMSIS_NN_GROUPS "${g}" _idx)
      if(_idx EQUAL -1)
        message(FATAL_ERROR
          "ns_cmsis_nn_attach: unknown group '${g}'. Valid groups: ${_NS_CMSIS_NN_GROUPS}")
      endif()
    endforeach()
  endif()

  # Resolve DTYPES (default = ALL).
  if(NOT NSA_DTYPES)
    set(NSA_DTYPES ALL)
  endif()
  list(LENGTH NSA_DTYPES _nd)
  if(_nd EQUAL 1 AND NSA_DTYPES STREQUAL "ALL")
    set(dtypes "${_NS_CMSIS_NN_DTYPES}")
    set(dtype_filter_active FALSE)
  else()
    set(dtypes "${NSA_DTYPES}")
    set(dtype_filter_active TRUE)
    foreach(d IN LISTS dtypes)
      list(FIND _NS_CMSIS_NN_DTYPES "${d}" _idx)
      if(_idx EQUAL -1)
        message(FATAL_ERROR
          "ns_cmsis_nn_attach: unknown dtype '${d}'. Valid dtypes: ${_NS_CMSIS_NN_DTYPES}")
      endif()
    endforeach()
  endif()

  # Resolve include-directory visibility (default = PUBLIC).
  if(NOT NSA_INCLUDE_DIRS_VISIBILITY)
    set(NSA_INCLUDE_DIRS_VISIBILITY PUBLIC)
  endif()
  if(NOT NSA_INCLUDE_DIRS_VISIBILITY MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
    message(FATAL_ERROR
      "ns_cmsis_nn_attach: INCLUDE_DIRS_VISIBILITY must be PUBLIC, PRIVATE, or INTERFACE (got '${NSA_INCLUDE_DIRS_VISIBILITY}')")
  endif()

  # Collect sources.
  set(all_sources "")
  foreach(g IN LISTS groups)
    ns_cmsis_nn_group_sources("${g}" _src)
    list(APPEND all_sources ${_src})
  endforeach()
  if(all_sources)
    list(REMOVE_DUPLICATES all_sources)
  endif()

  if(dtype_filter_active)
    _ns_cmsis_nn_filter_dtypes("${dtypes}" all_sources ${all_sources})
  endif()

  if(NOT all_sources)
    message(WARNING "ns_cmsis_nn_attach(${target}): no sources matched (groups=${groups}, dtypes=${dtypes})")
  endif()

  target_sources(${target} PRIVATE ${all_sources})
  # Wrap in $<BUILD_INTERFACE:...> so consumers that re-export <target> via
  # install(TARGETS ... EXPORT ...) do not leak this absolute build-tree path
  # into their installed export file. Consumers that need an install-tree
  # include path attach their own $<INSTALL_INTERFACE:...> separately.
  target_include_directories(${target} ${NSA_INCLUDE_DIRS_VISIBILITY}
    "$<BUILD_INTERFACE:${NS_CMSIS_NN_ROOT}/Include>")
endfunction()
