# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

# Internal entry-point helpers. Callers own options/cache entries, platform
# wiring and source attachment order; these functions do not change that state.
# Compiler policy is also the caller's: CMP0123 must be NEW before project()
# enables armclang, not changed here after compiler initialization. See AmbiqAI/ns-cmsis-nn#444.
include_guard(GLOBAL)

function(_ns_cmsis_nn_float_values f32 f16 out_f32 out_f16)
  foreach(width f32 f16)
    if(${width})
      set(${out_${width}} 1 PARENT_SCOPE)
    else()
      set(${out_${width}} 0 PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

# Translate existing option names, preserving each entry point's group order.
# NSX already accepts group IDs and does not need this translation.
function(_ns_cmsis_nn_groups_from_options out_var entrypoint)
  ns_cmsis_nn_groups(groups)
  if(entrypoint STREQUAL "ZEPHYR")
    set(prefix CONFIG_NS_CMSIS_NN_)
    set(select_option SELECT)
    list(REMOVE_ITEM groups nnsupport)
    list(FIND groups lstm support_index)
    math(EXPR support_index "${support_index} + 1")
    list(INSERT groups ${support_index} nnsupport)
  elseif(entrypoint STREQUAL "STANDALONE")
    set(prefix "")
    set(select_option SELECTOPS)
  else()
    message(FATAL_ERROR "Unknown CMSIS-NN configuration entry point: ${entrypoint}")
  endif()

  set(enabled)
  foreach(group IN LISTS groups)
    string(TOUPPER "${group}" option_name)
    if(group STREQUAL "basicmath")
      set(option_name BASICMATHSNN)
    elseif(group STREQUAL "svd")
      set(option_name SVDF)
    elseif(group STREQUAL "select")
      set(option_name ${select_option})
    endif()
    if(${prefix}${option_name})
      list(APPEND enabled ${group})
    endif()
  endforeach()
  set(${out_var} "${enabled}" PARENT_SCOPE)
endfunction()

# Apply caller-resolved source-build settings without creating cache defaults.
# Zephyr owns optimization and global definitions through its platform API;
# it uses the normalization above and keeps that application in its adapter.
function(_ns_cmsis_nn_source_options target f32 f16 optimization requantize)
  _ns_cmsis_nn_float_values("${f32}" "${f16}" value_f32 value_f16)
  target_compile_options(${target} PRIVATE ${optimization})
  target_compile_definitions(${target} PUBLIC
    ARM_NN_ENABLE_F32=${value_f32} ARM_NN_ENABLE_F16=${value_f16})
  if(requantize)
    target_compile_definitions(${target} PUBLIC CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
  endif()
endfunction()
