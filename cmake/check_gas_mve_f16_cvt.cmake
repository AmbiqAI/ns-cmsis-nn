# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Refuse to build the float16 kernels with an assembler that mis-encodes the
# Q-register form of VCVTB/VCVTT.F16<->F32: it writes Qn into the field as
# Q2n, so the widen/narrow steps land in the wrong registers and the higher Q
# numbers overflow into an UNDEFINED word that faults at run time. Nothing in
# the preprocessor can see the assembler, so this asks it directly.
# See AmbiqAI/ns-cmsis-nn#427.
#
# Exposes one entry point:
#
#   ns_cmsis_nn_check_gas_mve_f16_cvt()
#       Runs the probe against the calling directory's compiler and flags. It
#       is a no-op unless ARM_NN_ENABLE_F16 is set, the compiler is GNU, and
#       the configured arch flags carry MVE float16. Call it where the
#       float16 sources are selected, so every consumer of
#       cmake/ns_cmsis_nn.cmake is covered and not just the standalone build.

option(ARM_NN_SKIP_GAS_F16_PROBE
       "Downgrade the MVE half<->single assembler check to a warning." OFF)

function(ns_cmsis_nn_check_gas_mve_f16_cvt)
  if(NOT ARM_NN_ENABLE_F16)
    return()
  endif()

  # Clang and armclang encode MVE themselves and never hand this to gas.
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
    return()
  endif()

  # `vcvtb.f16.f32 q1, q2` as a correct assembler lays it down in the object,
  # little-endian. Measured across the installed toolchains, not derived.
  set(_ns_gas_f16_expected "3fee052e")

  # The ethos-u toolchain file the Unity build uses carries -mcpu in directory
  # compile options rather than in CMAKE_C_FLAGS, so the probe has to read both
  # or it silently decides the target has no MVE. Generator expressions have no
  # value at configure time and a single one spans several list elements, so
  # skip them by tracking the bracket depth.
  get_directory_property(_ns_gas_f16_dir_opts COMPILE_OPTIONS)
  set(_ns_gas_f16_opts "")
  set(_ns_gas_f16_depth 0)
  foreach(_ns_gas_f16_opt IN LISTS _ns_gas_f16_dir_opts)
    string(REGEX MATCHALL "\\$<" _ns_gas_f16_open "${_ns_gas_f16_opt}")
    string(REGEX MATCHALL ">" _ns_gas_f16_close "${_ns_gas_f16_opt}")
    list(LENGTH _ns_gas_f16_open _ns_gas_f16_nopen)
    list(LENGTH _ns_gas_f16_close _ns_gas_f16_nclose)
    if(_ns_gas_f16_depth EQUAL 0 AND _ns_gas_f16_nopen EQUAL 0)
      list(APPEND _ns_gas_f16_opts "${_ns_gas_f16_opt}")
    endif()
    math(EXPR _ns_gas_f16_depth
         "${_ns_gas_f16_depth} + ${_ns_gas_f16_nopen} - ${_ns_gas_f16_nclose}")
    if(_ns_gas_f16_depth LESS 0)
      set(_ns_gas_f16_depth 0)
    endif()
  endforeach()

  set(_ns_gas_f16_signature "${CMAKE_C_COMPILER}|${CMAKE_C_FLAGS}|${_ns_gas_f16_opts}")

  # Several targets can attach float16 sources in one configure; report on each
  # distinct compiler and flag combination once rather than once per target.
  get_property(_ns_gas_f16_seen GLOBAL PROPERTY NS_GAS_MVE_F16_CVT_REPORTED)
  if("${_ns_gas_f16_signature}" IN_LIST _ns_gas_f16_seen)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY NS_GAS_MVE_F16_CVT_REPORTED
               "${_ns_gas_f16_signature}")

  if(NOT DEFINED NS_GAS_MVE_F16_CVT_SIGNATURE OR
     NOT NS_GAS_MVE_F16_CVT_SIGNATURE STREQUAL _ns_gas_f16_signature)

    set(_ns_gas_f16_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ns_gas_mve_f16_cvt")
    set(_ns_gas_f16_src "${_ns_gas_f16_dir}/probe.S")
    set(_ns_gas_f16_obj "${_ns_gas_f16_dir}/probe.o")
    file(MAKE_DIRECTORY "${_ns_gas_f16_dir}")
    file(REMOVE "${_ns_gas_f16_obj}")

    # A .S is preprocessed, so the feature macro that gates the kernels also
    # decides whether this target has anything worth checking. The sentinels
    # bracket the instruction so its word can be lifted out of the object
    # without parsing ELF.
    file(WRITE "${_ns_gas_f16_src}"
"#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
    .syntax unified
    .thumb
    .text
    .word 0xdeadbee1
    vcvtb.f16.f32 q1, q2
    .word 0xdeadbee2
#endif
")

    # Through the driver with the project's own C flags, in the order the build
    # uses, so a -B or -mcpu the user passed applies here exactly as it will to
    # the kernels.
    separate_arguments(_ns_gas_f16_flags NATIVE_COMMAND "${CMAKE_C_FLAGS}")
    execute_process(
      COMMAND "${CMAKE_C_COMPILER}" ${_ns_gas_f16_flags} ${_ns_gas_f16_opts}
              -c "${_ns_gas_f16_src}" -o "${_ns_gas_f16_obj}"
      RESULT_VARIABLE _ns_gas_f16_rc
      OUTPUT_VARIABLE _ns_gas_f16_log
      ERROR_VARIABLE  _ns_gas_f16_log)

    if(NOT _ns_gas_f16_rc EQUAL 0)
      set(_ns_gas_f16_word "PROBE-FAILED")
    elseif(EXISTS "${_ns_gas_f16_obj}")
      file(READ "${_ns_gas_f16_obj}" _ns_gas_f16_hex HEX)
      set(_ns_gas_f16_word "")
      if(_ns_gas_f16_hex MATCHES "e1beadde(........)e2beadde")
        set(_ns_gas_f16_word "${CMAKE_MATCH_1}")
      endif()
    else()
      set(_ns_gas_f16_word "PROBE-FAILED")
    endif()

    set(NS_GAS_MVE_F16_CVT_WORD "${_ns_gas_f16_word}" CACHE INTERNAL
        "Encoding this compiler's assembler emits for the MVE Q-form vcvtb.f16.f32.")
    set(NS_GAS_MVE_F16_CVT_LOG "${_ns_gas_f16_log}" CACHE INTERNAL
        "Output of the MVE half<->single assembler probe.")
    # Keyed on what was measured, so a compiler or flag change re-runs it.
    set(NS_GAS_MVE_F16_CVT_SIGNATURE "${_ns_gas_f16_signature}" CACHE INTERNAL
        "Compiler and C flags NS_GAS_MVE_F16_CVT_WORD was measured with.")
  endif()

  if(NS_GAS_MVE_F16_CVT_WORD STREQUAL "")
    # No MVE float16 in the configured arch flags, so no kernel reaches it.
    return()
  endif()

  if(NS_GAS_MVE_F16_CVT_WORD STREQUAL "PROBE-FAILED")
    message(WARNING
      "Could not assemble the MVE half<->single conversion probe with "
      "${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS} ${_ns_gas_f16_opts}; skipping the "
      "assembler check. "
      "See AmbiqAI/ns-cmsis-nn#427.\n${NS_GAS_MVE_F16_CVT_LOG}")
    return()
  endif()

  if(NS_GAS_MVE_F16_CVT_WORD STREQUAL _ns_gas_f16_expected)
    message(STATUS
      "MVE half<->single conversions encode correctly (0x${NS_GAS_MVE_F16_CVT_WORD})")
    return()
  endif()

  # One string, expanded quoted: message() splits an unquoted list on the
  # semicolons this text needs.
  set(_ns_gas_f16_diag "\
The assembler behind ${CMAKE_C_COMPILER} mis-encodes the MVE Q-register form of \
VCVTB/VCVTT.F16<->F32, which the ARM_NN_ENABLE_F16 kernels are built from. \
`vcvtb.f16.f32 q1, q2' assembled to 0x${NS_GAS_MVE_F16_CVT_WORD}, expected \
0x${_ns_gas_f16_expected}: it reads and writes the wrong Q registers, and the \
higher Q numbers become UNDEFINED words that fault at run time.
Fix it either way:
* Build with Arm GNU Toolchain 14.2.Rel1 or newer (binutils 2.43 or newer).
* Keep this compiler and put -B<dir> in its C flags, where <dir> holds the \
unprefixed `as' of a binutils 2.43 or newer arm-none-eabi install (in an Arm \
GNU install that is <root>/arm-none-eabi/bin). Set it through the CFLAGS \
environment variable on a fresh build directory, or through CMAKE_C_FLAGS \
alongside the arch flags -- a bare -DCMAKE_C_FLAGS replaces the ones the \
toolchain file supplies.
ARM_NN_ENABLE_F16=OFF builds without the float16 kernels; integer and float32 \
builds are unaffected. ARM_NN_SKIP_GAS_F16_PROBE=ON downgrades this to a \
warning and builds the broken encoding anyway.
See AmbiqAI/ns-cmsis-nn#427.")

  if(ARM_NN_SKIP_GAS_F16_PROBE)
    message(WARNING "${_ns_gas_f16_diag}")
  else()
    message(FATAL_ERROR "${_ns_gas_f16_diag}")
  endif()
endfunction()
