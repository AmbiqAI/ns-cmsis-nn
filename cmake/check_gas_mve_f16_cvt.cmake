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
#   ns_cmsis_nn_check_gas_mve_f16_cvt(<target>)
#       Assembles a witness instruction with the flags <target> will really
#       compile with -- CMAKE_C_FLAGS, the target's own options, definitions
#       and include directories, and the usage requirements of everything it
#       links -- and compares the encoded word. A no-op unless
#       ARM_NN_ENABLE_F16 is set and the compiler is GNU. Call it where the
#       float16 sources are selected, so every consumer of
#       cmake/ns_cmsis_nn.cmake is covered and not just the standalone build.
#
# On a good assembler the probe defines ARM_NN_GAS_F16_VERIFIED=1 on <target>.
# Include/arm_nnsupportfunctions_flt.h refuses a GCC 13 or older float16 MVE
# build without that definition, so the shapes this probe cannot see -- flags
# wired after the directory finishes, flags carried only inside a generator
# expression, and consumers that never run CMake at all -- fail closed at
# compile time instead of silently building the mis-encoded conversions.
#
# try_compile() was the obvious mechanism and does not work here. Its
# LINK_LIBRARIES only carries imported targets into the generated project;
# a plain INTERFACE target from the calling project (NSX's
# NSX_BOARD_FLAGS_TARGET, Zephyr's zephyr_interface) is written out as a bare
# library name, so its INTERFACE_COMPILE_OPTIONS never reach the probe -- and
# under CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY there is no link step to
# fail, so the probe would silently measure unflagged code. This walks the
# usage requirements instead.

option(ARM_NN_SKIP_GAS_F16_PROBE
       "Downgrade the MVE half<->single assembler check to a warning and lift the matching compile-time guard."
       OFF)

# Drop the generator expressions from an option list. They have no value at
# configure time and a single one can span several list elements, so track the
# bracket depth rather than filtering element by element. Sets <out_skipped>
# TRUE when anything was dropped, so the caller can say the view is partial.
function(_ns_cmsis_nn_gas_f16_plain out_var out_skipped)
  set(_kept "")
  set(_depth 0)
  set(_skipped FALSE)
  foreach(_opt IN LISTS ARGN)
    string(REGEX MATCHALL "\\$<" _open "${_opt}")
    string(REGEX MATCHALL ">" _close "${_opt}")
    list(LENGTH _open _nopen)
    list(LENGTH _close _nclose)
    if(_depth EQUAL 0 AND _nopen EQUAL 0)
      list(APPEND _kept "${_opt}")
    else()
      set(_skipped TRUE)
    endif()
    math(EXPR _depth "${_depth} + ${_nopen} - ${_nclose}")
    if(_depth LESS 0)
      set(_depth 0)
    endif()
  endforeach()
  set(${out_var} "${_kept}" PARENT_SCOPE)
  set(${out_skipped} "${_skipped}" PARENT_SCOPE)
endfunction()

# Compile options, definitions and include directories <target> will carry, in
# the order CMake lays them down: the target's own, then the usage requirements
# of everything it links, transitively. Consumers routinely put -mcpu on an
# INTERFACE target rather than in CMAKE_C_FLAGS, and reading only the latter
# makes the probe decide there is no MVE and skip itself. Include directories
# come along as -I because an option can depend on them: a board target that
# carries `-include board_cfg.h` puts the header's directory in
# INTERFACE_INCLUDE_DIRECTORIES, and without it the witness does not assemble
# at all.
#
# The directory property is not read. CMake seeds a target's COMPILE_OPTIONS
# from it when the target is created, so the target property already carries
# the directory options that apply and only those. This runs at the end of the
# directory, by which time a later add_compile_options() has landed in the
# directory property but reaches no target created before it -- reading it here
# would judge an early target with flags it never compiles with.
function(_ns_cmsis_nn_gas_f16_flags target out_var out_skipped)
  set(_skipped_any FALSE)
  set(_opts "")

  foreach(_prop COMPILE_OPTIONS COMPILE_DEFINITIONS INCLUDE_DIRECTORIES)
    get_target_property(_own ${target} ${_prop})
    if(_own)
      _ns_cmsis_nn_gas_f16_plain(_own_plain _skipped ${_own})
      if(_skipped)
        set(_skipped_any TRUE)
      endif()
      foreach(_item IN LISTS _own_plain)
        if(_prop STREQUAL "COMPILE_DEFINITIONS")
          list(APPEND _opts "-D${_item}")
        elseif(_prop STREQUAL "INCLUDE_DIRECTORIES")
          list(APPEND _opts "-I${_item}")
        else()
          list(APPEND _opts "${_item}")
        endif()
      endforeach()
    endif()
  endforeach()

  # Breadth-first over the link closure. $<LINK_ONLY:...> entries fall out with
  # every other generator expression, which is what we want: they contribute no
  # usage requirements to a compile.
  set(_pending "")
  foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(_libs ${target} ${_prop})
    if(_libs)
      _ns_cmsis_nn_gas_f16_plain(_libs_plain _skipped ${_libs})
      if(_skipped)
        set(_skipped_any TRUE)
      endif()
      list(APPEND _pending ${_libs_plain})
    endif()
  endforeach()

  set(_seen "")
  while(_pending)
    list(POP_FRONT _pending _dep)
    if(NOT TARGET ${_dep})
      continue()
    endif()
    # An alias and its target are the same usage requirements twice.
    get_target_property(_aliased ${_dep} ALIASED_TARGET)
    if(_aliased)
      set(_dep "${_aliased}")
    endif()
    if("${_dep}" IN_LIST _seen)
      continue()
    endif()
    list(APPEND _seen "${_dep}")

    foreach(_prop INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS
                  INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(_vals ${_dep} ${_prop})
      if(_vals)
        _ns_cmsis_nn_gas_f16_plain(_vals_plain _skipped ${_vals})
        if(_skipped)
          set(_skipped_any TRUE)
        endif()
        foreach(_item IN LISTS _vals_plain)
          if(_prop STREQUAL "INTERFACE_COMPILE_DEFINITIONS")
            list(APPEND _opts "-D${_item}")
          elseif(_prop STREQUAL "INTERFACE_INCLUDE_DIRECTORIES")
            list(APPEND _opts "-I${_item}")
          else()
            list(APPEND _opts "${_item}")
          endif()
        endforeach()
      endif()
    endforeach()

    get_target_property(_next ${_dep} INTERFACE_LINK_LIBRARIES)
    if(_next)
      _ns_cmsis_nn_gas_f16_plain(_next_plain _skipped ${_next})
      if(_skipped)
        set(_skipped_any TRUE)
      endif()
      list(APPEND _pending ${_next_plain})
    endif()
  endwhile()

  set(${out_var} "${_opts}" PARENT_SCOPE)
  set(${out_skipped} "${_skipped_any}" PARENT_SCOPE)
endfunction()

# Assert to the header check that this target's assembler was measured. An
# INTERFACE library has no private scope to put it in.
function(_ns_cmsis_nn_gas_f16_mark target)
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    target_compile_definitions(${target} INTERFACE ARM_NN_GAS_F16_VERIFIED=1)
  else()
    target_compile_definitions(${target} PRIVATE ARM_NN_GAS_F16_VERIFIED=1)
  endif()
endfunction()

function(_ns_cmsis_nn_gas_f16_run target)
  if(NOT TARGET ${target})
    return()
  endif()

  # `vcvtb.f16.f32 q1, q2` as a correct assembler lays it down in the object,
  # little-endian. Measured across the installed toolchains, not derived.
  set(_expected "3fee052e")

  _ns_cmsis_nn_gas_f16_flags(${target} _opts _genex_skipped)
  set(_signature "${CMAKE_C_COMPILER}|${CMAKE_C_FLAGS}|${_opts}")

  # Several targets can attach float16 sources in one configure; report on each
  # distinct compiler and flag combination once rather than once per target.
  # Only the message is deduplicated -- every target still has to be measured
  # and marked, or the header check would reject the ones that were skipped.
  set(_report TRUE)
  get_property(_seen GLOBAL PROPERTY NS_GAS_MVE_F16_CVT_REPORTED)
  if("${_signature}" IN_LIST _seen)
    set(_report FALSE)
  else()
    set_property(GLOBAL APPEND PROPERTY NS_GAS_MVE_F16_CVT_REPORTED "${_signature}")
  endif()

  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ns_gas_mve_f16_cvt")
  set(_src "${_dir}/probe.S")
  set(_obj "${_dir}/probe.o")
  file(MAKE_DIRECTORY "${_dir}")
  file(REMOVE "${_obj}")

  # A .S is preprocessed, so the feature macro that gates the kernels also
  # decides whether this target has anything worth checking, and one assemble
  # answers both questions with the flags the kernels will really see. The
  # sentinels bracket the instruction so its word can be lifted out of the
  # object without parsing ELF.
  file(WRITE "${_src}"
"#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
    .syntax unified
    .thumb
    .text
    .word 0xdeadbee1
    vcvtb.f16.f32 q1, q2
    .word 0xdeadbee2
#else
    .text
    .word 0xdeadbee0
#endif
")

  # Through the driver with the project's own flags, in the order the build
  # uses, so a -B or -mcpu the user passed applies here exactly as it will to
  # the kernels.
  separate_arguments(_cflags NATIVE_COMMAND "${CMAKE_C_FLAGS}")
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" ${_cflags} ${_opts}
            -c "${_src}" -o "${_obj}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _log
    ERROR_VARIABLE  _log)

  if(NOT _rc EQUAL 0 OR NOT EXISTS "${_obj}")
    # The flags are a list; joined so the reader sees the command line rather
    # than semicolon-separated elements.
    list(JOIN _opts " " _opts_text)
    # One string, expanded quoted: message() splits an unquoted list on the
    # semicolons this text needs.
    set(_unmeasured "\
The MVE half<->single assembler check could not assemble its witness for \
target '${target}' with this target's flags, so the assembler was not \
measured.
Compiler: ${CMAKE_C_COMPILER}
Flags: ${CMAKE_C_FLAGS} ${_opts_text}
Fix the flags so the witness assembles -- most often an include directory, or \
an `-include' header the probe cannot see, such as one carried inside a \
generator expression -- or set ARM_NN_SKIP_GAS_F16_PROBE=ON to proceed \
unmeasured.
See AmbiqAI/ns-cmsis-nn#427.
${_log}")

    if(NOT ARM_NN_SKIP_GAS_F16_PROBE)
      message(FATAL_ERROR "${_unmeasured}")
    endif()

    # The option is the documented last resort, so it has to work from here
    # too: a build whose flags the probe cannot even assemble is exactly the
    # one that would otherwise stop at the header guard with no way past it.
    _ns_cmsis_nn_gas_f16_mark(${target})
    if(_report)
      message(WARNING "${_unmeasured}
ARM_NN_SKIP_GAS_F16_PROBE=ON was set, so this is a warning and \
ARM_NN_GAS_F16_VERIFIED=1 is defined on target '${target}', which lifts the \
compile-time guard in arm_nnsupportfunctions_flt.h as well. Nothing measured \
this assembler; if it is one of the mis-encoding ones, the float16 kernels in \
this build are wrong.")
    endif()
    return()
  endif()

  file(READ "${_obj}" _hex HEX)
  if(NOT _hex MATCHES "e1beadde(........)e2beadde")
    # The witness took its #else arm: the flags visible here select no MVE
    # float16. That is a statement about what the probe could see, not a
    # verdict on the build, so nothing is marked as verified unless the user
    # has explicitly taken the check off.
    if(ARM_NN_SKIP_GAS_F16_PROBE)
      _ns_cmsis_nn_gas_f16_mark(${target})
    endif()
    if(_report)
      set(_why "")
      if(_genex_skipped)
        set(_why " Generator expressions in the options were skipped, so an "
                 "architecture flag carried only inside one is invisible here.")
        string(JOIN "" _why ${_why})
      endif()
      message(STATUS
        "MVE half<->single assembler check does not apply to target '${target}': "
        "its flags select no MVE float16 (__ARM_FEATURE_MVE & 2 is 0).${_why} "
        "Flags added to this target after the current directory finishes, or "
        "carried only inside a generator expression, are not visible here "
        "either; for those the compile-time guard in "
        "arm_nnsupportfunctions_flt.h stands in, and a float16 MVE build on "
        "GCC 13 or older will refuse to compile. ARM_NN_SKIP_GAS_F16_PROBE=ON "
        "lifts that guard too.")
    endif()
    return()
  endif()
  set(_word "${CMAKE_MATCH_1}")

  if(_word STREQUAL _expected)
    _ns_cmsis_nn_gas_f16_mark(${target})
    if(_report)
      message(STATUS
        "MVE half<->single conversions encode correctly (0x${_word})")
    endif()
    return()
  endif()

  # One string, expanded quoted: message() splits an unquoted list on the
  # semicolons this text needs.
  set(_diag "\
The assembler behind ${CMAKE_C_COMPILER} mis-encodes the MVE Q-register form of \
VCVTB/VCVTT.F16<->F32, which the ARM_NN_ENABLE_F16 kernels are built from. \
`vcvtb.f16.f32 q1, q2' assembled to 0x${_word}, expected \
0x${_expected}: it reads and writes the wrong Q registers, and the \
higher Q numbers become UNDEFINED words that fault at run time.
Fix it either way:
* Build with Arm GNU Toolchain 14.2.Rel1 or newer (binutils 2.43 or newer).
* Keep this compiler and put -B<dir>/ in its C flags, where <dir> holds the \
unprefixed `as' of a binutils 2.43 or newer arm-none-eabi install (in an Arm \
GNU install that is <root>/arm-none-eabi/bin/; keep the trailing slash). Set \
it through the CFLAGS environment variable on a fresh build directory, or \
through CMAKE_C_FLAGS alongside the arch flags -- a bare -DCMAKE_C_FLAGS \
replaces the ones the toolchain file supplies.
ARM_NN_ENABLE_F16=OFF builds without the float16 kernels; integer and float32 \
builds are unaffected. ARM_NN_SKIP_GAS_F16_PROBE=ON downgrades this to a \
warning and builds the broken encoding anyway.
See AmbiqAI/ns-cmsis-nn#427.")

  if(ARM_NN_SKIP_GAS_F16_PROBE)
    # The option would be dead without this: the compile-time guard in
    # arm_nnsupportfunctions_flt.h would stop the build the probe was told to
    # let through.
    _ns_cmsis_nn_gas_f16_mark(${target})
    if(_report)
      message(WARNING "${_diag}\nARM_NN_SKIP_GAS_F16_PROBE=ON was set, so this "
                      "is a warning and ARM_NN_GAS_F16_VERIFIED=1 is defined on "
                      "target '${target}' to let the compile-time guard through "
                      "as well. The float16 kernels in this build are "
                      "mis-encoded by choice.")
    endif()
  else()
    message(FATAL_ERROR "${_diag}")
  endif()
endfunction()

function(ns_cmsis_nn_check_gas_mve_f16_cvt target)
  if(NOT ARM_NN_ENABLE_F16)
    return()
  endif()

  # Clang and armclang encode MVE themselves and never hand this to gas.
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
    return()
  endif()

  get_property(_scheduled GLOBAL PROPERTY NS_GAS_MVE_F16_CVT_SCHEDULED)
  if("${target}" IN_LIST _scheduled)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY NS_GAS_MVE_F16_CVT_SCHEDULED "${target}")

  # Consumers link the target that carries -mcpu after attaching the sources
  # (nsx/CMakeLists.txt does), so reading the link closure now would see an
  # empty one. Defer to the end of the directory, when the wiring is complete.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    # DEFER re-expands its arguments in the directory scope when the call
    # runs, where this function's `target' no longer exists; EVAL bakes the
    # name in now as a bracket argument that survives that second pass.
    cmake_language(EVAL CODE
      "cmake_language(DEFER CALL _ns_cmsis_nn_gas_f16_run [[${target}]])")
  else()
    _ns_cmsis_nn_gas_f16_run("${target}")
  endif()
endfunction()
