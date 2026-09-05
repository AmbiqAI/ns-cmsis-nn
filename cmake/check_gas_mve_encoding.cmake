# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Measure the MVE instruction families whose encoding is not stable across the
# gas releases this library supports, and tell the kernels which form to emit.
# A sweep of the MVE space over binutils 2.39 to 2.45 found exactly two.
#
#   VCVTB/VCVTT.F16<->F32, Q-register form. gas before 2.43 writes Qn into the
#   field as Q2n, so the widen/narrow steps land in the wrong registers and the
#   higher Q numbers overflow into an UNDEFINED word that faults at run time.
#   The float16 kernels use these. See AmbiqAI/ns-cmsis-nn#427.
#
#   VQSHRN/VQSHRUN, the saturating narrowing shifts. gas before 2.43 sets the
#   rounding bit, so every one of them assembles as its VQRSHRN/VQRSHRUN
#   variant: no fault, results off by a half ulp. No kernel uses these yet.
#   See AmbiqAI/ns-cmsis-nn#437.
#
# Nothing in the preprocessor can see the assembler, so this asks it directly.
#
# Exposes one entry point:
#
#   ns_cmsis_nn_check_gas_mve_encoding(<target>)
#       Compiles a witness with the flags <target> will really compile with
#       -- CMAKE_C_FLAGS, the target's own options, definitions and include
#       directories, and the usage requirements of everything it links -- and
#       compares the encoded words. A no-op unless ARM_NN_ENABLE_F16 is set
#       and the compiler is GNU. Call it where the float16 sources are
#       selected, so every consumer of cmake/ns_cmsis_nn.cmake is covered and
#       not just the standalone build.
#
# The verdicts are compile definitions on <target>. ARM_NN_GAS_F16_VERIFIED=1
# on a correct assembler and ARM_NN_GAS_VCVT_F16_BROKEN=1 on a mis-encoding
# one are read by Include/Internal/arm_nn_vcvt_f16.h, which switches to the
# scalar-form conversions. ARM_NN_GAS_VQSHRN_BROKEN=1 has no reader yet; it is
# there so the helper #437 will add can select on it the same way. None of
# them is a refusal -- every combination builds. The shapes this probe cannot
# see -- flags wired after the directory finishes, flags carried only inside a
# generator expression, and consumers that never run CMake at all -- get no
# definitions and fall back to the header's compiler-version guard, which is
# conservative for GCC 13 and blind to a GCC 14 driver over an older binutils.
#
# try_compile() was the obvious mechanism and does not work here. Its
# LINK_LIBRARIES only carries imported targets into the generated project;
# a plain INTERFACE target from the calling project (NSX's
# NSX_BOARD_FLAGS_TARGET, Zephyr's zephyr_interface) is written out as a bare
# library name, so its INTERFACE_COMPILE_OPTIONS never reach the probe -- and
# under CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY there is no link step to
# fail, so the probe would silently measure unflagged code. This walks the
# usage requirements instead.
#
# The witness is a C translation unit whose whole body is a top-level asm, not
# a .S. Same flags either way, but not the same architecture: cc1 opens every
# object it writes with .arch/.fpu/.arch_extension directives derived from
# -mcpu, and those override whatever the driver forwards to gas. A .S has no
# such prologue, so it is judged on the forwarded flags alone -- and a
# consumer that names -mcpu=cortex-m55 together with -mfpu=fpv5-sp-d16
# (helia-core-tester does) gets MVE float16 in cc1, and no MVE at all in gas.
# The preprocessor runs the same in both, so the witness selected the vector
# instruction and then could not assemble it. Going through the C path puts
# the probe on the kernels' own footing. See AmbiqAI/ns-cmsis-nn#427.

option(ARM_NN_SKIP_GAS_F16_PROBE
       "Downgrade an uncompilable MVE encoding witness from an error to a warning, leaving the assembler unmeasured."
       OFF)

# Drop the generator expressions from an option list. They have no value at
# configure time and a single one can span several list elements, so track the
# bracket depth rather than filtering element by element. Sets <out_skipped>
# TRUE when anything was dropped, so the caller can say the view is partial.
function(_ns_cmsis_nn_gas_mve_plain out_var out_skipped)
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

# Drop the options that inject a header into the translation unit. No header
# can change how an instruction encodes, and the directory that holds one is
# routinely carried in a generator expression this probe cannot read, so
# keeping them only gives the witness a way to fail on something the probe is
# not asking about. Both spellings, split and joined.
# See AmbiqAI/ns-cmsis-nn#427.
function(_ns_cmsis_nn_gas_mve_drop_injected out_var out_dropped)
  set(_kept "")
  set(_dropped "")
  set(_pending "")
  foreach(_opt IN LISTS ARGN)
    if(_pending)
      list(APPEND _dropped "${_pending} ${_opt}")
      set(_pending "")
    elseif(_opt MATCHES "^-(include|imacros)$")
      set(_pending "${_opt}")
    elseif(_opt MATCHES "^-(include|imacros).")
      list(APPEND _dropped "${_opt}")
    else()
      list(APPEND _kept "${_opt}")
    endif()
  endforeach()
  if(_pending)
    list(APPEND _dropped "${_pending}")
  endif()
  set(${out_var} "${_kept}" PARENT_SCOPE)
  set(${out_dropped} "${_dropped}" PARENT_SCOPE)
endfunction()

# Compile options, definitions and include directories <target> will carry, in
# the order CMake lays them down: the target's own, then the usage requirements
# of everything it links, transitively. Consumers routinely put -mcpu on an
# INTERFACE target rather than in CMAKE_C_FLAGS, and reading only the latter
# makes the probe decide there is no MVE and skip itself. Include directories
# come along as -I so the witness is compiled with the search path the kernels
# get. Header-injecting options are dropped; <out_dropped>, when the caller
# passes a fourth argument, names what went.
#
# The directory property is not read. CMake seeds a target's COMPILE_OPTIONS
# from it when the target is created, so the target property already carries
# the directory options that apply and only those. This runs at the end of the
# directory, by which time a later add_compile_options() has landed in the
# directory property but reaches no target created before it -- reading it here
# would judge an early target with flags it never compiles with.
function(_ns_cmsis_nn_gas_mve_flags target out_var out_skipped)
  set(_skipped_any FALSE)
  set(_opts "")

  foreach(_prop COMPILE_OPTIONS COMPILE_DEFINITIONS INCLUDE_DIRECTORIES)
    get_target_property(_own ${target} ${_prop})
    if(_own)
      _ns_cmsis_nn_gas_mve_plain(_own_plain _skipped ${_own})
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
      _ns_cmsis_nn_gas_mve_plain(_libs_plain _skipped ${_libs})
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
        _ns_cmsis_nn_gas_mve_plain(_vals_plain _skipped ${_vals})
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
      _ns_cmsis_nn_gas_mve_plain(_next_plain _skipped ${_next})
      if(_skipped)
        set(_skipped_any TRUE)
      endif()
      list(APPEND _pending ${_next_plain})
    endif()
  endwhile()

  _ns_cmsis_nn_gas_mve_drop_injected(_opts _dropped ${_opts})

  set(${out_var} "${_opts}" PARENT_SCOPE)
  set(${out_skipped} "${_skipped_any}" PARENT_SCOPE)
  if(ARGC GREATER 3)
    set(${ARGV3} "${_dropped}" PARENT_SCOPE)
  endif()
endfunction()

# Carry the verdict to Include/Internal/arm_nn_vcvt_f16.h. An INTERFACE library
# has no private scope to put it in.
function(_ns_cmsis_nn_gas_mve_define target definition)
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    target_compile_definitions(${target} INTERFACE ${definition})
  else()
    target_compile_definitions(${target} PRIVATE ${definition})
  endif()
endfunction()

function(_ns_cmsis_nn_gas_mve_run target)
  if(NOT TARGET ${target})
    return()
  endif()

  # The two witness instructions as a correct assembler lays them down in the
  # object, little-endian. Measured across the installed toolchains, not
  # derived. A mis-encoding gas gives 3fee094e and 88ee410f respectively.
  set(_expected "3fee052e")
  set(_expected_vqshrn "88ee400f")

  _ns_cmsis_nn_gas_mve_flags(${target} _opts _genex_skipped _dropped)
  set(_signature "${CMAKE_C_COMPILER}|${CMAKE_C_FLAGS}|${_opts}")

  # Several targets can attach float16 sources in one configure; report on each
  # distinct compiler and flag combination once rather than once per target.
  # Only the message is deduplicated -- every target still has to be measured
  # and marked, or the header check would reject the ones that were skipped.
  set(_report TRUE)
  get_property(_seen GLOBAL PROPERTY NS_GAS_MVE_ENCODING_REPORTED)
  if("${_signature}" IN_LIST _seen)
    set(_report FALSE)
  else()
    set_property(GLOBAL APPEND PROPERTY NS_GAS_MVE_ENCODING_REPORTED "${_signature}")
  endif()

  if(_dropped AND _report)
    list(JOIN _dropped " " _dropped_text)
    message(STATUS
      "MVE encoding check ignored ${_dropped_text} for target "
      "'${target}': a header injected into C cannot change how an instruction "
      "encodes.")
  endif()

  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ns_gas_mve_encoding")
  set(_src "${_dir}/probe.c")
  set(_obj "${_dir}/probe.o")
  file(MAKE_DIRECTORY "${_dir}")
  file(REMOVE "${_obj}")

  # The feature macro that gates the kernels also decides whether this target
  # has anything worth checking, and one compile answers every question with
  # the flags the kernels will really see. The sentinels bracket each
  # instruction so its word can be lifted out of the object without parsing
  # ELF. MVE float16 implies MVE integer, so the narrowing shift rides along on
  # the same gate rather than needing its own.
  file(WRITE "${_src}" [==[
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
__asm__(".syntax unified\n"
        ".thumb\n"
        ".text\n"
        ".word 0xdeadbee1\n"
        "vcvtb.f16.f32 q1, q2\n"
        ".word 0xdeadbee2\n"
        "vqshrnb.s16 q0, q0, #8\n"
        ".word 0xdeadbee3\n");
#else
__asm__(".text\n"
        ".word 0xdeadbee0\n");
#endif
]==])

  # Through the driver with the project's own flags, in the order the build
  # uses, so a -B or -mcpu the user passed applies here exactly as it will to
  # the kernels. -fno-lto goes last because under -flto the object holds IR and
  # no instruction word, which would read here as a target with no MVE.
  separate_arguments(_cflags NATIVE_COMMAND "${CMAKE_C_FLAGS}")
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" ${_cflags} ${_opts} -fno-lto
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
The MVE encoding check could not compile its witness for \
target '${target}' with this target's flags, so the assembler was not \
measured.
Compiler: ${CMAKE_C_COMPILER}
Flags: ${CMAKE_C_FLAGS} ${_opts_text}
Fix the flags so the witness compiles -- most often a flag the probe cannot \
read, such as an include directory carried inside a generator expression that \
an option here depends on -- or set ARM_NN_SKIP_GAS_F16_PROBE=ON to proceed \
unmeasured.
See AmbiqAI/ns-cmsis-nn#427.
${_log}")

    if(NOT ARM_NN_SKIP_GAS_F16_PROBE)
      message(FATAL_ERROR "${_unmeasured}")
    endif()

    if(_report)
      message(WARNING "${_unmeasured}
ARM_NN_SKIP_GAS_F16_PROBE=ON was set, so this is a warning. Neither \
ARM_NN_GAS_VCVT_F16_BROKEN nor ARM_NN_GAS_F16_VERIFIED is defined on target \
'${target}', so Include/Internal/arm_nn_vcvt_f16.h decides on the compiler \
major instead: GCC 13 and older take the scalar-form conversions, GCC 14 and \
newer take the vector form. That is the right answer for every Arm GNU \
release, and the wrong one for a GCC 14 or newer driver paired by hand with a \
binutils below 2.43.")
    endif()
    return()
  endif()

  file(READ "${_obj}" _hex HEX)
  if(NOT _hex MATCHES "e1beadde(........)e2beadde(........)e3beadde")
    # The witness took its #else arm: the flags visible here select no MVE
    # float16. That is a statement about what the probe could see, not a
    # verdict on the build, so nothing is marked as verified unless the user
    # has explicitly taken the check off.
    if(_report)
      set(_why "")
      if(_genex_skipped)
        set(_why " Generator expressions in the options were skipped, so an "
                 "architecture flag carried only inside one is invisible here.")
        string(JOIN "" _why ${_why})
      endif()
      message(STATUS
        "MVE encoding check does not apply to target '${target}': "
        "its flags select no MVE float16 (__ARM_FEATURE_MVE & 2 is 0).${_why} "
        "Flags added to this target after the current directory finishes, or "
        "carried only inside a generator expression, are not visible here "
        "either; for those the compiler-version guard in "
        "Include/Internal/arm_nn_vcvt_f16.h picks the conversion form.")
    endif()
    return()
  endif()
  set(_word "${CMAKE_MATCH_1}")
  set(_word_vqshrn "${CMAKE_MATCH_2}")

  # A -B in the flags moves the assembler out from under the driver, so the
  # verdict belongs to a binary the compiler path does not name. Ask the driver
  # which one it just used, with the same flags the measurement ran under.
  set(_as "")
  if(_report)
    execute_process(
      COMMAND "${CMAKE_C_COMPILER}" ${_cflags} ${_opts} -print-prog-name=as
      OUTPUT_VARIABLE _as
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(NOT _as MATCHES "/")
      # The driver has no `as' of its own to point at, so it will exec whatever
      # PATH resolves at assemble time.
      set(_as "${_as}, resolved from PATH")
    endif()
  endif()

  # Reported separately from the conversions and never gates them: the two
  # families are fixed by the same binutils release, but the library's f16
  # correctness rides only on the first, and nothing here should make a
  # future divergence between them look like one verdict.
  if(NOT _word_vqshrn STREQUAL _expected_vqshrn)
    _ns_cmsis_nn_gas_mve_define(${target} ARM_NN_GAS_VQSHRN_BROKEN=1)
    if(_report)
      # One string, expanded quoted: message() splits an unquoted list on the
      # semicolons this text needs.
      message(STATUS "\
The assembler behind ${CMAKE_C_COMPILER} (${_as}) also mis-encodes the MVE \
saturating narrowing shifts: `vqshrnb.s16 q0, q0, #8' assembled to \
0x${_word_vqshrn}, \
expected 0x${_expected_vqshrn}, which is the rounding variant VQRSHRNB. \
VQSHRUN is affected the same way. No kernel emits either family today, so \
this is a record, not a defect in this build; target '${target}' is built \
with ARM_NN_GAS_VQSHRN_BROKEN=1 so that a helper can select on it.
See AmbiqAI/ns-cmsis-nn#437.")
    endif()
  endif()

  if(_word STREQUAL _expected)
    _ns_cmsis_nn_gas_mve_define(${target} ARM_NN_GAS_F16_VERIFIED=1)
    if(_report)
      message(STATUS
        "MVE half<->single conversions encode correctly (0x${_word})")
    endif()
    return()
  endif()

  # Not a failure: the kernels have a second way to emit these conversions, so
  # the measurement selects it rather than stopping the build.
  _ns_cmsis_nn_gas_mve_define(${target} ARM_NN_GAS_VCVT_F16_BROKEN=1)
  if(_report)
    # One string, expanded quoted: message() splits an unquoted list on the
    # semicolons this text needs.
    message(STATUS "\
The assembler behind ${CMAKE_C_COMPILER} (${_as}) mis-encodes the MVE \
Q-register form of VCVTB/VCVTT.F16<->F32: `vcvtb.f16.f32 q1, q2' assembled to \
0x${_word}, \
expected 0x${_expected}. Target '${target}' is built with \
ARM_NN_GAS_VCVT_F16_BROKEN=1, so the float16 kernels emit the scalar-form \
helper in Include/Internal/arm_nn_vcvt_f16.h for the half<->single \
conversions. Same results, four instructions per conversion instead of one.
To get the vector form back, assemble with binutils 2.43 or newer: build with \
Arm GNU Toolchain 14.2.Rel1 or newer, or keep this compiler and put -B<dir>/ \
in its C flags, where <dir> holds the unprefixed `as' of a binutils 2.43 or \
newer arm-none-eabi install (in an Arm GNU install that is \
<root>/arm-none-eabi/bin/; keep the trailing slash). Set it through the CFLAGS \
environment variable on a fresh build directory, or through CMAKE_C_FLAGS \
alongside the arch flags -- a bare -DCMAKE_C_FLAGS replaces the ones the \
toolchain file supplies.
See AmbiqAI/ns-cmsis-nn#427.")
  endif()
endfunction()

function(ns_cmsis_nn_check_gas_mve_encoding target)
  if(NOT ARM_NN_ENABLE_F16)
    return()
  endif()

  # Clang and armclang encode MVE themselves and never hand this to gas.
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
    return()
  endif()

  get_property(_scheduled GLOBAL PROPERTY NS_GAS_MVE_ENCODING_SCHEDULED)
  if("${target}" IN_LIST _scheduled)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY NS_GAS_MVE_ENCODING_SCHEDULED "${target}")

  # Consumers link the target that carries -mcpu after attaching the sources
  # (nsx/CMakeLists.txt does), so reading the link closure now would see an
  # empty one. Defer to the end of the directory, when the wiring is complete.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    # DEFER re-expands its arguments in the directory scope when the call
    # runs, where this function's `target' no longer exists; EVAL bakes the
    # name in now as a bracket argument that survives that second pass.
    cmake_language(EVAL CODE
      "cmake_language(DEFER CALL _ns_cmsis_nn_gas_mve_run [[${target}]])")
  else()
    _ns_cmsis_nn_gas_mve_run("${target}")
  endif()
endfunction()
