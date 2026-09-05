#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Single source of truth for the float feature switches.
#
# Three entry points take the float request under their own option spelling
# and hand it to ns_cmsis_nn_publish_float_switches():
#
#   top-level CMakeLists.txt   ARM_NN_ENABLE_F32 / ARM_NN_ENABLE_F16
#   nsx/CMakeLists.txt         NSX_CMSIS_NN_ENABLE_F32 / _F16
#   zephyr/CMakeLists.txt      CONFIG_NS_CMSIS_NN_ENABLE_F32 / _F16
#
# find_package() against a published SDK tarball
# (cmake/templates/ns-cmsis-nn-config.cmake.in) is the fourth entry point and
# does not call this function: it has no request switch of its own and exposes
# the archive's compiled-in capabilities on the imported target only, so it
# publishes no cache entries.
#
#   ns_cmsis_nn_publish_float_switches(F32 <value> F16 <value>
#                                      REQUEST_PREFIX <prefix>
#                                      [REQUEST_DEFAULT <ON|OFF>]
#                                      [REQUESTED_BY <text>]
#                                      [AUTHORITATIVE [AUTHORITY_NOTE <text>]])
#
# REQUEST_PREFIX is the caller's own spelling with F32/F16 stripped off, and
# REQUEST_DEFAULT is the literal default that switch declares; the caller must
# share one variable between its option()/Kconfig default and this argument so
# the declaration and the comparison cannot drift.
#
# On the NSX and Zephyr paths the function writes each effective value to two
# places:
#
#   - the CMake cache, as BOOL 0/1, for consumers that add this repository as
#     a subdirectory or a module and read the switches before any target of
#     ours is in scope;
#   - the calling directory scope, as 0/1, for the caller's own
#     target_compile_definitions() and for ns_cmsis_nn_group_sources().
#
# On the top-level path the request already arrives under the published name,
# so the cache entry option() created is left exactly as the user supplied it
# (ON stays ON) and only the directory-scope 0/1 is set.
#
# The library target's compile definitions remain the compile-time truth; the
# cache entry is a configure-time mirror of the same decision, so the two
# cannot drift.
#
# Resolution when both spellings are in play on the NSX path:
#
#   - the caller's own switch at its declared default plus ARM_NN_ENABLE_*
#     set: the ARM_NN_ENABLE_* value is the request, reported on a STATUS
#     line. Setting ARM_NN_ENABLE_* directly is as valid a way to ask as the
#     path's own spelling;
#   - both set away from their defaults and agreeing: honored as asked;
#   - the caller's own switch away from its default and ARM_NN_ENABLE_* set to
#     a different value: FATAL_ERROR. Two explicit requests disagree and only
#     the caller knows which one it meant.
#
# AUTHORITATIVE says the caller's own switch is the only way to ask on that
# path, which is what the Zephyr path passes: the request comes from Kconfig,
# whose arch and toolchain dependencies a CMake variable must not be able to
# route around. ARM_NN_ENABLE_* is then never adopted; a value that agrees with
# the caller's switch is accepted silently, and one that disagrees is a
# FATAL_ERROR pointing at the caller's own switch as the authority.
# AUTHORITY_NOTE is the caller's explanation of why its switch cannot be
# overridden, quoted into that message.
#
# For each width the function also leaves NS_CMSIS_NN_FLOAT_REQUEST_<width> and
# NS_CMSIS_NN_FLOAT_DROP_HINT_<width> in the calling scope: the name of the
# variable the published value actually came from, and how to drop it. The
# prebuilt manifest checks quote both, so a rejection names the spelling that
# was really in play rather than the one the caller happens to read.
#
# _NS_CMSIS_NN_PUBLISHED_ARM_NN_ENABLE_F32/F16 are INTERNAL cache entries this
# function keeps for itself: each holds what the entry point's own switch
# resolved to on the last successful configure, so a value left in the cache by
# an earlier configure of ours reads as stale rather than as a fresh request
# and the entry point's own switch can be flipped in place. A consumer value
# that happens to equal the previous configure's result is indistinguishable
# from that stale value and is treated as ours.
#
# See AmbiqAI/ns-cmsis-nn#420.
#

if(DEFINED NS_CMSIS_NN_FLOAT_SWITCHES_INCLUDED)
  return()
endif()
set(NS_CMSIS_NN_FLOAT_SWITCHES_INCLUDED TRUE)

# How to drop <var> again, worded for the form it actually arrived in: -U only
# reaches cache entries, and CONFIG_* comes from Kconfig rather than the cache.
function(_ns_cmsis_nn_float_drop_hint _var _out)
  if(DEFINED CACHE{${_var}})
    set(${_out}
      "re-run cmake with -U${_var} (or delete CMakeCache.txt in the build directory); if the project that adds ns-cmsis-nn re-creates the entry itself, with set(${_var} ... CACHE ...) or option(${_var} ...) above add_subdirectory(), -U alone cannot reach it and dropping the call alone leaves the seeded entry in the cache: remove that call and then re-run with -U${_var} (or delete CMakeCache.txt)"
      PARENT_SCOPE)
  elseif(_var MATCHES "^CONFIG_")
    set(${_out}
      "drop it from the Kconfig fragment that sets it (prj.conf, a board overlay, or the command line)"
      PARENT_SCOPE)
  else()
    set(${_out}
      "remove the set(${_var} ...) from the scope that adds ns-cmsis-nn"
      PARENT_SCOPE)
  endif()
endfunction()

function(ns_cmsis_nn_publish_float_switches)
  cmake_parse_arguments(NSF "AUTHORITATIVE"
    "F32;F16;REQUEST_PREFIX;REQUEST_DEFAULT;REQUESTED_BY;AUTHORITY_NOTE"
    "" ${ARGN})
  if(NSF_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "ns_cmsis_nn_publish_float_switches: unexpected arguments: "
      "${NSF_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT NSF_REQUEST_PREFIX)
    message(FATAL_ERROR
      "ns_cmsis_nn_publish_float_switches: REQUEST_PREFIX is required; pass "
      "the entry point's own switch name with F32/F16 stripped off, for "
      "example NSX_CMSIS_NN_ENABLE_.")
  endif()
  if(NOT NSF_REQUESTED_BY)
    set(NSF_REQUESTED_BY "${NSF_REQUEST_PREFIX}F32/F16")
  endif()
  if(NOT NSF_AUTHORITY_NOTE)
    set(NSF_AUTHORITY_NOTE
      "That switch is the only supported way to ask on this path.")
  endif()
  if(NSF_REQUEST_DEFAULT)
    set(_default 1)
  else()
    set(_default 0)
  endif()

  foreach(_width F32 F16)
    set(_name "ARM_NN_ENABLE_${_width}")
    set(_own "${NSF_REQUEST_PREFIX}${_width}")
    set(_sentinel "_NS_CMSIS_NN_PUBLISHED_${_name}")
    string(REGEX REPLACE "^F" "float" _pretty "${_width}")

    if(NSF_${_width})
      set(_own_want 1)
    else()
      set(_own_want 0)
    endif()
    set(_want ${_own_want})
    set(_source "${_own}")

    if(NOT _own STREQUAL _name)
      # A plain variable in the calling scope is as much a request as a cache
      # entry, so read through the normal variable lookup, which prefers the
      # plain one when both exist.
      set(_have_request FALSE)
      if(DEFINED ${_name})
        set(_raw "${${_name}}")
        if(_raw)
          set(_request 1)
        else()
          set(_request 0)
        endif()
        set(_have_request TRUE)
        if(DEFINED CACHE{${_sentinel}}
           AND "$CACHE{${_sentinel}}" STREQUAL "${_request}")
          set(_have_request FALSE)
        endif()
      endif()

      if(_have_request AND NSF_AUTHORITATIVE)
        # The caller's switch carries dependencies CMake cannot see, so an
        # ARM_NN_ENABLE_* value is never adopted here: it either agrees and is
        # redundant, or it is trying to route around them.
        if(NOT "${_request}" EQUAL "${_own_want}")
          set(_own_raw "${${_own}}")
          if(_own_raw STREQUAL "")
            set(_own_state "unset (n)")
          else()
            set(_own_state "${_own_raw}")
          endif()
          string(REGEX REPLACE "^CONFIG_" "" _own_symbol "${_own}")
          _ns_cmsis_nn_float_drop_hint("${_name}" _drop_published)
          message(FATAL_ERROR
            "ns-cmsis-nn: ${_name} cannot change the ${_pretty} setting on "
            "this path.\n"
            "  ${_own} is the authority here and is currently ${_own_state}, "
            "which asks for ${_name}=${_own_want}; ${_name}=${_raw} asks for "
            "${_name}=${_request}.\n"
            "  ${NSF_AUTHORITY_NOTE}\n"
            "  Ask through ${_own_symbol} instead, and drop the CMake "
            "request: ${_drop_published}.")
        endif()
      elseif(_have_request)
        if("${_own_want}" EQUAL "${_default}")
          set(_want ${_request})
          set(_source "${_name}")
          message(STATUS
            "ns-cmsis-nn: ${_name}=${_request} requested directly and ${_own} "
            "is at its declared default; publishing ${_name}=${_request}.")
        elseif(NOT "${_request}" EQUAL "${_own_want}")
          set(_own_raw "${${_own}}")
          if(_own_raw STREQUAL "")
            set(_own_raw "${_own_want}")
          endif()
          _ns_cmsis_nn_float_drop_hint("${_name}" _drop_published)
          _ns_cmsis_nn_float_drop_hint("${_own}" _drop_own)
          message(FATAL_ERROR
            "ns-cmsis-nn: contradictory ${_pretty} switches.\n"
            "  ${_own}=${_own_raw} (${NSF_REQUESTED_BY}) asks for "
            "${_name}=${_own_want}, and ${_name}=${_raw} asks for "
            "${_name}=${_request}. Both are explicit requests, so this build "
            "will not pick a winner.\n"
            "  To go with ${_own}=${_own_raw}: ${_drop_published}.\n"
            "  To go with ${_name}=${_raw}: ${_drop_own}.")
        endif()
      endif()
    endif()

    # What the published value came from, and how to drop it, for the prebuilt
    # manifest checks in the entry points.
    _ns_cmsis_nn_float_drop_hint("${_source}" _source_hint)
    set(NS_CMSIS_NN_FLOAT_REQUEST_${_width} "${_source}" PARENT_SCOPE)
    set(NS_CMSIS_NN_FLOAT_DROP_HINT_${_width} "${_source_hint}" PARENT_SCOPE)

    if(_own STREQUAL _name)
      # The cache entry is the user's own input on this path; rewriting it
      # would turn their ON into a 0/1 for no gain.
      set(${_name} ${_want} PARENT_SCOPE)
    else()
      set(_doc "Enable the CMSIS-NN ${_pretty} extensions.")
      if(DEFINED CACHE{${_name}})
        get_property(_prev_doc CACHE "${_name}" PROPERTY HELPSTRING)
        # A bare -D leaves CMake's placeholder helpstring behind; keep this
        # module's own description rather than copying it forward.
        if(_prev_doc AND NOT _prev_doc MATCHES "No help, variable specified on the command line")
          set(_doc "${_prev_doc}")
        endif()
      endif()

      set(${_name} ${_want} CACHE BOOL "${_doc}" FORCE)
      set(${_sentinel} ${_own_want} CACHE INTERNAL
        "What ${_own} resolved to when ns-cmsis-nn last published ${_name}.")
      set(${_name} ${_want} PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
