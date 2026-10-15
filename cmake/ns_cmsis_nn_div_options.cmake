# Exact DIV semantics require the compiler to retain NaNs, infinities and signed zeros.
# Keep other kernels' optimization flags unchanged.
function(ns_cmsis_nn_div_options target)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    foreach(src IN LISTS ARGN)
      if(src MATCHES "/arm_elementwise_div_f(16|32)\\.c$")
        if(CMAKE_VERSION VERSION_LESS 3.18)
          message(FATAL_ERROR "Clang DIV source options require CMake 3.18 or newer")
        endif()
        set_property(SOURCE "${src}" TARGET_DIRECTORY "${target}" APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math)
      endif()
    endforeach()
  endif()
endfunction()
