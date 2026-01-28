#ifndef CMSIS_H
#define CMSIS_H

// CMSIS header for PMU/DWT functionality
// This file provides the necessary CMSIS definitions for the test runners

#include "cmsis_compiler.h"
#include "cmsis_gcc.h"

// Core Debug definitions
#ifndef CoreDebug
    #define CoreDebug ((CoreDebug_Type *)CoreDebug_BASE)
#endif

// DWT definitions
#ifndef DWT
    #define DWT ((DWT_Type *)DWT_BASE)
#endif

// DWT Control Register definitions
#ifndef DWT_CTRL_CYCCNTENA_Msk
    #define DWT_CTRL_CYCCNTENA_Msk (1UL << DWT_CTRL_CYCCNTENA_Pos)
#endif

#ifndef DWT_CTRL_CYCCNTENA_Pos
    #define DWT_CTRL_CYCCNTENA_Pos 0U
#endif

// Core Debug DEMCR definitions
#ifndef CoreDebug_DEMCR_TRCENA_Msk
    #define CoreDebug_DEMCR_TRCENA_Msk (1UL << CoreDebug_DEMCR_TRCENA_Pos)
#endif

#ifndef CoreDebug_DEMCR_TRCENA_Pos
    #define CoreDebug_DEMCR_TRCENA_Pos 24U
#endif

// DWT LAR (Lock Access Register) - only define if not already defined
#ifndef DWT_LAR
    #define DWT_LAR 0xC5ACCE55U
#endif

// DWT Type structure is already defined by AmbiqSuite
// No need to redefine it here

// Memory barriers
#ifndef __DSB
    #define __DSB() __asm volatile("dsb" ::: "memory")
#endif

#ifndef __ISB
    #define __ISB() __asm volatile("isb" ::: "memory")
#endif

#endif // CMSIS_H
