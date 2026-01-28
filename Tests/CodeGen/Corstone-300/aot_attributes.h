#pragma once

#include <stdio.h>

// ── Logging hooks ─────────────────────────────────────────────────────────────

#ifndef E2E_PRINTF
    #define E2E_PRINTF(...) printf(__VA_ARGS__)
#endif

// ── Memory hooks ──────────────────────────────────────────────────────────────

/* -------- DTCM (RW) -------- */
#define E2E_PUT_IN_DTCM __attribute__((section(".dtcm_bss")))
#define E2E_PUT_IN_DTCM_INIT __attribute__((section(".dtcm_data")))

/* -------- SRAM (RW) -------- */
#define E2E_PUT_IN_SRAM __attribute__((section(".sram_bss")))
#define E2E_PUT_IN_SRAM_INIT __attribute__((section(".sram_data")))

/* -------- MRAM/DDR (RO) -------- */
#define E2E_PUT_IN_MRAM_INIT __attribute__((section(".ddr_rodata")))

// ── Instruction hooks ─────────────────────────────────────────────────────────

#ifndef E2E_PUT_IN_ITCM
    #define E2E_PUT_IN_ITCM __attribute__((section(".itcm_text")))
#endif
