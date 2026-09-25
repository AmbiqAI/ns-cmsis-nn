#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check the float16 format contract with GCC: IEEE binary16 only.

Compiles a float16 kernel and a public-header-only consumer with the real
-mfp16-format option. IEEE must build; the Arm alternative format must fail
with the header's #error as the first diagnostic. Then runs the scalar IEEE
half ARG regression under Cortex-M4 QEMU.

Example: python3 check_f16_format_contract.py --cc arm-none-eabi-gcc
No MVE or physical hardware claim. See AmbiqAI/ns-cmsis-nn#511.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

REJECTION = "ARM_NN_ENABLE_F16 does not support the Arm alternative half format"

# Targets where GCC accepts either format: no FP16 arithmetic, or a Cortex-M55
# with its floating-point unit disabled (soft-float ABI, or +nofp, which keeps
# integer MVE). GCC refuses the alternative format when the M55 FPU is enabled.
TARGETS = {
    "cortex-m4-soft": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft"],
    "cortex-m4-hard": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
    "cortex-m55-soft": ["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=soft"],
    "cortex-m55-nofp-mvei": ["-mcpu=cortex-m55+nofp", "-mthumb", "-mfloat-abi=hard"],
}

CONSUMER = r'''#include "arm_nnfunctions.h"
arm_cmsis_nn_status (*const consumer_argmax_f16)(const float16_t *, const cmsis_nn_dims *, int32_t, int32_t *) =
    arm_argmax_f16;
'''


def first_error(output):
    return next((line for line in output.splitlines() if "error:" in line), "")


def check_rejection(args, out):
    consumer = out / "header_only_consumer.c"
    consumer.write_text(CONSUMER)
    units = {"kernel": args.source_root / "Source/BasicMathFunctions/arm_argmax_f16.c", "header-only": consumer}
    records = []
    for target, flags in TARGETS.items():
        for fmt in ("ieee", "alternative"):
            for f32 in (0, 1):
                for unit, source in units.items():
                    command = [args.cc, *flags, f"-mfp16-format={fmt}", "-O2",
                               "-DARM_NN_ENABLE_F16=1", f"-DARM_NN_ENABLE_F32={f32}",
                               "-I" + str(args.source_root / "Include"), "-c", str(source), "-o", "/dev/null"]
                    built = subprocess.run(command, capture_output=True, text=True)
                    output = built.stdout + built.stderr
                    error = first_error(output)
                    if fmt == "ieee":
                        ok = built.returncode == 0
                    else:
                        ok = built.returncode != 0 and "arm_nn_math_types_flt.h" in error and REJECTION in error
                    records.append({"target": target, "format": fmt, "f32": f32, "unit": unit, "command": command,
                                    "rc": built.returncode, "first_error": error, "ok": ok})
                    print(f"{'PASS' if ok else 'FAIL'} {target} {fmt} F32={f32} {unit}"
                          + ("" if ok else f": rc={built.returncode} {error or output.strip()[:200]}"), flush=True)
    return records


def check_ieee_execution(args, out):
    startup = out / "startup.c"
    startup.write_text(r'''#include <stdint.h>
#include <stddef.h>
void *memcpy(void *dest, const void *src, size_t size) {
    unsigned char *d = dest;
    const unsigned char *s = src;
    while (size--) *d++ = *s++;
    return dest;
}
extern int test_formats(void);
static int semihost(int operation, const void *argument) {
    register int r0 __asm__("r0") = operation;
    register const void *r1 __asm__("r1") = argument;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}
void Reset_Handler(void) {
    int result = test_formats();
    semihost(4, result ? "ARG_FORMAT_FAIL\n" : "ARG_FORMAT_PASS\n");
    uint32_t exit_block[2] = {0x20026, (uint32_t)result};
    semihost(0x20, exit_block);
    for (;;) {}
}
__attribute__((section(".vectors"), used))
const uintptr_t vectors[] = {0x20040000, (uintptr_t)Reset_Handler};
''')
    linker = out / "link.ld"
    linker.write_text("ENTRY(Reset_Handler)\nSECTIONS { . = 0; .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) } . = 0x20000000; .data : { *(.data*) } .bss : { *(.bss*) *(COMMON) } }\n")
    records = []
    for f32 in (0, 1):
        elf = out / f"ieee-f32{f32}.elf"
        sources = [args.source_root / f"Source/BasicMathFunctions/arm_arg{op}_{dtype}.c"
                   for op in ("min", "max") for dtype in ("f16", "f32")]
        command = [args.cc, *TARGETS["cortex-m4-soft"], "-mfp16-format=ieee", "-O2", "-ffreestanding", "-nostdlib",
                   "-DARM_NN_ENABLE_F16=1", f"-DARM_NN_ENABLE_F32={f32}",
                   "-I" + str(args.source_root / "Include"), str(startup),
                   str(Path(__file__).with_name("arg_extrema_formats.c")),
                   *map(str, sources), "-T", str(linker), "-o", str(elf)]
        built = subprocess.run(command, capture_output=True, text=True)
        record = {"format": "ieee", "f32": f32, "compile_command": command,
                  "compile_rc": built.returncode, "compile_output": built.stdout + built.stderr}
        if built.returncode == 0:
            run_command = [args.qemu, "-M", "mps2-an386", "-nographic", "-semihosting", "-kernel", str(elf)]
            ran = subprocess.run(run_command, capture_output=True, text=True, timeout=120)
            record.update(run_command=run_command, run_rc=ran.returncode, run_output=ran.stdout + ran.stderr)
        record["ok"] = record.get("run_rc") == 0 and "ARG_FORMAT_PASS" in record.get("run_output", "")
        records.append(record)
        print(f"{'PASS' if record['ok'] else 'FAIL'} ieee F32={f32} ARG execution: "
              + record.get("run_output", record["compile_output"]).strip(), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="arm-none-eabi-gcc")
    parser.add_argument("--qemu", default="qemu-system-arm")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="f16-format-") as temp:
        out = args.output or Path(temp)
        out.mkdir(parents=True, exist_ok=True)
        results = {"rejection": check_rejection(args, out), "ieee_execution": check_ieee_execution(args, out)}
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        failed = [r for group in results.values() for r in group if not r["ok"]]
        if failed:
            raise SystemExit(f"{len(failed)} float16 format contract check(s) failed")
        print("float16 format contract holds")


if __name__ == "__main__":
    main()
