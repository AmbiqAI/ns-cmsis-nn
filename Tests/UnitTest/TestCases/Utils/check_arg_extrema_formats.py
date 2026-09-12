#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Run scalar IEEE/alternative half ARG regressions with GCC and Cortex-M4 QEMU.

Example: python3 check_arg_extrema_formats.py --cc arm-none-eabi-gcc
No MVE or physical hardware claim. See AmbiqAI/ns-cmsis-nn#503.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="arm-none-eabi-gcc")
    parser.add_argument("--qemu", default="qemu-system-arm")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="arg-formats-") as temp:
        out = args.output or Path(temp)
        out.mkdir(parents=True, exist_ok=True)
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
        for fmt, f32 in [("alternative", 0), ("alternative", 1), ("ieee", 1)]:
            elf = out / f"{fmt}-f32{f32}.elf"
            sources = [args.source_root / f"Source/BasicMathFunctions/arm_arg{op}_{dtype}.c"
                       for op in ("min", "max") for dtype in ("f16", "f32")]
            command = [args.cc, "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft",
                       f"-mfp16-format={fmt}", "-O2", "-ffreestanding", "-nostdlib",
                       "-DARM_NN_ENABLE_F16=1", f"-DARM_NN_ENABLE_F32={f32}",
                       "-I" + str(args.source_root / "Include"), str(startup),
                       str(Path(__file__).with_name("arg_extrema_formats.c")),
                       *map(str, sources), "-T", str(linker), "-o", str(elf)]
            built = subprocess.run(command, capture_output=True, text=True)
            record = {"format": fmt, "f32": f32, "compile_command": command,
                      "compile_rc": built.returncode, "compile_output": built.stdout + built.stderr}
            if built.returncode == 0:
                run_command = [args.qemu, "-M", "mps2-an386", "-nographic", "-semihosting",
                               "-kernel", str(elf)]
                ran = subprocess.run(run_command, capture_output=True, text=True, timeout=60)
                record.update(run_command=run_command, run_rc=ran.returncode,
                              run_output=ran.stdout + ran.stderr)
            records.append(record)
            (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(fmt, "F32=" + str(f32), record.get("run_output", record["compile_output"]), flush=True)
        assert all(r["compile_rc"] == 0 and r.get("run_rc") == 0 and
                   "ARG_FORMAT_PASS" in r.get("run_output", "") for r in records), records


if __name__ == "__main__":
    main()
