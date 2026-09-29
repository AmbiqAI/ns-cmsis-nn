#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check the float16 format contract with GCC: IEEE binary16 only.

Compiles a float16 kernel and a public-header-only consumer with the real
-mfp16-format option. IEEE must build; the Arm alternative format must fail
with the header's #error as the first diagnostic. Armv8.1-M without MVE must
fail the same way on GCC before 15.3; where it builds, and on the MVE and
integer-MVE controls, the object must contain no Advanced SIMD loads or stores.
Then runs the scalar IEEE half ARG regression under Cortex-M4 QEMU.

Example: python3 check_f16_format_contract.py --cc arm-none-eabi-gcc
Compile, disassembly and QEMU evidence only; no hardware claim. See
AmbiqAI/ns-cmsis-nn#511 and #487.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

REJECTION = "ARM_NN_ENABLE_F16 does not support the Arm alternative half format"
NOMVE_REJECTION = "ARM_NN_ENABLE_F16 on Armv8.1-M without MVE needs GCC 15.3 or later"

# Targets where GCC accepts either format: no FP16 arithmetic, or a Cortex-M55
# with its floating-point unit disabled (soft-float ABI, or +nofp, which keeps
# integer MVE). GCC refuses the alternative format when the M55 FPU is enabled.
TARGETS = {
    "cortex-m4-soft": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft"],
    "cortex-m4-hard": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
    "cortex-m55-soft": ["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=soft"],
    "cortex-m55-nofp-mvei": ["-mcpu=cortex-m55+nofp", "-mthumb", "-mfloat-abi=hard"],
}

# Armv8.1-M with scalar FP16 arithmetic and no MVE: GCC before 15.3 emits
# Advanced SIMD half loads and stores there (#487).
# MVE, integer-only MVE (+nomve.fp) and A-profile scalar FP16, which uses
# Advanced SIMD legally, must build on every GCC. Fields: flags, affected by
# the GCC defect, M-profile (Advanced SIMD loads/stores are UNDEFINED there).
NOMVE_TARGETS = {
    "cortex-m55-nomve": (["-mcpu=cortex-m55+nomve", "-mthumb", "-mfloat-abi=hard"], True, True),
    "cortex-m85-nomve": (["-mcpu=cortex-m85+nomve", "-mthumb", "-mfloat-abi=hard"], True, True),
    "cortex-m55-mve": (["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=hard"], False, True),
    "cortex-m55-nomve.fp": (["-mcpu=cortex-m55+nomve.fp", "-mthumb", "-mfloat-abi=hard"], False, True),
    "armv8.2-a-fp16": (["-march=armv8.2-a+fp16", "-marm", "-mfloat-abi=hard", "-mfpu=neon-fp-armv8"], False, False),
}

# Advanced SIMD d-register loads/stores, as older objdump prints them, or the
# UNDEFINED form newer objdump prints for the same 0xf9xxxxxx words on
# M-profile. Other UNDEFINED lines (older objdump cannot decode some Armv8.1-M
# instructions, e.g. csinv) are not this defect.
ADVSIMD_LDST = re.compile(r"\bv(ld|st)[1-4]\.\d+\s+\{d\d+|<UNDEFINED> instruction: 0xf9[0-9a-f]{6}\b")

CONSUMER = r'''#include "arm_nnfunctions.h"
arm_cmsis_nn_status (*const consumer_argmax_f16)(const float16_t *, const cmsis_nn_dims *, int32_t, int32_t *) =
    arm_argmax_f16;
'''

# A scalar float16_t load and store, the code the #487 defect miscompiles.
HALF_ACCESS = r'''#include "arm_nnfunctions.h"
void half_copy_scaled(float16_t *dst, const float16_t *src, float scale) { *dst = (float16_t)((float)*src * scale); }
'''


def first_error(output):
    return next((line for line in output.splitlines() if "error:" in line), "")


def gcc_version(cc):
    macros = subprocess.run([cc, "-dM", "-E", "-x", "c", "/dev/null"], capture_output=True, text=True,
                            check=True).stdout
    if "#define __clang__ " in macros:
        raise SystemExit(f"{cc} is not GCC; this check covers GCC only")
    return tuple(int(re.search(rf"#define __GNUC{k}__ (\d+)", macros).group(1)) for k in ("", "_MINOR"))


def units_for(args, out):
    consumer = out / "header_only_consumer.c"
    consumer.write_text(CONSUMER)
    return {"kernel": args.source_root / "Source/BasicMathFunctions/arm_argmax_f16.c", "header-only": consumer}


def check_nomve(args, out):
    version = gcc_version(args.cc)
    units = units_for(args, out)
    half_access = out / "half_access.c"
    half_access.write_text(HALF_ACCESS)
    units["half-access"] = half_access
    obj = out / "nomve.o"
    records = []
    for target, (flags, affected, m_profile) in NOMVE_TARGETS.items():
        # F16 off must build everywhere: the guard is scoped to float16 builds.
        for f16, f32 in ((1, 0), (1, 1), (0, 1)):
            rejected = affected and f16 and version < (15, 3)
            for unit, source in units.items():
                if not f16 and unit != "kernel":
                    continue
                if not f16:
                    source = args.source_root / "Source/BasicMathFunctions/arm_argmax_f32.c"
                command = [args.cc, *flags, "-O2", f"-DARM_NN_ENABLE_F16={f16}", f"-DARM_NN_ENABLE_F32={f32}",
                           "-I" + str(args.source_root / "Include"), "-c", str(source), "-o", str(obj)]
                obj.unlink(missing_ok=True)
                built = subprocess.run(command, capture_output=True, text=True)
                output = built.stdout + built.stderr
                error = first_error(output)
                advsimd = []
                if rejected:
                    ok = built.returncode != 0 and "arm_nn_math_types_flt.h" in error and NOMVE_REJECTION in error
                else:
                    ok = built.returncode == 0
                    if ok and m_profile:
                        dump = subprocess.run([args.objdump, "-d", str(obj)], capture_output=True, text=True,
                                              check=True).stdout
                        advsimd = [line.strip() for line in dump.splitlines() if ADVSIMD_LDST.search(line)]
                        ok = not advsimd
                        error = error or (advsimd[0] if advsimd else "")
                records.append({"target": target, "gcc": list(version), "expect_rejection": rejected, "f16": f16,
                                "f32": f32,
                                "unit": unit, "command": command, "rc": built.returncode, "first_error": error,
                                "advsimd_ldst": advsimd, "ok": ok})
                print(f"{'PASS' if ok else 'FAIL'} {target} gcc {version[0]}.{version[1]} "
                      f"{'rejected' if rejected else 'builds'} F16={f16} F32={f32} {unit}"
                      + ("" if ok else f": rc={built.returncode} {error or output.strip()[:200]}"), flush=True)
    return records


def check_rejection(args, out):
    units = units_for(args, out)
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
    parser.add_argument("--objdump", help="defaults to the objdump next to --cc")
    parser.add_argument("--qemu", default="qemu-system-arm")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.objdump:
        args.objdump = re.sub(r"gcc$", "objdump", args.cc)
    with tempfile.TemporaryDirectory(prefix="f16-format-") as temp:
        out = args.output or Path(temp)
        out.mkdir(parents=True, exist_ok=True)
        results = {"rejection": check_rejection(args, out), "nomve": check_nomve(args, out),
                   "ieee_execution": check_ieee_execution(args, out)}
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        failed = [r for group in results.values() for r in group if not r["ok"]]
        if failed:
            raise SystemExit(f"{len(failed)} float16 format contract check(s) failed")
        print("float16 format contract holds")


if __name__ == "__main__":
    main()
