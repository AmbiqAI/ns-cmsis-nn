#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

"""Mechanical proof for Include/Internal/arm_nn_vcvt_f16_fixup.h.

The gas bundled with Arm GNU Toolchain releases before 14.2.Rel1 mis-encodes
the Q-register form of VCVTB/VCVTT.F16<->F32, so the header emits those four
conversions as raw instruction words instead. This script is the evidence that
the substitute encoder is exact rather than merely plausible; see
AmbiqAI/ns-cmsis-nn#427.

Stage 1 assembles every q0..q7 x q0..q7 pair of all four forms twice, once as
native mnemonics and once through the header's own gas macro (lifted out of the
header by the preprocessor, so the two cannot drift), with each toolchain, and
compares raw .text bytes.

Stage 2 compiles the header's C wrappers with register allocation pinned to
each pair, which is the part stage 1 cannot cover: that the compiler
substitutes a plain `q<n>` name for the %q operand modifier for every register
the allocator can pick.

14.2.Rel1's objdump is the only decoder used, and that release must be
installed: earlier objdumps mis-render MVE and cannot judge their own
assembler's output.

This is developer-side proof, not a CI gate: it wants every Arm GNU release in
RELEASES side by side, which no runner image has.

How to run it:

    # Unpack the releases you have under one root, each in a directory named
    # for its release, so that
    #   <root>/13.2.rel1/arm-none-eabi/bin/arm-none-eabi-gcc
    # exists. The default root is /Applications/ArmGNUToolchain.
    scripts/verify_vcvt_f16_fixup.py
    ARM_TOOLCHAIN_ROOT=/opt/arm scripts/verify_vcvt_f16_fixup.py
    scripts/verify_vcvt_f16_fixup.py --keep   # leave the sources and objects

Releases that are not installed are skipped with a notice. A PASS needs the
judge plus at least one release on each side of the defect, because a run over
only-defective or only-fixed toolchains proves nothing about the difference.
"""

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TOOLCHAIN_ROOT = Path(os.environ.get("ARM_TOOLCHAIN_ROOT", "/Applications/ArmGNUToolchain"))

# Releases whose gas is exercised, oldest first. JUDGE is both the reference
# encoder and the only decoder, so it is the one release that cannot be
# skipped.
RELEASES = [
    "12.2.rel1",
    "13.2.rel1",
    "13.3.rel1",
    "14.2.rel1",
    "14.3.rel1",
    "15.2.rel1",
    "15.3.rel1",
]
JUDGE = "14.2.rel1"

# Releases expected to mis-encode the native mnemonics.
DEFECTIVE = {"12.2.rel1", "13.2.rel1", "13.3.rel1"}

CFLAGS = [
    "-mcpu=cortex-m55",
    "-mfloat-abi=hard",
    "-mfpu=auto",
    "-mthumb",
    "-DARM_MATH_MVEF",
    "-DARM_MATH_MVE_FLOAT16",
    "-DARM_NN_ENABLE_F16=1",
    "-DARM_NN_ENABLE_F32=1",
]

# mnemonic -> (word for q0, q0, header macro naming that word). Qd sits at bit
# 13 and Qm at bit 1; the base carries the direction (bit 28) and the
# top/bottom half select (bit 12).
FORMS = [
    ("vcvtb.f16.f32", 0xEE3F0E01, "ARM_NN_VCVTB_F16_F32_WORD"),
    ("vcvtt.f16.f32", 0xEE3F1E01, "ARM_NN_VCVTT_F16_F32_WORD"),
    ("vcvtb.f32.f16", 0xFE3F0E01, "ARM_NN_VCVTB_F32_F16_WORD"),
    ("vcvtt.f32.f16", 0xFE3F1E01, "ARM_NN_VCVTT_F32_F16_WORD"),
]

# The narrowing forms take the inactive-lane operand first.
WRAPPERS = [
    ("vcvtb.f16.f32", "arm_nn_vcvtbq_f16_f32", True),
    ("vcvtt.f16.f32", "arm_nn_vcvttq_f16_f32", True),
    ("vcvtb.f32.f16", "arm_nn_vcvtbq_f32_f16", False),
    ("vcvtt.f32.f16", "arm_nn_vcvttq_f32_f16", False),
]

REGS = range(8)


def tool_path(release, name):
    return TOOLCHAIN_ROOT / release / "arm-none-eabi" / "bin" / f"arm-none-eabi-{name}"


def tool(release, name):
    path = tool_path(release, name)
    if not path.exists():
        sys.exit(f"missing tool: {path}")
    return str(path)


def installed(release):
    """A release counts as present only if every tool this script drives is."""
    return all(tool_path(release, name).exists() for name in ("gcc", "objcopy", "objdump"))


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def word_for(base, qd, qm):
    return base + (qd << 13) + (qm << 1)


def includes():
    """The shim header pulls in no CMSIS Core header; CMSIS_CORE_INCLUDE is
    there only for a tree where arm_nn_types_flt.h grows one."""
    args = [f"-I{REPO / 'Include'}"]
    core = os.environ.get("CMSIS_CORE_INCLUDE")
    if core:
        args.append(f"-I{core}")
    return args


def extract_macro_text(work, release):
    """Lift each form's asm template out of the header with the preprocessor.

    Preprocessed by a defective release so the fixup branch is the live one.
    Returns mnemonic -> asm text still carrying the %q0 / %q1 operands.
    """
    src = work / "extract.c"
    body = ['#include "Internal/arm_nn_vcvt_f16_fixup.h"']
    for mnemonic, _base, word_macro in FORMS:
        body.append(f"NSVCVT_BEGIN {mnemonic} NSVCVT_MID ARM_NN_VCVT_FIXUP_ASM({word_macro}) NSVCVT_END")
    src.write_text("\n".join(body) + "\n")

    out = run([tool(release, "gcc"), "-E"] + CFLAGS + includes() + [str(src)]).stdout

    texts = {}
    for chunk in re.findall(r"NSVCVT_BEGIN(.*?)NSVCVT_END", out, re.DOTALL):
        mnemonic, literals = chunk.split("NSVCVT_MID", 1)
        pieces = re.findall(r'"(?:[^"\\]|\\.)*"', literals)
        if not pieces:
            sys.exit("could not lift the fixup asm out of the header")
        texts[mnemonic.strip()] = "".join(ast.literal_eval(p) for p in pieces)
    missing = [m for m, _b, _w in FORMS if m not in texts]
    if missing:
        sys.exit(f"no fixup asm lifted for: {', '.join(missing)}")
    return texts


def write_sources(work, macro_texts):
    """native.S and shim.S: the same 256 conversions, two ways to spell them."""
    prologue = [".syntax unified", ".thumb", ".text", ".thumb_func", "nsvcvt_probe:"]

    native = list(prologue)
    shim = list(prologue)
    for mnemonic, _base, _word_macro in FORMS:
        for qd in REGS:
            for qm in REGS:
                native.append(f"\t{mnemonic} q{qd}, q{qm}")
                shim.append(macro_texts[mnemonic].replace("%q0", f"q{qd}").replace("%q1", f"q{qm}"))

    (work / "native.S").write_text("\n".join(native) + "\n")
    (work / "shim.S").write_text("\n".join(shim) + "\n")


def text_bytes(release, source, work, tag):
    obj = work / f"{tag}.o"
    binary = work / f"{tag}.bin"
    run([tool(release, "gcc"), "-c"] + CFLAGS + [str(source), "-o", str(obj)])
    run([tool(release, "objcopy"), "-O", "binary", "--only-section=.text", str(obj), str(binary)])
    return obj, binary.read_bytes()


def disassemble(obj):
    """(symbol, address, mnemonic, operands) for every instruction in obj."""
    out = run([tool(JUDGE, "objdump"), "-d", "--no-show-raw-insn", str(obj)]).stdout
    rows = []
    symbol = "?"
    for line in out.splitlines():
        sym = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if sym:
            symbol = sym.group(1)
            continue
        # An unrecognised word has no mnemonic column; objdump renders it as a
        # trailing comment, so match that before the normal instruction shape.
        bad = re.match(r"^\s+([0-9a-f]+):.*<UNDEFINED> instruction: (0x[0-9a-f]+)", line)
        if bad:
            rows.append((symbol, int(bad.group(1), 16), "<UNDEFINED>", bad.group(2)))
            continue
        insn = re.match(r"^\s+([0-9a-f]+):\t(\S+)\s*(.*)$", line)
        if insn:
            rows.append((symbol, int(insn.group(1), 16), insn.group(2), insn.group(3).strip()))
    return rows


def stage1(work, releases, failures, summary):
    macro_texts = extract_macro_text(work, next(r for r in releases if r in DEFECTIVE))
    write_sources(work, macro_texts)

    expected = bytearray()
    for _mnemonic, base, _word_macro in FORMS:
        for qd in REGS:
            for qm in REGS:
                w = word_for(base, qd, qm)
                # .inst.w emits the high halfword first, each halfword little endian.
                expected += (w >> 16).to_bytes(2, "little") + (w & 0xFFFF).to_bytes(2, "little")

    native = {}
    shim = {}
    objects = {}
    for release in releases:
        objects[release], native[release] = text_bytes(release, work / "native.S", work, f"native_{release}")
        _obj, shim[release] = text_bytes(release, work / "shim.S", work, f"shim_{release}")

    reference = native[JUDGE]

    if reference != bytes(expected):
        failures.append(f"stage 1: {JUDGE} native encoding does not match the documented field layout")

    print("stage 1: 4 forms x 64 register pairs, assembled natively and through the shim")
    print(f"  reference: {JUDGE} native, {len(reference)} bytes, layout check "
          f"{'ok' if reference == bytes(expected) else 'FAILED'}")
    for release in releases:
        shim_ok = shim[release] == reference
        native_ok = native[release] == reference
        bad = sum(1 for _s, _a, m, _o in disassemble(objects[release]) if m == "<UNDEFINED>")
        print(f"  {release:<10} native {'matches' if native_ok else 'DIFFERS':<7} "
              f"({bad:3d} undefined words)   shim {'matches' if shim_ok else 'DIFFERS'}")
        if not shim_ok:
            failures.append(f"stage 1: shim under {release} does not match {JUDGE} native")
        if release in DEFECTIVE and native_ok:
            failures.append(f"stage 1: native under {release} unexpectedly matches {JUDGE}; "
                            "the defect this header works around is gone")
        if release == JUDGE and not native_ok:
            failures.append(f"stage 1: {JUDGE} native is not self-consistent")

    summary.append(f"stage 1: shim byte-identical to {JUDGE} native under "
                   f"{sum(1 for r in releases if shim[r] == reference)}/{len(releases)} toolchains")


def stage2(work, releases, failures, summary):
    """Compile the header's C wrappers with the allocation pinned to each pair.

    Stage 1 proves the encoder. This proves the operand path: that %q0 / %q1
    reach gas as plain `q<n>` names, for whichever registers the allocator
    picks, and that the word the shim then emits decodes back to exactly those
    registers. Each release is judged against its own assembly listing, not
    against another release: which register the allocator lands on legitimately
    differs between compiler majors.
    """
    lines = ['#include "Internal/arm_nn_vcvt_f16_fixup.h"']
    cases = []
    for mnemonic, wrapper, narrowing in WRAPPERS:
        for qd in REGS:
            for qm in REGS:
                name = f"p{len(cases)}"
                cases.append((name, mnemonic, qd, qm))
                if narrowing:
                    lines += [
                        f"void {name}(void)",
                        "{",
                        f'    register float16x8_t d __asm__("q{qd}");',
                        f'    register float32x4_t a __asm__("q{qm}");',
                        '    __asm__ volatile("" : "=w"(d));',
                        '    __asm__ volatile("" : "=w"(a));',
                        f"    d = {wrapper}(d, a);",
                        '    __asm__ volatile("" : : "w"(d));',
                        "}",
                    ]
                else:
                    lines += [
                        f"void {name}(void)",
                        "{",
                        f'    register float32x4_t d __asm__("q{qd}");',
                        f'    register float16x8_t a __asm__("q{qm}");',
                        '    __asm__ volatile("" : "=w"(a));',
                        f"    d = {wrapper}(a);",
                        '    __asm__ volatile("" : : "w"(d));',
                        "}",
                    ]
    src = work / "pinned.c"
    src.write_text("\n".join(lines) + "\n")

    by_word = {base: mnemonic for mnemonic, base, _macro in FORMS}

    print("stage 2: the same 256 conversions through the C wrappers, allocation pinned per pair")
    for release in releases:
        obj = work / f"pinned_{release}.o"
        listing = work / f"pinned_{release}.s"
        run([tool(release, "gcc"), "-c", "-O2"] + CFLAGS + includes() + [str(src), "-o", str(obj)])
        run([tool(release, "gcc"), "-S", "-O2"] + CFLAGS + includes() + [str(src), "-o", str(listing)])

        emitted = [
            (int(m.group(1), 16), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(
                r"ns_vcvt_fixup\s+(0x[0-9a-fA-F]+)\s*,\s*q(\d)\s*,\s*q(\d)", listing.read_text()
            )
        ]
        decoded = [
            (mnemonic, operands)
            for _sym, _addr, mnemonic, operands in disassemble(obj)
            if mnemonic.startswith(("vcvtb.", "vcvtt.")) or mnemonic == "<UNDEFINED>"
        ]
        undefined = sum(1 for mnemonic, _o in decoded if mnemonic == "<UNDEFINED>")

        fixup_active = release in DEFECTIVE
        if not fixup_active:
            print(f"  {release:<10} shim inactive, {len(decoded):3d} native conversions, "
                  f"{undefined} undefined words")
            if len(decoded) != len(cases):
                failures.append(f"stage 2: {release} emitted {len(decoded)} conversions, expected {len(cases)}")
            if undefined:
                failures.append(f"stage 2: {release} emitted {undefined} undefined word(s)")
            continue

        if len(emitted) != len(cases):
            failures.append(f"stage 2: {release} routed {len(emitted)} of {len(cases)} sites through the shim")
        if len(decoded) != len(emitted):
            failures.append(f"stage 2: {release} decodes {len(decoded)} conversions for {len(emitted)} shim sites")

        wrong = 0
        seen_d, seen_m = set(), set()
        for (word, qd, qm), (mnemonic, operands) in zip(emitted, decoded):
            seen_d.add(qd)
            seen_m.add(qm)
            if (mnemonic, operands) != (by_word.get(word), f"q{qd}, q{qm}"):
                wrong += 1
        print(f"  {release:<10} shim active, {len(emitted):3d} sites, {wrong} decoding other than requested, "
              f"{undefined} undefined words, Qd/Qm coverage {len(seen_d)}/8 and {len(seen_m)}/8")
        if wrong:
            failures.append(f"stage 2: {release} decodes {wrong} shim site(s) to the wrong form or registers")
        if undefined:
            failures.append(f"stage 2: {release} emitted {undefined} undefined word(s)")
        if len(seen_d) != 8 or len(seen_m) != 8:
            failures.append(f"stage 2: {release} exercised only {len(seen_d)}/8 Qd and {len(seen_m)}/8 Qm names")

    summary.append(f"stage 2: {len(cases)} wrapper sites per toolchain decode to the registers the compiler asked for")


def select_releases():
    """The installed subset of RELEASES, or None after explaining what is short.

    Skipping is deliberate: nobody has all seven unpacked, and refusing to run
    would make this proof unrunnable rather than partial. What cannot be
    skipped is the judge, and having a release on each side of the defect --
    an all-defective or all-fixed run compares the shim only against itself.
    """
    available = [r for r in RELEASES if installed(r)]
    for release in RELEASES:
        if release not in available:
            print(f"notice: {release} is not installed under {TOOLCHAIN_ROOT}, skipping it")

    if JUDGE not in available:
        print(f"cannot run: {JUDGE} is the reference encoder and the only decoder, "
              f"and it is not installed under {TOOLCHAIN_ROOT}.", file=sys.stderr)
        return None
    if not any(r in DEFECTIVE for r in available):
        print("cannot run: no release with the defective assembler is installed, so "
              f"nothing here would exercise the shim. Install one of {sorted(DEFECTIVE)}.",
              file=sys.stderr)
        return None
    if not any(r not in DEFECTIVE for r in available):
        print("cannot run: no release with a correct assembler is installed, so there "
              "is nothing to judge the shim against.", file=sys.stderr)
        return None

    print(f"toolchain root: {TOOLCHAIN_ROOT}")
    print(f"exercising: {', '.join(available)}   judge: {JUDGE}")
    print()
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the generated sources and objects")
    args = parser.parse_args()

    releases = select_releases()
    if releases is None:
        return 1

    work = Path(tempfile.mkdtemp(prefix="nsvcvt-"))
    failures = []
    summary = []
    try:
        stage1(work, releases, failures, summary)
        print()
        stage2(work, releases, failures, summary)
    finally:
        if args.keep:
            print(f"\nartifacts: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print()
    for line in summary:
        print(line)
    if failures:
        print()
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS: the shim encodes every form and register pair exactly as a correct assembler does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
