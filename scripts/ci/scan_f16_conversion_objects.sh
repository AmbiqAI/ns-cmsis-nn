#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Disassemble the float16 members of a built archive and say what form of the
# MVE half<->single conversion they carry.
#
# The binutils that mis-encodes VCVTB/VCVTT.F16<->F32 also mis-renders the
# words it produced, so a matching-era objdump reports a clean archive either
# way. --root therefore has to name a toolchain of binutils 2.43 or newer;
# the caller is what guarantees that, by pinning it.
#
# Two verdicts, both of which have to be assertable, because each alone
# passes vacuously:
#
#   --expect-qform none     the scalar-form helper was selected, so no
#                           Q-register conversion may survive
#   --expect-qform present  the vector form was selected, so at least one
#                           must be there and none may be UNDEFINED
#
# UNDEFINED words are refused either way: that is what a mis-encoded operand
# above q3 turns into, and it faults on first execution.
# See AmbiqAI/ns-cmsis-nn#427.

set -euo pipefail

root=""
archive=""
workdir=""
expect_qform=""

while (($# > 0)); do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --archive) archive="$2"; shift 2 ;;
    --workdir) workdir="$2"; shift 2 ;;
    --expect-qform) expect_qform="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in root archive workdir expect_qform; do
  if [[ -z "${!required}" ]]; then
    echo "missing --${required//_/-}" >&2
    exit 2
  fi
done
case "${expect_qform}" in
  none | present) ;;
  *) echo "--expect-qform must be 'none' or 'present', got: ${expect_qform}" >&2; exit 2 ;;
esac

# The extraction runs from inside --workdir, so a relative --archive would be
# resolved against that directory instead of the caller's. Pin it here, before
# anything changes directory. See AmbiqAI/ns-cmsis-nn#427.
if [[ ! -f "${archive}" ]]; then
  echo "no such archive: ${archive} (from $(pwd))" >&2
  exit 2
fi
archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"

ar="${root}/bin/arm-none-eabi-ar"
objdump="${root}/bin/arm-none-eabi-objdump"
echo "objdump: $("${objdump}" --version | head -1)"

rm -rf "${workdir}"
mkdir -p "${workdir}"
( cd "${workdir}" && "${ar}" x "${archive}" )

# The Q-register form of the four conversions, as objdump renders them.
qform='vcvt[bt]\.(f16\.f32|f32\.f16)[[:space:]]+q'

scanned=0
undefined=0
qforms=0
for obj in "${workdir}"/*f16*; do
  [[ -e "${obj}" ]] || continue
  scanned=$((scanned + 1))
  disasm="$("${objdump}" -d "${obj}")"
  hits="$(grep -c UNDEFINED <<<"${disasm}" || true)"
  if ((hits > 0)); then
    undefined=$((undefined + hits))
    echo "::error title=UNDEFINED instruction word::$(basename "${obj}") contains an architecturally UNDEFINED word, which faults on first execution. See AmbiqAI/ns-cmsis-nn#427."
    grep -B2 UNDEFINED <<<"${disasm}" | head -20
  fi
  found="$(grep -ciE "${qform}" <<<"${disasm}" || true)"
  if ((found > 0)); then
    qforms=$((qforms + found))
    if [[ "${expect_qform}" == "none" ]]; then
      echo "::error title=Vector conversion survived::$(basename "${obj}") contains ${found} Q-register vcvt f16<->f32, but this build selected the scalar form. See AmbiqAI/ns-cmsis-nn#427."
    fi
  fi
done

if ((scanned == 0)); then
  echo "::error title=No float16 objects to scan::${archive} has no *f16* members, so this check asserted nothing. See AmbiqAI/ns-cmsis-nn#427." >&2
  exit 1
fi
echo "float16 objects scanned: ${scanned}; UNDEFINED words: ${undefined}; Q-form conversions: ${qforms}"

rc=0
((undefined == 0)) || rc=1
if [[ "${expect_qform}" == "none" ]]; then
  ((qforms == 0)) || rc=1
elif ((qforms == 0)); then
  # Without this the whole cell passes on an archive that stopped emitting
  # the instruction at all, which is not the same as emitting it correctly.
  echo "::error title=No Q-form conversion to check::No float16 object contains a Q-register vcvt f16<->f32, so a mis-encoding assembler would have nothing to get wrong here and this cell proves nothing. See AmbiqAI/ns-cmsis-nn#427." >&2
  rc=1
fi
exit "${rc}"
