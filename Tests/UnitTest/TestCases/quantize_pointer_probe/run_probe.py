# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check actual MVE bodies with exact arrays; arithmetic inputs are exact integers."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--compiler', default='clang++')
parser.add_argument('--target')
args = parser.parse_args()
shim = r'''
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#define ARM_MATH_MVEF 1
using mve_pred16_t = unsigned;
struct float32x4_t { float v[4]; };
struct int32x4_t { int32_t v[4]; };
enum arm_cmsis_nn_status { ARM_CMSIS_NN_SUCCESS, ARM_CMSIS_NN_ARG_ERROR };
constexpr mve_pred16_t vctp32q(unsigned n) { return n < 4 ? n : 4; }
constexpr float32x4_t vldrwq_z_f32(const float *p, unsigned n) {
    float32x4_t r{}; for (unsigned i=0;i<n;i++) r.v[i]=p[i]; return r;
}
constexpr int32x4_t vdupq_n_s32(int32_t n) { return {{n,n,n,n}}; }
constexpr float32x4_t vmulq_n_f32(float32x4_t a, float n) {
    for (int i=0;i<4;i++) a.v[i]*=n; return a;
}
// Only exact small integers enter this probe; target numerical tests cover rounding/saturation.
constexpr int32x4_t vcvtaq_s32_f32(float32x4_t a) {
    int32x4_t r{}; for (int i=0;i<4;i++) r.v[i]=(int32_t)a.v[i]; return r;
}
constexpr int32x4_t vmaxq_s32(int32x4_t a, int32x4_t b) {
    for (int i=0;i<4;i++) if (a.v[i]<b.v[i]) a.v[i]=b.v[i]; return a;
}
constexpr int32x4_t vminq_s32(int32x4_t a, int32x4_t b) {
    for (int i=0;i<4;i++) if (a.v[i]>b.v[i]) a.v[i]=b.v[i]; return a;
}
constexpr int32x4_t vaddq_n_s32(int32x4_t a, int n) {
    for (int i=0;i<4;i++) a.v[i]+=n; return a;
}
constexpr void vstrbq_p_s32(int8_t *p, int32x4_t a, unsigned n) {
    for (unsigned i=0;i<n;i++) p[i]=(int8_t)a.v[i];
}
constexpr void vstrhq_p_s32(int16_t *p, int32x4_t a, unsigned n) {
    for (unsigned i=0;i<n;i++) p[i]=(int16_t)a.v[i];
}
'''
failed = False
for dtype, scalar in [('s8', 'int8_t'), ('s16', 'int16_t')]:
    name = 'arm_quantize_f32_' + dtype
    text = (args.source / ('Source/QuantizationFunctions/' + name + '.c')).read_text()
    start = text.index('arm_cmsis_nn_status\n' + name + '(')
    end = text.index('\n}', start) + 2
    source = shim + '\nconstexpr ' + text[start:end] + '\n'
    source += '''
template<int N> constexpr bool exact_arrays() {
    float input[N]{}; TYPE output[N]{};
    for(int i=0;i<N;i++) input[i]=(float)(i-7);
    if(FUNCTION(input,output,N,0,1.0f)!=ARM_CMSIS_NN_SUCCESS) return false;
    for(int i=0;i<N;i++) if(output[i] != i-7) return false;
    return true;
}
'''.replace('TYPE', scalar).replace('FUNCTION', name)
    source += '\n'.join('static_assert(exact_arrays<' + str(n) + '>());' for n in [1,2,3,4,5,7,8,9,15,16,17])
    with tempfile.TemporaryDirectory(prefix='quantize-pointer-') as directory:
        generated = Path(directory) / (dtype + '.cpp')
        generated.write_text(source)
        result = subprocess.run([args.compiler, *(['--target=' + args.target] if args.target else []),
                                 '-std=c++20', '-fsyntax-only', str(generated)])
        failed |= result.returncode != 0
raise SystemExit(1 if failed else 0)
