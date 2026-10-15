# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Compile exact-array checks of actual vector-sum bodies and SVDF output blocks.

MVE shims model active lanes. SVDF scaling is deliberately identity: these
checks establish pointer legality, not intrinsic numerical equivalence.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--compiler', default='clang++')
parser.add_argument('--target')
args = parser.parse_args()
shim = Path(__file__).with_name('pointer_shim.h').read_text()
failed = False
for name in ['arm_vector_sum_s8', 'arm_vector_sum_s8_s64',
             'arm_svdf_s8', 'arm_svdf_state_s16_s8']:
    vector = 'vector_sum' in name
    group = 'FullyConnectedFunctions' if vector else 'SVDFunctions'
    text = (args.source / 'Source' / group / (name + '.c')).read_text()
    if vector:
        start = text.index('arm_cmsis_nn_status ' + name + '(')
        end = text.index('\n}', start) + 2
        source = shim + '\nconstexpr ' + text[start:end] + '\n'
        scalar = 'int64_t' if name.endswith('s64') else 'int32_t'
        extra = '' if name.endswith('s64') else ', 0'
        source += """
template<int R,int C,bool Bias> constexpr bool check() {
    int8_t input[R*C]{}; TYPE output[R]{}; TYPE bias[R]{};
    for(int i=0;i<R*C;i++) input[i]=(int8_t)(i%9-4);
    for(int i=0;i<R;i++) bias[i]=i-2;
    if(FUNCTION(output,C,R,input,3 EXTRA,Bias?bias:nullptr)!=0) return false;
    for(int r=0;r<R;r++) {
        TYPE expected=Bias?bias[r]:0;
        for(int c=0;c<C;c++) expected+=3*input[r*C+c];
        if(output[r]!=expected) return false;
    }
    return true;
}
""".replace('TYPE',scalar).replace('FUNCTION',name).replace(' EXTRA',extra)
        source += '\n'.join(f'static_assert(check<{r},{c},{b}>());'
                            for r in [1,4,5,6,10,11] for c in range(1,34)
                            for b in ['false','true'])
    else:
        start = text.index('    int32_t num_elements = input_batches * unit_count;')
        end = text.index('\n#else', start)
        source = shim + """
constexpr void output_block(int32_t *buffer_b,int8_t *output_data,
                            int input_batches,int unit_count) {
    const int multiplier_out=1, shift_2=0, zp_out=2;
    const int out_activation_min=-10, out_activation_max=10;
""" + text[start:end] + '\n}\n'
        source += """
template<int B,int U> constexpr bool check() {
    int32_t input[B*U]{}; int8_t output[B*U]{};
    for(int i=0;i<B*U;i++) input[i]=i%31-15;
    output_block(input,output,B,U);
    for(int i=0;i<B*U;i++) {
        int expected=input[i]+2;
        if(expected < -10) expected=-10;
        if(expected > 10) expected=10;
        if(output[i]!=expected) return false;
    }
    return true;
}
"""
        source += '\n'.join(f'static_assert(check<{b},{u}>());'
                            for b in [1,2,3] for u in range(1,18))
    with tempfile.TemporaryDirectory(prefix='vector-svdf-pointer-') as directory:
        generated = Path(directory) / (name + '.cpp')
        generated.write_text(source)
        result = subprocess.run([args.compiler,
                                 *(['--target=' + args.target] if args.target else []),
                                 '-std=c++20', '-fsyntax-only', str(generated)])
        print(name, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
        failed |= result.returncode != 0
raise SystemExit(1 if failed else 0)
