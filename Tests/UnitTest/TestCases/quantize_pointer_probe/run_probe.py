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
shim = (Path(__file__).resolve().parent / 'pointer_shim.h').read_text()
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
