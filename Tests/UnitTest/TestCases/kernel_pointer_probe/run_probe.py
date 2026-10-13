# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Constant-evaluate actual production bodies; never rewrite pointer expressions."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import re

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--compiler', default='clang++')
p.add_argument('--target')
p.add_argument('--case', choices=['scalar', 'dsp', 'mve', 'taps1', 'taps4', 'batch', 'f16', 'f32'], required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent

def body(path, name):
    source = (a.source / path).read_text()
    declaration = re.search(r'(?m)^(?![ /*])[^\n]*\b' + re.escape(name) + r'\(', source)
    if declaration is None:
        raise ValueError('Missing production declaration: ' + name)
    pos = source.index(name + '(', declaration.start())
    start = source.rfind('\n', 0, pos) + 1
    # Wrapped signatures begin on the previous line.
    if source[start:pos].strip() == '':
        start = source.rfind('\n', 0, start - 1) + 1
    end = source.index('{', pos)
    depth = 1
    cursor = end + 1
    while depth:
        depth += (source[cursor] == '{') - (source[cursor] == '}')
        cursor += 1
    signature = source[start:end]
    signature = signature.replace('__STATIC_FORCEINLINE', '').replace('static ', '')
    return 'constexpr ' + signature + source[end:cursor] + '\n'

s = '#include "pointer_shim.h"\n'
case = a.case
if case in ('scalar', 'dsp', 'mve'):
    path = 'Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c'
    if case == 'mve':
        s += '#define ARM_MATH_MVEI 1\n'
        s += body(path, 'arm_nn_mat_mult_nt_t_s8_mve')
    else:
        if case == 'dsp':
            s += '#define ARM_MATH_DSP 1\n'
        s += body(path, 'arm_nn_mat_mult_nt_t_s8')
    s += '''
template<int M,int N,int K,int LS,int DS> constexpr bool matrix() {
 int8_t lhs[(M-1)*LS+K]={},rhs[N*K]={},out[(M-1)*DS+N]={};
 int32_t bias[N]={},mult[N]={},shift[N]={};
 for(int i=0;i<M;i++) for(int k=0;k<K;k++) lhs[i*LS+k]=(i+k)%3-1;
 for(int n=0;n<N;n++) for(int k=0;k<K;k++) rhs[n*K+k]=(n+k)%3-1;
 for(int n=0;n<N;n++) bias[n]=n%3-1;
'''
    if case == 'mve':
        s += 'arm_nn_mat_mult_nt_t_s8_mve(bias,lhs,rhs,out,mult,shift,M,N,K,0,-128,127,DS,LS,false);\n'
    else:
        s += 'arm_nn_mat_mult_nt_t_s8(nullptr,lhs,rhs,bias,out,mult,shift,M,N,K,0,0,-128,127,DS,LS);\n'
    s += '''
 for(int i=0;i<M;i++) for(int n=0;n<N;n++) {
  int ref=bias[n];for(int k=0;k<K;k++)ref+=lhs[i*LS+k]*rhs[n*K+k];
  if(out[i*DS+n]!=ref)return false;
 }
 return true;
}
static_assert(matrix<4,4,17,21,4>());
static_assert(matrix<5,5,3,7,5>());
static_assert(matrix<1,3,1,5,3>());
'''
    if case == 'mve':
        s += 'static_assert(matrix<8,4,2,5,7>());\nstatic_assert(matrix<5,5,3,7,9>());\n'
elif case in ('f16', 'f32'):
    s += '#undef ARM_MATH_AUTOVECTORIZE\n#define ARM_MATH_MVEF 1\n#define ARM_MATH_MVE_FLOAT16 1\n#define ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8 (65535/7)\n'
    s += body('Source/TransposeFunctions/arm_transpose_' + case + '.c', 'arm_transpose_2d_kernel_' + case)
    s += '''
template<int R,int C> constexpr bool transpose() {
 float input[R*C]={},output[R*C]={};
 for(int i=0;i<R*C;i++)input[i]=(float)i;
 arm_transpose_2d_kernel_''' + case + '''(input,output,R,C);
 for(int r=0;r<R;r++)for(int c=0;c<C;c++)if(output[c*R+r]!=input[r*C+c])return false;
 return true;
}
static_assert(transpose<16,9>());
static_assert(transpose<17,9>());
static_assert(transpose<9,17>());
static_assert(transpose<2,3>());
'''
else:
    path = 'Source/ConvolutionFunctions/arm_depthwise_conv_s16.c'
    for name in ('dw_s16_ker', 'dw_s16_mac', 'dw_s16_taps1', 'dw_s16_taps4'):
        s += body(path, name)
    if case == 'batch':
        s += body(path, 'dw_s16_mve')
        s += '''
template<int C,int CM,int B,int W,int H,int K,int D,int S> constexpr bool depthwise() {
 constexpr int OC=C*CM,OX=(W-(K-1)*D-1)/S+1,OY=(H-(K-1)*D-1)/S+1;
 int16_t input[B*H*W*C]={},output[B*OY*OX*OC]={};int8_t kernel[K*K*OC]={};
 int32_t mult[OC]={},shift[OC]={};int64_t bias[OC]={};
 for(int i=0;i<B*H*W*C;i++)input[i]=i%3-1;
 for(int i=0;i<K*K*OC;i++)kernel[i]=i%3-1;
 dw_s16_mve(input,B,W,H,C,kernel,CM,K,K,0,0,S,S,bias,output,shift,mult,OX,OY,-32768,32767,D,D,0);
 for(int b=0;b<B;b++)for(int y=0;y<OY;y++)for(int x=0;x<OX;x++)for(int c=0;c<OC;c++) {
 int ref=0;for(int ky=0;ky<K;ky++)for(int kx=0;kx<K;kx++)ref+=input[((b*H+y*S+ky*D)*W+x*S+kx*D)*C+c/CM]*kernel[(ky*K+kx)*OC+c];
 if(output[((b*OY+y)*OX+x)*OC+c]!=ref)return false;
 }return true;
}
static_assert(depthwise<8,1,2,7,4,3,1,1>());
static_assert(depthwise<3,1,2,5,4,3,1,1>());
static_assert(depthwise<3,2,2,7,4,3,1,1>());
static_assert(depthwise<2,4,2,7,4,3,1,1>());
static_assert(depthwise<5,1,1,9,7,3,2,2>());
'''
    else:
        s += '''
template<int NY,int NX,int D,int OC> constexpr bool taps() {
 constexpr int W=(NX-1)*D+4, H=(NY-1)*D+1;
 int16_t input[W*H*OC]={};int8_t kernel[NY*NX*OC]={};
 for(int i=0;i<W*H*OC;i++)input[i]=i%3-1;
 for(int i=0;i<NY*NX*OC;i++)kernel[i]=i%3-1;
 for(int oc=0;oc<OC;oc+=4) {
 unsigned lanes=(unsigned)ARM_NN_MIN(4,OC-oc);unsigned p=lanes;
 int32x4_t a[4]={};
'''
        if case == 'taps1':
            s += 'a[3]=dw_s16_taps1(a[3],input+3*OC+oc,kernel+oc,DW_S16_IN_VEC_P,{},p,NY,NX,D*OC,D*W*OC,OC,NX*OC);\n'
        else:
            s += 'dw_s16_taps4(a,input+oc,input+OC+oc,input+2*OC+oc,input+3*OC+oc,kernel+oc,DW_S16_IN_VEC_P,{},p,NY,NX,D*OC,D*W*OC,OC,NX*OC);\n'
        s += '''
 for(unsigned c=0;c<lanes;c++) {
 int ref=0;for(int y=0;y<NY;y++)for(int x=0;x<NX;x++)ref+=input[(y*D*W+x*D+3)*OC+oc+c]*kernel[(y*NX+x)*OC+oc+c];
 if(a[3][c]!=ref)return false;
 }
 }return true;
}
static_assert(taps<3,3,1,8>());
static_assert(taps<2,3,2,5>());
static_assert(taps<1,1,1,3>());
'''
with tempfile.TemporaryDirectory(prefix='kernel-pointer-') as directory:
    generated = Path(directory) / (case + '.cpp')
    generated.write_text(s)
    subprocess.run([a.compiler, *(['--target=' + a.target] if a.target else []), '-std=c++20', '-fsyntax-only', '-fconstexpr-steps=10000000', '-I' + str(here), str(generated)], check=True)
