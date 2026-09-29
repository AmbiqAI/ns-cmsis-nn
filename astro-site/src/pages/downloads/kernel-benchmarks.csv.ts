// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import type { APIRoute } from 'astro';
import { CONVOLUTION_KERNELS, ELEMENTWISE_KERNELS } from '../../data/kernel-benchmarks';

export const prerender = true;

export const GET: APIRoute = () => {
  const rows = [
    ['Kernel', 'Shape', 'REF cycles', 'DSP cycles', 'MVE cycles', 'REF / DSP', 'REF / MVE'],
    ...[...CONVOLUTION_KERNELS, ...ELEMENTWISE_KERNELS].map(row => [
      row.kernel, row.shape, row.cycles.ref, row.cycles.dsp, row.cycles.mve,
      row.cycles.ref / row.cycles.dsp, row.cycles.ref / row.cycles.mve,
    ]),
  ];
  const csv = rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
  return new Response(`${csv}\r\n`, {
    headers: { 'Content-Type': 'text/csv; charset=utf-8' },
  });
};
