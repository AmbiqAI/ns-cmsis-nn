# ns-cmsis-nn host planner

Plans ns-cmsis-nn layers ahead of time for a target build (RFC #829): which kernel a call reaches, the size and role
of its `ctx` buffer, and the size of any prepared constant stream. It reuses the library's own route selectors
(`Include/Internal/arm_nn_dispatch.h`) and scratch-size queries, compiled for the host, so a planner and the firmware
cannot disagree. API: `include/arm_nn_plan.h`.

- The target ISA (scalar, DSP, MVE), `ARM_MATH_AUTOVECTORIZE` and `CMSIS_NN_USE_SINGLE_ROUNDING` are runtime inputs.
- The target toolchain is fixed per build (`ARM_NN_PLAN_TOOLCHAIN=gcc|armclang`) because a few routes and sizes
  depend on it at compile time; a plan for another toolchain returns `ARM_NN_PLAN_E_TARGET`.
- Covered so far: `arm_depthwise_conv_wrapper_s8`, `arm_depthwise_conv_s8_opt` (planar or channelwise for a given
  `ctx->size`), `arm_fully_connected_wrapper_s8`, `arm_fully_connected_per_channel_packed_s8` and `arm_avgpool_s8` /
  `arm_max_pool_s8`.
- `arm_nn_plan_buffer_bytes` answers every buffer `size_query` in `Tests/KernelContracts/kernel_implementations.json`
  for a target ISA, with the library's own query; a new query in the manifest needs an entry there.

```
cmake -S Planner -B build/plan -DARM_NN_PLAN_TOOLCHAIN=gcc
cmake --build build/plan
```

`sources.txt` lists the sources (planner and the library queries it calls); `scripts/tests/test_planner.py` builds
from it and checks every plan against those queries.
