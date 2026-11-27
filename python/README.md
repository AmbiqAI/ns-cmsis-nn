# ns_cmsis_nn_contracts

The one reader of an ns-cmsis-nn checkout's kernel contracts (RFC #829), shared by helia-core-tester and heliaAOT so
neither keeps its own port of the library's dispatch, sizing or weight preparation.

- `load_signatures(root)`: `Tests/KernelContracts/kernel_contracts.json`, every public prototype, cross-checked
  against the headers it names.
- `load_implementations(root, signatures)`: `Tests/KernelContracts/kernel_implementations.json`, per implementation its
  entry, wrapper route, build features, buffers and their size queries, preparation steps and numerics claim; bound to
  the signature export by sha256.
- `build_planner(root, toolchain)` and `Planner`: the host planner library (`Planner/`), built with the host C
  compiler from the checkout's own sources and cached by their content. A plan names the kernel, the manifest
  implementation it reaches and the bytes its buffers need; `buffer_bytes(target, implementation_id, buffer, **operands)`
  evaluates one manifest buffer's size query for the target; a routed plan's `route_value` is what the wrapper's
  `arm_nn_*_select()` returns on the target and `route_name` its `arm_nn_dispatch.h` constant (`route_enums(root)`); `vector_sum_s8` and
  `pack_fully_connected_per_channel_packed_s8` run the library's own preparation code.
- `open_planner(root, toolchain)`: all three, bound together.
- `probe_compiler(root, cc, flags)`: the `BuildTarget` (planner `Target`, toolchain, manifest feature tokens, library
  stamp) that a cross compiler with the firmware's exact flags builds, read by preprocessing
  `Include/arm_nn_build_target.h` instead of from a CPU table. `unmet(implementation, build.features)` lists the
  `requires` tokens that build lacks; `Planner.check_build(build)` refuses a build for another toolchain.
- `render_guard(target, toolchain, implementations)`: the C block for generated code. It defines the plan's
  `ARM_NN_EXPECT_*` and includes `arm_nn_build_target_check.h`, so a compiler that builds anything else stops with an
  `#error` per difference, and `ARM_NN_LIBRARY_STAMP()` (call it from the model's init) fails to link against a
  library archive of another build.

Every inconsistency raises (`ContractError`, `PlannerError`, `PlanRefused`, `BuildTargetError`); nothing falls back
to a default.

```
pip install "ns-cmsis-nn-contracts @ git+https://github.com/AmbiqAI/ns-cmsis-nn@<sha>#subdirectory=python"
```

```python
from ns_cmsis_nn_contracts import Target, open_planner, probe_compiler, render_guard, unmet

planner = open_planner(cmsis_nn_root, "gcc")
plan = planner.depthwise_conv_wrapper_s8(Target("mve"), input_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16),
                                         output_dims=(1, 8, 8, 16), ch_mult=1, padding=(1, 1))
plan.implementation.id, plan.ctx_bytes

build = probe_compiler(cmsis_nn_root, "arm-none-eabi-gcc", ["-mcpu=cortex-m55", "-mfloat-abi=hard", "-Ofast"])
target = planner.check_build(build)  # Target("mve"): plan with it, never with a stated one
picked = planner.implementations.require("depthwise_conv_s8_opt_3x3")
assert not unmet(picked, build.features)
model_c_prelude = render_guard(target, build.toolchain, [picked])
```

Tests: `python -m pytest python/tests` (needs a host C compiler) and `scripts/tests/test_build_target_cross.py` (needs
arm-none-eabi-gcc; `NS_CMSIS_NN_REQUIRE_CROSS=1` turns its skip into a failure).
