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

Every inconsistency raises (`ContractError`, `PlannerError`, `PlanRefused`); nothing falls back to a default.

```
pip install "ns-cmsis-nn-contracts @ git+https://github.com/AmbiqAI/ns-cmsis-nn@<sha>#subdirectory=python"
```

```python
from ns_cmsis_nn_contracts import Target, open_planner

planner = open_planner(cmsis_nn_root, "gcc")
plan = planner.depthwise_conv_wrapper_s8(Target("mve"), input_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16),
                                         output_dims=(1, 8, 8, 16), ch_mult=1, padding=(1, 1))
plan.implementation.id, plan.ctx_bytes
```

Tests: `python -m pytest python/tests` (needs a host C compiler).
