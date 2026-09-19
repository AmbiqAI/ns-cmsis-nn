# Hard Swish compatibility arithmetic regression

The native GNU CTest target recompiles the production compatibility kernel with
ASan, UBSan and shift-base checking enabled. It tests both s16 helpers against
wider arithmetic references, including rounding ties and extreme shift counts.
The `shift` and `multiply` executable arguments isolate the helper checks.

`kernel_check.h` is also called by the existing `test_arm_hard_swish_s8` Unity
suite. It compares eight quantization cases with LiteRT at sizes 1, 7, 8, 9, 256,
4159, 4160 and 4161, including the MVE vector/table boundary and output sentinels.
The host probe is omitted in cross configurations; target execution uses Unity.

Regenerate the independent reference outputs with:

```sh
python -m pip install ai-edge-litert==2.1.6 numpy==2.5.1
python Tests/UnitTest/TestCases/hard_swish_compat_probe/generate.py
```

The eight checked-in `.tflite` files are small single-operator HARD_SWISH models
with input/output shape `[1,256]`, INT8 tensors and no learned weights. Cases 0–5
preserve the AOT #387 compatibility regression models; cases 6–7 add ReLU
exponents zero and minus one. All model inputs are the complete INT8 ramp.
`models.json` records exact model hashes, serialized quantization, derived kernel
parameters and expected output hashes. Regeneration needs neither AOT nor CORE
binaries. The generated goldens and model fixtures use Apache-2.0; the probe
source uses the repository's Ambiq license.
