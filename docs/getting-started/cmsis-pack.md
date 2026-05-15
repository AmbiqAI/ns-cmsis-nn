# CMSIS-Pack

heliaCORE publishes a CMSIS-Pack on every release. The pack ships with
two Cvariants so you can choose between rebuilding from source or
linking the prebuilt `.a` we already validated in CI.

## Install

```bash
VERSION=7.25.0
curl -LO https://github.com/AmbiqAI/ns-cmsis-nn/releases/download/v${VERSION}/Ambiq.NS-CMSIS-NN.${VERSION}.pack

# CMSIS-Toolbox
cpackget add Ambiq.NS-CMSIS-NN.${VERSION}.pack

# Keil MDK / older tooling
# Double-click the .pack file, or import via Pack Installer.
```

## Pick a Cvariant

The pack defines the component **`Ambiq::NN Lib`** with two variants:

| Cvariant   | What you get                                                  | When to use                            |
|------------|---------------------------------------------------------------|----------------------------------------|
| `Source`   | The CMSIS-NN sources compiled by your project's toolchain.    | You want maximum control / portability.|
| `Prebuilt` | A vendored `libns-cmsis-nn.a` we built with GCC 13.2.         | You want fewer moving parts.           |

In your `.cproject` or IDE, select **one** of:

```xml
<component Cclass="Ambiq" Cgroup="NN Lib" Cvariant="Source"   Cversion="7.25.0"/>
<component Cclass="Ambiq" Cgroup="NN Lib" Cvariant="Prebuilt" Cversion="7.25.0"/>
```

## Prebuilt — supported architectures

The `Prebuilt` Cvariant is gated by per-arch conditions. The pack ships
GCC-built archives for:

- ARMv6-M (`cortex-m0` / `cortex-m0+`)
- ARMv7E-M (`cortex-m4` with FPv4 SP-D16)
- ARMv8.1-M MVE (`cortex-m55`)

If your target doesn't match one of these, switch to the `Source`
Cvariant — the kernels will be recompiled by your toolchain.

!!! warning "Toolchain note"
    The prebuilt `.a` files are GCC-only today. If your project uses
    Arm Compiler 6 (ARMClang), select the `Source` Cvariant. Mixing
    GCC-built archives into an ARMClang link will not work.

## Reference

- Pack manifest: [`Ambiq.NS-CMSIS-NN.pdsc`](https://github.com/AmbiqAI/ns-cmsis-nn/blob/main/Ambiq.NS-CMSIS-NN.pdsc)
- Pack build script: [`gen_pack.sh`](https://github.com/AmbiqAI/ns-cmsis-nn/blob/main/gen_pack.sh)
