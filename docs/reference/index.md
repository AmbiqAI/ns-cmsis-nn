# Reference

This section will host the auto-generated heliaCORE C API reference: inherited
CMSIS-NN-compatible entry points plus Ambiq/HELIA kernel additions exposed by
`ns-cmsis-nn`.

The plan is to run Doxygen against `Include/*.h` to produce an XML
intermediate, then render it with
[**mkdoxy**](https://github.com/JakubAndrysek/mkdoxy) into native
MkDocs pages that share the heliaCORE theme.

Until that lands, inherited CMSIS-NN-compatible APIs can be cross-referenced
against the upstream Doxygen reference:

- [ARM-software/CMSIS-NN Doxygen](https://arm-software.github.io/CMSIS-NN/latest/)

The full heliaCORE reference will be generated from this repository's headers so
Ambiq-specific kernels and integration APIs are documented alongside inherited
CMSIS-NN-compatible functions. See [Why heliaCORE](../why.md#what-stays-compatible).
