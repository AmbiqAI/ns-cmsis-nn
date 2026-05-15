# Reference

This section will host the auto-generated CMSIS-NN C API reference.

The plan is to run Doxygen against `Include/*.h` to produce an XML
intermediate, then render it with
[**mkdoxy**](https://github.com/JakubAndrysek/mkdoxy) into native
MkDocs pages that share the heliaCORE theme.

Until that lands, see the upstream Doxygen reference at:

- [ARM-software/CMSIS-NN Doxygen](https://arm-software.github.io/CMSIS-NN/latest/)

The public `arm_*` API is kept 1:1 with upstream — see
[Why heliaCORE](../why.md#what-stays-upstream-aligned).
