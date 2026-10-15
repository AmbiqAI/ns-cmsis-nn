# SVDF unit-test metadata

SVDF test callers should initialize every automatic metadata object passed through the public interface, including state and output dimensions that the current kernels ignore. A zero-initialized object gives these ignored arguments a defined value without assigning a new shape contract. Existing input, feature, time and bias dimensions continue to carry their current meanings.

The proposed change covers the state/output declarations in the s8 and s16-state/s8-output SVDF suites. Kernel implementations, reference outputs, input sequences and tolerances are unchanged. Strict compilation must reject the original uninitialized declarations and accept initialized declarations; the affected existing numerical suites must retain their outputs.
