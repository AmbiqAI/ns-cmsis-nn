# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Keras import compatibility for TensorFlow version skew."""

import inspect

try:
    import keras  # Keras 3 package
except ModuleNotFoundError:
    try:
        import tf_keras as keras  # Legacy standalone package
    except ModuleNotFoundError:
        from tensorflow import keras  # Last-resort fallback


def _patch_input_signature():
    """Bridge Keras 2 style `batch_input_shape` to Keras 3 `batch_shape`."""
    input_fn = keras.layers.Input
    try:
        sig = inspect.signature(input_fn)
    except (TypeError, ValueError):
        return

    if "batch_input_shape" in sig.parameters:
        return

    def input_compat(*args, **kwargs):
        if "batch_input_shape" in kwargs and "batch_shape" not in kwargs:
            kwargs["batch_shape"] = kwargs.pop("batch_input_shape")
        return input_fn(*args, **kwargs)

    keras.layers.Input = input_compat


_patch_input_signature()
