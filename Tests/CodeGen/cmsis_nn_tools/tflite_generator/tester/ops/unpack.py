"""
Unpack operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpUnpack(OperationBase):
    """
    Unpack operation - splits a tensor along an axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Unpack operation."""
        input_shape = self.desc['input_shape']
        axis = self.desc.get('axis', 0)
        num_tensors = self.desc.get('num_tensors', 1)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Adjust axis to account for batch dimension removal
        if axis >= 0:
            axis_adjusted = axis - 1 if axis > 0 else axis
        else:
            axis_adjusted = axis
        
        # Unpack operation - split along axis
        x = tf.unstack(inputs, axis=axis_adjusted, num=num_tensors)
        
        # For TFLite, we need to return a single output
        # We'll use the first unpacked tensor as output
        output = x[0] if len(x) > 0 else inputs
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

