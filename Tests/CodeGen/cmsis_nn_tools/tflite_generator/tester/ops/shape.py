"""
Shape operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpShape(OperationBase):
    """
    Shape operation - returns the shape of a tensor.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Shape operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Shape operation - returns int32 tensor
        x = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.shape(x), tf.int32),
            name='shape'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Shape outputs int32, so we don't quantize
        converter.optimizations = []
        converter.inference_input_type = tf.int8 if self.desc.get('activation_dtype') == 'S8' else tf.int16
        converter.inference_output_type = tf.int32
        
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

