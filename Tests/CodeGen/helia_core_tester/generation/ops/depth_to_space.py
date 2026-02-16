"""
DepthToSpace operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpDepthToSpace(OperationBase):
    """
    DepthToSpace operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for DepthToSpace operation."""
        input_shape = self.desc['input_shape']
        block_size = self.desc.get('block_size', 2)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # DepthToSpace operation
        x = tf.nn.depth_to_space(inputs, block_size)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files - not yet implemented for DepthToSpace.
        """
        pass
