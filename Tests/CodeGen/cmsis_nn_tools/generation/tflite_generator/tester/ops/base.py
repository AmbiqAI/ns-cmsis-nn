"""
Simplified base operation class for TFLite model generation.
All operations inherit from this and implement build_keras_model().
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import jinja2

from cmsis_nn_tools.core.discovery import find_tester_templates_dir

# Note: tensorflow import is only needed in subclasses for convert_to_tflite()


class OperationBase(ABC):
    """
    Base class for all CMSIS-NN operations.
    
    Each operation must implement:
    1. build_keras_model() - Construct the Keras model
    2. convert_to_tflite() - Convert model to TFLite with operation-specific quantization
    """
    
    def __init__(self, desc: Dict[str, Any], seed: int = 1):
        """
        Initialize operation with descriptor and random seed.
        
        Args:
            desc: YAML descriptor dictionary
            seed: Random seed for reproducible generation
        """
        self.desc = desc
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._litert_interpreter = None
        self._tflite_path = None
        
    @abstractmethod
    def build_keras_model(self):
        """
        Build the Keras model for this operation.
        
        Returns:
            Keras model ready for TFLite conversion
        """
        pass
    
    @abstractmethod
    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """
        Convert Keras model to TFLite with operation-specific quantization.
        
        Each operator implements its own quantization strategy based on:
        - activation_dtype (S8, S16, etc.)
        - weight_dtype (S8, S4, etc.)
        - operation-specific requirements
        
        Args:
            model: Keras model to convert
            out_path: Output path for .tflite file
            rep_seed: Representative dataset seed for quantization
        """
        pass
    
    def load_litert_interpreter(self, tflite_path: str):
        """
        Load LiteRT interpreter from .tflite file.
        
        Args:
            tflite_path: Path to .tflite file
            
        Returns:
            LiteRT interpreter instance
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import load_litert_interpreter
        
        if self._tflite_path != tflite_path or self._litert_interpreter is None:
            interpreter = load_litert_interpreter(tflite_path)
            self._litert_interpreter = interpreter
            self._tflite_path = tflite_path
        
        return self._litert_interpreter
    
    def load_litert_model(self, tflite_path: str, subgraph_index: int = 0):
        """
        Load TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file
            subgraph_index: Index of subgraph to use (default: 0)
            
        Returns:
            Tuple of (model, subgraph) from LiteRT schema
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import load_litert_model
        
        if self._tflite_path != tflite_path or not hasattr(self, '_litert_model'):
            model, subgraph = load_litert_model(tflite_path, subgraph_index)
            self._litert_model = model
            self._litert_subgraph = subgraph
            self._tflite_path = tflite_path
        
        return self._litert_model, self._litert_subgraph
    
    def extract_quantization_params(self, tflite_path: str) -> Dict[str, Any]:
        """
        Extract quantization parameters from TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file (required)
            
        Returns:
            Dictionary with quantization parameters for input, output, and weights
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        
        # Get first operator's tensors
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Get input quantization (first input tensor)
        input_quant = None
        if op_tensors['inputs']:
            input_quant = op_tensors['inputs'][0]['quantization']
        
        # Get output quantization (first output tensor)
        output_quant = None
        if op_tensors['outputs']:
            output_quant = op_tensors['outputs'][0]['quantization']
        
        # Get weight quantization (from weights tensor)
        weight_quant = None
        if op_tensors['weights'] is not None:
            # Find the weight tensor in inputs
            for input_tensor_info in op_tensors['inputs']:
                if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                    weight_quant = input_tensor_info['quantization']
                    break
        
        return {
            'input': input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'output': output_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'weight': weight_quant or input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
    
    def extract_weights_biases(self, tflite_path: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract weights and biases from TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file (required)
            
        Returns:
            Dictionary with 'weights' and 'biases' keys
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import (
            load_litert_model, extract_weights_biases_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        return extract_weights_biases_from_litert(model, subgraph, 0)
    
    def run_inference(self, tflite_path: str, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on model using LiteRT interpreter.
        
        Args:
            tflite_path: Path to .tflite file
            input_data: Input data as numpy array
            
        Returns:
            Output data as numpy array
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import run_inference_litert
        
        return run_inference_litert(tflite_path, input_data, subgraph_index=0)
    
    def generate_input_data(self) -> np.ndarray:
        """
        Generate test input data.
        
        Returns:
            Input data as numpy array
        """
        input_shape = self.desc.get('input_shape', [1, 1, 1, 1])
        # Generate random input data in appropriate range
        input_data = self.rng.integers(-32, 32, size=input_shape).astype(np.float32)
        return input_data
    
    def render_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """
        Render Jinja template with context.
        
        Args:
            template_path: Path to template file
            context: Template context dictionary
            
        Returns:
            Rendered template as string
        """
        template_dir = find_tester_templates_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        template = env.get_template(template_path)
        return template.render(**context)
    
    def get_tensor_shapes_from_litert(self, tflite_path: str) -> Dict[str, Any]:
        """
        Get input and output tensor shapes using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file
            
        Returns:
            Dictionary with 'input_shape' and 'output_shape' keys
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input_shape = op_tensors['inputs'][0]['shape'] if op_tensors['inputs'] else None
        output_shape = op_tensors['outputs'][0]['shape'] if op_tensors['outputs'] else None
        
        if input_shape is None or output_shape is None:
            raise ValueError("Missing shapes from LiteRT")
        
        return {
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    
    def get_shapes_from_litert(self, tflite_path: str, operator_index: int = 0) -> Dict[str, Any]:
        """
        Get input and output shapes using LiteRT schema (convenience wrapper).
        
        Args:
            tflite_path: Path to .tflite file
            operator_index: Index of the operator (default: 0)
            
        Returns:
            Dictionary with 'input_shapes' (list) and 'output_shapes' (list) keys
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import (
            load_litert_model, get_input_output_shapes_from_litert
        )
        
        model, subgraph = self.load_litert_model(tflite_path)
        return get_input_output_shapes_from_litert(model, subgraph, operator_index)
    
    def get_quantization_from_litert(self, tflite_path: str, operator_index: int = 0) -> Dict[str, Any]:
        """
        Get input and output quantization parameters using LiteRT schema (convenience wrapper).
        
        Args:
            tflite_path: Path to .tflite file
            operator_index: Index of the operator (default: 0)
            
        Returns:
            Dictionary with 'input_quantizations' (list) and 'output_quantizations' (list) keys
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.litert_utils import (
            get_input_output_quantization_from_litert
        )
        
        model, subgraph = self.load_litert_model(tflite_path)
        return get_input_output_quantization_from_litert(model, subgraph, operator_index)
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates.
        
        This method should be overridden by subclasses to implement
        operator-specific generation logic.
        
        Args:
            output_dir: Output directory for generated files
        """
        raise NotImplementedError("Subclasses must implement generate_c_files()")