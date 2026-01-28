"""
Utilities for working with TFLite models using LiteRT schema.
Extract quantization parameters, weights, biases, and tensor shapes directly from flatbuffer.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

try:
    from ai_edge_litert import schema_py_generated as litert
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False
    litert = None


def load_litert_model(tflite_path: str, subgraph_index: int = 0) -> Tuple[Any, Any]:
    """
    Load TFLite model using LiteRT schema.
    
    Args:
        tflite_path: Path to .tflite file
        subgraph_index: Index of subgraph to use (default: 0)
        
    Returns:
        Tuple of (model: litert.ModelT, subgraph: litert.SubGraphT)
    """
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")
    
    with open(tflite_path, 'rb') as f:
        model_content = f.read()
    
    # Parse the flatbuffer
    model: litert.ModelT = litert.ModelT.InitFromObj(litert.Model.GetRootAsModel(model_content))
    
    if subgraph_index < 0 or subgraph_index >= len(model.subgraphs):
        raise ValueError(
            f"Subgraph index {subgraph_index} is out of range. Model has {len(model.subgraphs)} subgraphs."
        )
    
    subgraph = model.subgraphs[subgraph_index]
    
    return model, subgraph


def get_tensor_shape_from_litert(tensor: Any) -> Optional[Tuple[int, ...]]:
    """
    Extract tensor shape from LiteRT tensor.
    
    Args:
        tensor: LiteRT TensorT object
        
    Returns:
        Tuple of shape dimensions, or None if shape is not available
    """
    if tensor.shape is not None and len(tensor.shape) > 0:
        return tuple(tensor.shape.tolist())
    return None


def get_tensor_data_from_litert(tensor: Any, model: Any) -> Optional[np.ndarray]:
    """
    Extract tensor data from LiteRT model.
    
    Args:
        tensor: LiteRT TensorT object
        model: LiteRT ModelT object
        
    Returns:
        Numpy array with tensor data, or None if buffer is empty
    """
    if tensor.buffer < 0:
        return None
    
    buffer = model.buffers[tensor.buffer]
    if buffer.data is None or len(buffer.data) == 0:
        return None
    
    # Get dtype mapping
    tensor_dtype_map = {
        litert.TensorType.FLOAT32: np.float32,
        litert.TensorType.FLOAT16: np.float16,
        litert.TensorType.INT32: np.int32,
        litert.TensorType.UINT8: np.uint8,
        litert.TensorType.INT64: np.int64,
        litert.TensorType.INT16: np.int16,
        litert.TensorType.INT8: np.int8,
        litert.TensorType.BOOL: bool,
    }
    
    dtype = tensor_dtype_map.get(tensor.type, np.int8)
    
    # Convert buffer data to numpy array
    data = buffer.data.view(dtype)
    
    # Reshape if tensor has a shape
    if tensor.shape is not None and len(tensor.shape) != 0:
        data = data.reshape(tensor.shape)
    
    return data


def get_tensor_quantization_from_litert(tensor: Any) -> Dict[str, Any]:
    """
    Extract quantization parameters from LiteRT tensor.
    
    Args:
        tensor: LiteRT TensorT object
        
    Returns:
        Dictionary with 'scale', 'zero_point', and 'per_channel' keys
    """
    if tensor.quantization is None:
        return {
            'scale': 1.0,
            'zero_point': 0,
            'per_channel': False
        }
    
    quant = tensor.quantization
    
    # Extract scales and zero points
    if quant.scale is not None and len(quant.scale) > 0:
        scales = np.array(quant.scale)
    else:
        scales = np.array([1.0])
    
    if quant.zeroPoint is not None and len(quant.zeroPoint) > 0:
        zero_points = np.array(quant.zeroPoint)
    else:
        zero_points = np.array([0])
    
    # Determine if per-channel
    per_channel = len(scales) > 1 or (quant.quantizedDimension is not None and quant.quantizedDimension >= 0)
    
    if per_channel:
        return {
            'scale': scales,
            'zero_point': zero_points,
            'per_channel': True
        }
    else:
        return {
            'scale': float(scales[0]) if len(scales) > 0 else 1.0,
            'zero_point': int(zero_points[0]) if len(zero_points) > 0 else 0,
            'per_channel': False
        }


def extract_weights_biases_from_litert(model: Any, subgraph: Any, operator_index: Optional[int] = None) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract weights and biases from LiteRT model using operator structure.
    
    This is more reliable than shape-based heuristics because it uses the
    operator's input/output tensor indices directly.
    
    Args:
        model: LiteRT ModelT object
        subgraph: LiteRT SubGraphT object
        operator_index: Optional operator index. If None, uses the first operator.
        
    Returns:
        Dictionary with 'weights' and 'biases' keys
    """
    weights = None
    biases = None
    
    # If operator_index is provided, use that operator's inputs
    if operator_index is not None and operator_index < len(subgraph.operators):
        op = subgraph.operators[operator_index]
        
        # Get input and output tensor indices to exclude them
        input_indices = set(subgraph.inputs)
        output_indices = set(subgraph.outputs)
        
        # For FullyConnected and similar ops, weights might be at index 1, biases at index 2
        # But weights might also be constants not in op.inputs
        # First, try to get from operator inputs
        if len(op.inputs) > 1:
            weights_tensor_idx = op.inputs[1]
            if weights_tensor_idx >= 0 and weights_tensor_idx < len(subgraph.tensors):
                weights_tensor = subgraph.tensors[weights_tensor_idx]
                tensor_data = get_tensor_data_from_litert(weights_tensor, model)
                tensor_shape = get_tensor_shape_from_litert(weights_tensor)
                # Only use if it's multi-dimensional (weights, not bias)
                if tensor_data is not None and tensor_shape is not None and len(tensor_shape) > 1:
                    weights = tensor_data
        
        if len(op.inputs) > 2:
            bias_tensor_idx = op.inputs[2]
            if bias_tensor_idx >= 0 and bias_tensor_idx < len(subgraph.tensors):
                bias_tensor = subgraph.tensors[bias_tensor_idx]
                biases = get_tensor_data_from_litert(bias_tensor, model)
        
        # If weights not found in operator inputs, search all tensors
        if weights is None:
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                # Skip if it's an input or output
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                # Skip if it's already in operator inputs (we checked above)
                if tensor_idx in op.inputs:
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
                if tensor_data is not None and tensor_shape is not None:
                    # Weight tensor: multi-dimensional (2D, 3D, 4D)
                    if len(tensor_shape) > 1:
                        weights = tensor_data
                        break
                    # Bias tensor: 1D (if we don't have one yet)
                    elif biases is None and len(tensor_shape) == 1:
                        biases = tensor_data
    else:
        # Fallback: find first operator and use its inputs
        if len(subgraph.operators) > 0:
            op = subgraph.operators[0]
            
            # Get input and output tensor indices to exclude them
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            # Look through operator inputs for weights and biases
            for input_idx in op.inputs:
                if input_idx in input_indices or input_idx in output_indices:
                    continue
                
                if input_idx >= 0 and input_idx < len(subgraph.tensors):
                    tensor = subgraph.tensors[input_idx]
                    tensor_data = get_tensor_data_from_litert(tensor, model)
                    
                    if tensor_data is not None:
                        tensor_shape = get_tensor_shape_from_litert(tensor)
                        if tensor_shape is None:
                            continue
                        
                        # Weight tensor: multi-dimensional (2D, 3D, 4D)
                        if weights is None and len(tensor_shape) > 1:
                            weights = tensor_data
                        # Bias tensor: 1D
                        elif biases is None and len(tensor_shape) == 1:
                            biases = tensor_data
    
    return {
        'weights': weights,
        'biases': biases
    }


def get_operator_tensors_from_litert(model: Any, subgraph: Any, operator_index: int) -> Dict[str, Any]:
    """
    Get all tensors associated with a specific operator.
    
    Args:
        model: LiteRT ModelT object
        subgraph: LiteRT SubGraphT object
        operator_index: Index of the operator
        
    Returns:
        Dictionary with 'inputs', 'outputs', 'weights', 'biases' keys
    """
    if operator_index < 0 or operator_index >= len(subgraph.operators):
        raise ValueError(f"Operator index {operator_index} is out of range")
    
    op = subgraph.operators[operator_index]
    
    # Get input tensors
    input_tensors = []
    for input_idx in op.inputs:
        if input_idx >= 0 and input_idx < len(subgraph.tensors):
            tensor = subgraph.tensors[input_idx]
            input_tensors.append({
                'index': input_idx,
                'tensor': tensor,
                'shape': get_tensor_shape_from_litert(tensor),
                'data': get_tensor_data_from_litert(tensor, model),
                'quantization': get_tensor_quantization_from_litert(tensor),
            })
    
    # Get output tensors
    output_tensors = []
    for output_idx in op.outputs:
        if output_idx >= 0 and output_idx < len(subgraph.tensors):
            tensor = subgraph.tensors[output_idx]
            output_tensors.append({
                'index': output_idx,
                'tensor': tensor,
                'shape': get_tensor_shape_from_litert(tensor),
                'data': get_tensor_data_from_litert(tensor, model),
                'quantization': get_tensor_quantization_from_litert(tensor),
            })
    
    # Extract weights and biases
    weights_biases = extract_weights_biases_from_litert(model, subgraph, operator_index)
    
    return {
        'inputs': input_tensors,
        'outputs': output_tensors,
        'weights': weights_biases.get('weights'),
        'biases': weights_biases.get('biases'),
    }
