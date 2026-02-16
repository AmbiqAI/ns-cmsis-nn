"""
Utilities for working with TFLite models using LiteRT schema.
Extract quantization parameters, weights, biases, and tensor shapes directly from flatbuffer.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

try:
    from ai_edge_litert import schema_py_generated as litert
    LITERT_AVAILABLE = True
    # Try to import interpreter, but don't fail if it's not available
    try:
        from ai_edge_litert.interpreter import Interpreter as LitertInterpreter
    except ImportError:
        LitertInterpreter = None
except ImportError:
    LITERT_AVAILABLE = False
    litert = None
    LitertInterpreter = None


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
                # Only use if it's multi-dimensional (weights, not bias) and not a small parameter tensor
                # Skip tensors with shape [2], [2,2] or similar small shapes that are likely parameters
                if (tensor_data is not None and tensor_shape is not None and len(tensor_shape) > 1 and
                    np.prod(tensor_shape) > 4):  # Skip small parameter tensors
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
                    # Prefer 3D or 4D tensors, and skip small parameter tensors
                    if len(tensor_shape) > 1 and np.prod(tensor_shape) > 4:
                        # Prefer 3D or 4D tensors for weights
                        if len(tensor_shape) >= 3:
                            weights = tensor_data
                            break
                        elif weights is None:  # Use 2D as fallback if no 3D/4D found
                            weights = tensor_data
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
                        # Skip small parameter tensors (like dilation, strides)
                        if weights is None and len(tensor_shape) > 1 and np.prod(tensor_shape) > 4:
                            # Prefer 3D or 4D tensors for weights
                            if len(tensor_shape) >= 3:
                                weights = tensor_data
                            elif weights is None:  # Use 2D as fallback if no 3D/4D found
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


def get_input_output_shapes_from_litert(model: Any, subgraph: Any, operator_index: int = 0) -> Dict[str, Tuple[int, ...]]:
    """
    Extract input and output shapes from LiteRT model for a specific operator.
    
    Args:
        model: LiteRT ModelT object
        subgraph: LiteRT SubGraphT object
        operator_index: Index of the operator (default: 0)
        
    Returns:
        Dictionary with 'input_shapes' (list) and 'output_shapes' (list) keys
    """
    op_tensors = get_operator_tensors_from_litert(model, subgraph, operator_index)
    
    input_shapes = [tensor['shape'] for tensor in op_tensors['inputs'] if tensor['shape'] is not None]
    output_shapes = [tensor['shape'] for tensor in op_tensors['outputs'] if tensor['shape'] is not None]
    
    return {
        'input_shapes': input_shapes,
        'output_shapes': output_shapes
    }


def get_input_output_quantization_from_litert(model: Any, subgraph: Any, operator_index: int = 0) -> Dict[str, Any]:
    """
    Extract input and output quantization parameters from LiteRT model for a specific operator.
    
    Args:
        model: LiteRT ModelT object
        subgraph: LiteRT SubGraphT object
        operator_index: Index of the operator (default: 0)
        
    Returns:
        Dictionary with 'input_quantizations' (list) and 'output_quantizations' (list) keys
    """
    op_tensors = get_operator_tensors_from_litert(model, subgraph, operator_index)
    
    input_quantizations = [tensor['quantization'] for tensor in op_tensors['inputs']]
    output_quantizations = [tensor['quantization'] for tensor in op_tensors['outputs']]
    
    return {
        'input_quantizations': input_quantizations,
        'output_quantizations': output_quantizations
    }


def load_litert_interpreter(tflite_path: str) -> Any:
    """
    Load LiteRT interpreter from .tflite file.
    Raises ImportError if LiteRT is not available.
    
    Args:
        tflite_path: Path to .tflite file
        
    Returns:
        LiteRT or TFLite Interpreter instance
    """
    if not LITERT_AVAILABLE or LitertInterpreter is None:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")
    
    interpreter = LitertInterpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference_with_litert(interpreter: Any, input_data: np.ndarray, model: Any = None, subgraph: Any = None, operator_index: int = 0) -> np.ndarray:
    """
    Run inference using LiteRT interpreter with optional shape validation.
    
    Args:
        interpreter: LiteRT interpreter instance
        input_data: Input data as numpy array
        model: Optional LiteRT ModelT object for shape validation
        subgraph: Optional LiteRT SubGraphT object for shape validation
        operator_index: Index of the operator (default: 0)
        
    Returns:
        Output data as numpy array
    """
    # If LiteRT model is provided, use it for shape validation
    if model is not None and subgraph is not None:
        try:
            shapes = get_input_output_shapes_from_litert(model, subgraph, operator_index)
            if shapes['input_shapes']:
                expected_shape = shapes['input_shapes'][0]
                
                # Validate and reshape input if needed
                input_shape = input_data.shape
                if input_shape != tuple(expected_shape):
                    # Handle batch size mismatch
                    if len(input_shape) == len(expected_shape) and input_shape[0] != expected_shape[0]:
                        batch_size = input_shape[0]
                        expected_batch = expected_shape[0]
                        
                        if expected_batch == 1 and batch_size > 1:
                            # Process each batch item separately
                            outputs = []
                            for i in range(batch_size):
                                batch_input = input_data[i:i+1]
                                interpreter.set_tensor(interpreter.get_input_details()[0]['index'], batch_input)
                                interpreter.invoke()
                                output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
                                outputs.append(output_data)
                            return np.concatenate(outputs, axis=0)
                        else:
                            input_data = input_data.reshape(expected_shape)
                    else:
                        input_data = input_data.reshape(expected_shape)
        except Exception:
            # Fallback to interpreter-based shape extraction
            pass
    
    # Get input tensor index and expected shape
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    expected_shape = input_details[0]['shape']
    
    # Handle batch size mismatch
    input_shape = input_data.shape
    if input_shape != tuple(expected_shape):
        # If batch dimension differs, process batch by batch
        if len(input_shape) == len(expected_shape) and input_shape[0] != expected_shape[0]:
            batch_size = input_shape[0]
            expected_batch = expected_shape[0]
            
            if expected_batch == 1 and batch_size > 1:
                # Process each batch item separately and concatenate
                outputs = []
                for i in range(batch_size):
                    batch_input = input_data[i:i+1]  # Keep batch dimension as 1
                    interpreter.set_tensor(input_index, batch_input)
                    interpreter.invoke()
                    output_details = interpreter.get_output_details()
                    output_index = output_details[0]['index']
                    batch_output = interpreter.get_tensor(output_index)
                    outputs.append(batch_output)
                return np.concatenate(outputs, axis=0)
            else:
                # Reshape to match expected shape
                input_data = input_data.reshape(expected_shape)
        else:
            # Reshape to match expected shape
            input_data = input_data.reshape(expected_shape)
    
    # Set input tensor
    interpreter.set_tensor(input_index, input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)
    
    return np.array(output_data)


def run_inference_litert(tflite_path: str, input_data: np.ndarray, subgraph_index: int = 0) -> np.ndarray:
    """
    Run inference using LiteRT interpreter.
    
    Args:
        tflite_path: Path to .tflite file
        input_data: Input data as numpy array
        subgraph_index: Index of subgraph to use (default: 0)
        
    Returns:
        Output data as numpy array
    """
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")
    
    # Load LiteRT interpreter
    interpreter = load_litert_interpreter(tflite_path)
    
    # Use LiteRT schema for shape validation if available
    model = None
    subgraph = None
    try:
        model, subgraph = load_litert_model(tflite_path, subgraph_index)
    except Exception:
        # Fallback to interpreter-based shape extraction if schema loading fails
        pass

    # Run inference using LiteRT interpreter - pass model/subgraph for proper shape validation
    return run_inference_with_litert(interpreter, input_data, model=model, subgraph=subgraph, operator_index=0)


def run_inference_litert_tensor(tflite_path: str, input_data: np.ndarray, tensor_index: int, subgraph_index: int = 0) -> np.ndarray:
    """
    Run inference using LiteRT interpreter and return a specific tensor by index.

    Args:
        tflite_path: Path to .tflite file
        input_data: Input data as numpy array
        tensor_index: Tensor index to fetch (LiteRT schema tensor index)
        subgraph_index: Index of subgraph to use (default: 0)

    Returns:
        Tensor data as numpy array
    """
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    interpreter = load_litert_interpreter(tflite_path)

    # Use LiteRT schema for shape validation if available
    model = None
    subgraph = None
    try:
        model, subgraph = load_litert_model(tflite_path, subgraph_index)
    except Exception:
        pass

    # Reuse the same input handling logic
    _ = run_inference_with_litert(interpreter, input_data, model=model, subgraph=subgraph, operator_index=0)

    # Fetch requested tensor by index
    return np.array(interpreter.get_tensor(int(tensor_index)))
