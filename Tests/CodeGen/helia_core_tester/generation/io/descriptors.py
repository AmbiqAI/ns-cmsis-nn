"""
Descriptor ingestion with dtype validation and kernel resolution.
"""

import yaml
import os
from typing import Dict, Any, List
from pathlib import Path


# Allowed dtype combinations
ALLOWED_DTYPE_COMBOS = {
    ('S8', 'S8'): 's8',
    ('S16', 'S8'): 's16', 
    ('S8', 'S4'): 's4'
}


def validate_dtype_combo(activation_dtype: str, weight_dtype: str) -> bool:
    """
    Validate dtype combination.
    
    Args:
        activation_dtype: Activation dtype (S8, S16)
        weight_dtype: Weight dtype (S8, S4)
        
    Returns:
        True if combination is allowed
    """
    return (activation_dtype, weight_dtype) in ALLOWED_DTYPE_COMBOS


def _validate_and_normalize_descriptor(desc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a single descriptor dictionary.
    """
    required_fields = ['operator', 'activation_dtype', 'weight_dtype']
    for field in required_fields:
        if field not in desc:
            raise ValueError(f"Missing required field: {field}")

    # Normalize dtypes to upper-case for consistency
    desc['activation_dtype'] = str(desc['activation_dtype']).upper()
    desc['weight_dtype'] = str(desc['weight_dtype']).upper()

    if not validate_dtype_combo(desc['activation_dtype'], desc['weight_dtype']):
        raise ValueError(f"Unsupported dtype combination: {desc['activation_dtype']} x {desc['weight_dtype']}")

    operator = desc['operator']

    # Operator-specific dtype constraints (prevent runtime crashes/hardfaults)
    if operator in ['Conv2D', 'DepthwiseConv2D', "FullyConnected"] and desc.get('weight_dtype') == 'S4':
        raise ValueError(f"{operator} with S4 weights is not supported by the current generator")

    if 'variations' not in desc:
        if operator == 'FullyConnected':
            if 'input_shape' not in desc or 'filter_shape' not in desc:
                raise ValueError("FullyConnected requires input_shape and filter_shape")
        elif operator in ['Conv2D', 'DepthwiseConv2D']:
            if 'input_shape' not in desc or 'filter_shape' not in desc:
                raise ValueError(f"{operator} requires input_shape and filter_shape")
        elif operator in ['MatMul', 'Elementwise', 'Add', 'Sub', 'Mul', 'Maximum', 'Minimum',
                          'Equal', 'NotEqual', 'Greater', 'GreaterEqual', 'Less', 'LessEqual']:
            if 'input_1_shape' not in desc or 'input_2_shape' not in desc:
                raise ValueError(f"{operator} requires input_1_shape and input_2_shape")
        elif operator == 'Pooling':
            if 'input_shape' not in desc or ('pool_size' not in desc and 'filter_shape' not in desc):
                raise ValueError(f"{operator} requires input_shape and pool_size (or filter_shape)")
        elif operator in ['Relu', 'Relu6', 'LeakyRelu', 'Softmax', 'Quantize', 'Dequantize',
                         'Transpose', 'StridedSlice', 'Pad', 'LSTM', 'SVDF',
                         'Mean', 'ReduceMax', 'ReduceMin', 'ArgMax', 'ArgMin', 'TransposeConv',
                         'Tanh', 'Logistic', 'HardSwish', 'PReLU', 'Fill', 'ZerosLike',
                         'Reshape', 'Shape', 'Slice', 'Squeeze', 'SpaceToDepth', 'DepthToSpace',
                         'Split', 'Pack', 'Unpack', 'Concatenation', 'SpaceToBatchND', 'BatchToSpaceND', 'VariableUpdate']:
            if 'input_shape' not in desc:
                raise ValueError(f"{operator} requires input_shape")
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    for shape_key in ['input_shape', 'filter_shape', 'input_1_shape', 'input_2_shape', 'pool_size']:
        if shape_key in desc:
            desc[shape_key] = list(desc[shape_key])

    if 'hint' not in desc:
        desc['hint'] = {}

    return desc


def load_descriptor(desc_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate YAML descriptor(s) from a file.
    Supports multiple descriptors in a single YAML file (separated by ---).
    
    Args:
        desc_path: Path to YAML descriptor file
        
    Returns:
        List of validated descriptor dictionaries (one per YAML document in the file)
    """
    with open(desc_path, 'r') as f:
        documents = list(yaml.safe_load_all(f))
    
    # Filter out None documents (empty separators)
    documents = [doc for doc in documents if doc is not None]
    
    if not documents:
        raise ValueError(f"No valid descriptors found in {desc_path}")
    
    # Validate and normalize each descriptor
    descriptors = []
    for i, doc in enumerate(documents):
        try:
            validated_desc = _validate_and_normalize_descriptor(doc)
            descriptors.append(validated_desc)
        except Exception as e:
            raise ValueError(f"Error validating descriptor {i+1} in {desc_path}: {e}")
    
    return descriptors


def resolve_kernel(desc: Dict[str, Any]) -> str:
    """
    Resolve kernel symbol from descriptor.
    
    Args:
        desc: YAML descriptor dictionary
        
    Returns:
        Kernel symbol string
    """
    # Check if explicit kernel is specified
    if 'hint' in desc and 'kernel' in desc['hint']:
        return desc['hint']['kernel']
        
    # Resolve from dtype combination
    activation_dtype = desc['activation_dtype']
    weight_dtype = desc['weight_dtype']
    
    key = (activation_dtype, weight_dtype)
    if key not in ALLOWED_DTYPE_COMBOS:
        raise ValueError(f"Unsupported dtype combination: {activation_dtype} x {weight_dtype}")
        
    # Map to actual kernel symbols
    kernel_map = {
        ('S8', 'S8'): 'arm_fully_connected_wrapper_s8',
        ('S16', 'S8'): 'arm_fully_connected_s16_wrapper',
        ('S8', 'S4'): 'arm_fully_connected_s4_wrapper'
    }
    
    return kernel_map[key]


def discover_descriptors(descriptors_dir: str) -> List[str]:
    """
    Discover all YAML descriptors in directory.
    
    Args:
        descriptors_dir: Directory containing YAML descriptors
        
    Returns:
        List of descriptor file paths
    """
    descriptors = []
    for root, dirs, files in os.walk(descriptors_dir):
        # Skip examples directory to avoid duplicates
        if 'examples' in root:
            continue
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                descriptors.append(os.path.join(root, file))
    return sorted(descriptors)


def expand_descriptor_variations(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a descriptor with variations into multiple individual descriptors.
    
    Args:
        desc: Original descriptor dictionary
        
    Returns:
        List of expanded descriptor dictionaries
    """
    # If no variations, return the original descriptor
    if 'variations' not in desc:
        return [desc]
    
    expanded_descriptors = []
    base_desc = {k: v for k, v in desc.items() if k != 'variations'}
    base_name = base_desc['name']  # Preserve original base name
    
    for variation in desc['variations']:
        # Create a new descriptor for this variation
        variation_desc = base_desc.copy()
        
        # Override with variation-specific fields
        for key, value in variation.items():
            variation_desc[key] = value
        if 'activation_dtype' in variation_desc:
            variation_desc['activation_dtype'] = str(variation_desc['activation_dtype']).upper()
        if 'weight_dtype' in variation_desc:
            variation_desc['weight_dtype'] = str(variation_desc['weight_dtype']).upper()
        
        # Use the variation name directly (shorter, cleaner)
        if 'name' in variation:
            variation_desc['name'] = variation['name']
        else:
            # Generate a name based on variation index
            variation_desc['name'] = f"{base_desc['name']}_var_{len(expanded_descriptors)}"
        
        # Store base descriptor name for filtering purposes
        variation_desc['_base_name'] = base_name
        
        # Validate the expanded variation has required shapes
        operator = variation_desc['operator']
        if operator == 'FullyConnected':
            if 'input_shape' not in variation_desc or 'filter_shape' not in variation_desc:
                raise ValueError("FullyConnected variation requires input_shape and filter_shape")
        elif operator in ['Conv2D', 'DepthwiseConv2D']:
            if 'input_shape' not in variation_desc or 'filter_shape' not in variation_desc:
                raise ValueError(f"{operator} variation requires input_shape and filter_shape")
        elif operator in ['MatMul', 'Elementwise', 'Add', 'Sub', 'Mul', 'Maximum', 'Minimum',
                          'Equal', 'NotEqual', 'Greater', 'GreaterEqual', 'Less', 'LessEqual']:
            if 'input_1_shape' not in variation_desc or 'input_2_shape' not in variation_desc:
                raise ValueError(f"{operator} variation requires input_1_shape and input_2_shape")
        elif operator == 'Pooling':
            if 'input_shape' not in variation_desc or ('pool_size' not in variation_desc and 'filter_shape' not in variation_desc):
                raise ValueError(f"{operator} variation requires input_shape and pool_size (or filter_shape)")
        elif operator in ['Relu', 'Relu6', 'LeakyRelu', 'Softmax', 'Quantize', 'Dequantize',
                         'Transpose', 'StridedSlice', 'Pad', 'LSTM', 'SVDF',
                         'Mean', 'ReduceMax', 'ReduceMin', 'ArgMax', 'ArgMin', 'TransposeConv',
                         'Tanh', 'Logistic', 'HardSwish', 'PReLU', 'Fill', 'ZerosLike',
                         'Reshape', 'Shape', 'Slice', 'Squeeze', 'SpaceToDepth', 'DepthToSpace',
                         'Split', 'Pack', 'Unpack', 'Concatenation', 'SpaceToBatchND', 'BatchToSpaceND', 'VariableUpdate']:
            if 'input_shape' not in variation_desc:
                raise ValueError(f"{operator} variation requires input_shape")
        
        # Normalize shapes (ensure they are lists)
        for shape_key in ['input_shape', 'filter_shape', 'input_1_shape', 'input_2_shape', 'pool_size']:
            if shape_key in variation_desc:
                variation_desc[shape_key] = list(variation_desc[shape_key])
        
        expanded_descriptors.append(variation_desc)
    
    return expanded_descriptors


def load_all_descriptors(descriptors_dir: str) -> List[Dict[str, Any]]:
    """
    Load and validate all descriptors in directory.
    Supports multiple descriptors per YAML file (separated by ---).
    Preserves original names from YAML descriptors when present.
    Falls back to numbered names (filename_1, filename_2, etc.) only when
    no name is specified in the descriptor.
    
    Args:
        descriptors_dir: Directory containing YAML descriptors
        
    Returns:
        List of validated descriptor dictionaries (expanded from variations)
    """
    descriptors = []
    desc_paths = discover_descriptors(descriptors_dir)
    
    for desc_path in desc_paths:
        try:
            # load_descriptor now returns a list (supports multiple docs per file)
            descs = load_descriptor(desc_path)
            
            # Get base filename without extension for numbering (fallback only)
            file_base = Path(desc_path).stem
            
            # If multiple descriptors from same file, preserve original names when available
            if len(descs) > 1:
                # Expand variations into individual descriptors for each descriptor
                for idx, desc in enumerate(descs, start=1):
                    # Create a copy
                    desc_copy = desc.copy()
                    
                    # Preserve original name if it exists, otherwise use numbered name
                    if 'name' not in desc_copy or not desc_copy['name']:
                        # No name specified, use numbered name as fallback
                        desc_copy['name'] = f"{file_base}_{idx}"
                    # If name exists, keep it as-is (preserve original meaningful names)
                    
                    expanded_descs = expand_descriptor_variations(desc_copy)
                    descriptors.extend(expanded_descs)
            else:
                # Single descriptor - preserve original name if present
                for desc in descs:
                    # If no name specified, use file-based name as fallback
                    if 'name' not in desc or not desc['name']:
                        desc['name'] = file_base
                    # Otherwise, keep the original name
                    
                    expanded_descs = expand_descriptor_variations(desc)
                    descriptors.extend(expanded_descs)
        except Exception as e:
            print(f"Warning: Failed to load descriptor {desc_path}: {e}")
            continue
            
    return descriptors


def get_io_dtypes(desc: Dict[str, Any]) -> Dict[str, str]:
    """
    Get I/O dtype mapping from descriptor.
    
    Args:
        desc: YAML descriptor dictionary
        
    Returns:
        Dictionary mapping I/O type names to C types
    """
    activation_dtype = desc['activation_dtype']
    weight_dtype = desc['weight_dtype']
    
    # Determine output dtype
    output_dtype = 'S8'  # Default
    if activation_dtype == 'S16':
        output_dtype = 'S16'
    elif weight_dtype == 'S4':
        output_dtype = 'S8'  # S4 weights typically produce S8 output
        
    # Override from hint if specified
    if 'hint' in desc and 'extras' in desc['hint']:
        if 'output_dtype' in desc['hint']['extras']:
            output_dtype = desc['hint']['extras']['output_dtype']
            
    # Map to C types
    ctype_map = {
        'S8': 'int8_t',
        'S16': 'int16_t',
        'S4': 'int8_t'  # S4 weights are packed into int8_t
    }
    
    # Special cases for Quantize/Dequantize operations
    operator = desc.get('operator', '')
    if operator == 'Quantize':
        act_type = 'float'  # Input is float32 for Quantize
    else:
        act_type = ctype_map[activation_dtype]
    
    # Special case: Dequantize operations have float32 output
    if operator == 'Dequantize':
        out_type = 'float'  # Output is float32 for Dequantize
    else:
        out_type = ctype_map[output_dtype]
    
    return {
        'ACT_T': act_type,
        'W_T': ctype_map[weight_dtype],
        'OUT_T': out_type
    }
