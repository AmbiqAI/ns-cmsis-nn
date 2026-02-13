"""
Main TFLite model generation for Helia-Core Tester.
Thin generator that discovers YAML descriptors and generates TFLite models.
"""

import pytest
import os
import tempfile
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from cmsis_nn_tools.core.discovery import find_descriptors_dir, find_generated_tests_dir
from cmsis_nn_tools.generation.tflite_generator.tester.io.descriptors import load_all_descriptors
from cmsis_nn_tools.generation.tflite_generator.tester.ops.fully_connected import OpFullyConnected
from cmsis_nn_tools.generation.tflite_generator.tester.ops.conv2d import OpConv2D
from cmsis_nn_tools.generation.tflite_generator.tester.ops.dwconv import OpDepthwiseConv2D
from cmsis_nn_tools.generation.tflite_generator.tester.ops.matmul_batch import OpMatMul
from cmsis_nn_tools.generation.tflite_generator.tester.ops.elementwise import OpElementwise
from cmsis_nn_tools.generation.tflite_generator.tester.ops.add import OpAdd
from cmsis_nn_tools.generation.tflite_generator.tester.ops.sub import OpSub
from cmsis_nn_tools.generation.tflite_generator.tester.ops.mul import OpMul
from cmsis_nn_tools.generation.tflite_generator.tester.ops.minmax import OpMinMax
from cmsis_nn_tools.generation.tflite_generator.tester.ops.relu import OpRelu
from cmsis_nn_tools.generation.tflite_generator.tester.ops.relu6 import OpRelu6
from cmsis_nn_tools.generation.tflite_generator.tester.ops.leakyrelu import OpLeakyRelu
from cmsis_nn_tools.generation.tflite_generator.tester.ops.softmax import OpSoftmax
from cmsis_nn_tools.generation.tflite_generator.tester.ops.tanh import OpTanh
from cmsis_nn_tools.generation.tflite_generator.tester.ops.logistic import OpLogistic
from cmsis_nn_tools.generation.tflite_generator.tester.ops.hard_swish import OpHardSwish
from cmsis_nn_tools.generation.tflite_generator.tester.ops.prelu import OpPReLU
from cmsis_nn_tools.generation.tflite_generator.tester.ops.quantize import OpQuantize
from cmsis_nn_tools.generation.tflite_generator.tester.ops.dequantize import OpDequantize
from cmsis_nn_tools.generation.tflite_generator.tester.ops.pooling import OpPooling
from cmsis_nn_tools.generation.tflite_generator.tester.ops.transpose import OpTranspose
from cmsis_nn_tools.generation.tflite_generator.tester.ops.stridedslice import OpStridedSlice
from cmsis_nn_tools.generation.tflite_generator.tester.ops.pad import OpPad
from cmsis_nn_tools.generation.tflite_generator.tester.ops.lstm import OpLSTM
from cmsis_nn_tools.generation.tflite_generator.tester.ops.svdf import OpSVDF
from cmsis_nn_tools.generation.tflite_generator.tester.ops.mean import OpMean
from cmsis_nn_tools.generation.tflite_generator.tester.ops.reducemax import OpReduceMax
from cmsis_nn_tools.generation.tflite_generator.tester.ops.reducemin import OpReduceMin
from cmsis_nn_tools.generation.tflite_generator.tester.ops.argmax import OpArgMax
from cmsis_nn_tools.generation.tflite_generator.tester.ops.argmin import OpArgMin
from cmsis_nn_tools.generation.tflite_generator.tester.ops.fill import OpFill
from cmsis_nn_tools.generation.tflite_generator.tester.ops.zeros_like import OpZerosLike
from cmsis_nn_tools.generation.tflite_generator.tester.ops.reshape import OpReshape
from cmsis_nn_tools.generation.tflite_generator.tester.ops.shape import OpShape
from cmsis_nn_tools.generation.tflite_generator.tester.ops.slice import OpSlice
from cmsis_nn_tools.generation.tflite_generator.tester.ops.squeeze import OpSqueeze
from cmsis_nn_tools.generation.tflite_generator.tester.ops.space_to_depth import OpSpaceToDepth
from cmsis_nn_tools.generation.tflite_generator.tester.ops.depth_to_space import OpDepthToSpace
from cmsis_nn_tools.generation.tflite_generator.tester.ops.split import OpSplit
from cmsis_nn_tools.generation.tflite_generator.tester.ops.pack import OpPack
from cmsis_nn_tools.generation.tflite_generator.tester.ops.unpack import OpUnpack
from cmsis_nn_tools.generation.tflite_generator.tester.ops.concatenation import OpConcatenation
from cmsis_nn_tools.generation.tflite_generator.tester.ops.space_to_batch_nd import OpSpaceToBatchND
from cmsis_nn_tools.generation.tflite_generator.tester.ops.batch_to_space_nd import OpBatchToSpaceND
from cmsis_nn_tools.generation.tflite_generator.tester.ops.variable_update import OpVariableUpdate
from cmsis_nn_tools.generation.tflite_generator.tester.ops.transposeconv import OpTransposeConv
from cmsis_nn_tools.generation.tflite_generator.tester.ops.equal import OpEqual
from cmsis_nn_tools.generation.tflite_generator.tester.ops.not_equal import OpNotEqual
from cmsis_nn_tools.generation.tflite_generator.tester.ops.greater import OpGreater
from cmsis_nn_tools.generation.tflite_generator.tester.ops.greater_equal import OpGreaterEqual
from cmsis_nn_tools.generation.tflite_generator.tester.ops.less import OpLess
from cmsis_nn_tools.generation.tflite_generator.tester.ops.less_equal import OpLessEqual


# Operation mapping
OP_MAP = {
    'FullyConnected': OpFullyConnected,
    'Conv2D': OpConv2D,
    'DepthwiseConv2D': OpDepthwiseConv2D,
    'MatMul': OpMatMul,
    'Elementwise': OpElementwise,
    'Add': OpAdd,
    'Sub': OpSub,
    'Mul': OpMul,
    'Maximum': OpMinMax,
    'Minimum': OpMinMax,
    'Relu': OpRelu,
    'Relu6': OpRelu6,
    'LeakyRelu': OpLeakyRelu,
    'Softmax': OpSoftmax,
    'Tanh': OpTanh,
    'Logistic': OpLogistic,
    'HardSwish': OpHardSwish,
    'PReLU': OpPReLU,
    'Quantize': OpQuantize,
    'Dequantize': OpDequantize,
    'Pooling': OpPooling,
    'Transpose': OpTranspose,
    'StridedSlice': OpStridedSlice,
    'Pad': OpPad,
    'LSTM': OpLSTM,
    'SVDF': OpSVDF,
    'Mean': OpMean,
    'ReduceMax': OpReduceMax,
    'ReduceMin': OpReduceMin,
    'ArgMax': OpArgMax,
    'ArgMin': OpArgMin,
    'Fill': OpFill,
    'ZerosLike': OpZerosLike,
    'Reshape': OpReshape,
    'Shape': OpShape,
    'Slice': OpSlice,
    'Squeeze': OpSqueeze,
    'SpaceToDepth': OpSpaceToDepth,
    'DepthToSpace': OpDepthToSpace,
    'Split': OpSplit,
    'Pack': OpPack,
    'Unpack': OpUnpack,
    'Concatenation': OpConcatenation,
    'SpaceToBatchND': OpSpaceToBatchND,
    'BatchToSpaceND': OpBatchToSpaceND,
    'VariableUpdate': OpVariableUpdate,
    'TransposeConv': OpTransposeConv,
    'Equal': OpEqual,
    'NotEqual': OpNotEqual,
    'Greater': OpGreater,
    'GreaterEqual': OpGreaterEqual,
    'Less': OpLess,
    'LessEqual': OpLessEqual,
}


def should_run_test(desc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Determine if test should run based on filters.
    
    Args:
        desc: Test descriptor
        filters: Filter dictionary from command line
        
    Returns:
        True if test should run
    """
    if filters.get('name'):
        if desc['name'] != filters['name']:
            return False

    if filters.get('op'):
        filter_op = filters['op']
        desc_name = desc['name']
        base_name = desc.get('_base_name', None)
        source_file = desc.get('_source_file', None)
        desc_operator = desc.get('operator', None)
        
        name_matches = desc_name == filter_op or desc_name.startswith(filter_op + '_')
        base_matches = base_name == filter_op if base_name else False
        file_matches = source_file == filter_op if source_file else False
        operator_matches = desc_operator == filter_op if desc_operator else False
        
        if not name_matches and not base_matches and not file_matches and not operator_matches:
            return False
        
    # Filter by activation dtype
    if filters.get('dtype') and desc['activation_dtype'] != filters['dtype']:
        return False
        
    # Filter by weight dtype
    if filters.get('wtype') and desc['weight_dtype'] != filters['wtype']:
        return False
            
    return True


def generate_test(desc: Dict[str, Any], out_dir: str, seed: Optional[int] = None) -> None:
    """
    Generate TFLite model for a descriptor.
    
    Args:
        desc: YAML descriptor
        out_dir: Output directory for generated files
        seed: Optional random seed (if None, uses hash of test name)
    """
    name = desc['name']
    operator = desc['operator']
    
    # Create output directory
    test_dir = Path(out_dir) / name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the complete descriptor as YAML in the test directory
    descriptor_path = test_dir / "descriptor.yaml"
    with open(descriptor_path, 'w') as f:
        yaml.dump(desc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Get operation class
    if operator not in OP_MAP:
        raise ValueError(f"Unsupported operator: {operator}")
        
    op_class = OP_MAP[operator]
    
    # Initialize operation with deterministic seed
    if seed is None:
        seed = hash(name) % (2**32)  # Deterministic seed from name
    op = op_class(desc, seed)
    
    # Build Keras model
    model = op.build_keras_model()
    
    # Convert to TFLite
    tflite_path = test_dir / f"{name}.tflite"
    op.convert_to_tflite(model, str(tflite_path), seed)
    
    print(f"Generated TFLite model: {name}")
    
    # Generate C/H files from templates
    try:
        op.generate_c_files(test_dir)
    except NotImplementedError:
        # Operator doesn't support C file generation yet
        print(f"INFO: {name} - C file generation not implemented")
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to generate C/H files for {name}: {e}")
        print(f"ERROR: Traceback:")
        traceback.print_exc()
        # Continue anyway - C generation is optional during transition


def test_generation(test_filters):
    """
    Generate TFLite models for all descriptors.
    """
    # Load all descriptors using discovery
    descriptors_dir = find_descriptors_dir()
    descriptors = load_all_descriptors(str(descriptors_dir))
    
    # Apply filters
    filtered_descriptors = []
    for desc in descriptors:
        if should_run_test(desc, test_filters):
            filtered_descriptors.append(desc)
            
    # Apply limit
    if test_filters.get('limit'):
        filtered_descriptors = filtered_descriptors[:test_filters['limit']]
        
    # Generate TFLite models for each descriptor
    # Place models in repo-top GeneratedTests
    generated_count = 0
    for desc in filtered_descriptors:
        try:
            top_generated = find_generated_tests_dir(create=True)
            generate_test(desc, str(top_generated), seed=test_filters.get('seed'))
            generated_count += 1
        except Exception as e:
            print(f"Failed to generate TFLite model for {desc['name']}: {e}")
            # Continue with other models
            continue
            
    print(f"Successfully generated {generated_count} TFLite models")
    assert generated_count > 0, "No TFLite models were generated"


def test_generated_files_exist():
    """
    Verify that generated TFLite files exist and are valid.
    This should run AFTER test_generation().
    """
    # Don't generate, just validate what test_generation() created
    generated_tests_dir = Path("../GeneratedTests")
    if not generated_tests_dir.exists():
        pytest.skip("No generated tests found")
        
    # Check that we have some generated tests
    test_dirs = [d for d in generated_tests_dir.iterdir() if d.is_dir()]
    assert len(test_dirs) > 0, "No test directories found"
    
    # Check that each test has TFLite file
    for test_dir in test_dirs:
        name = test_dir.name
        tflite_file = test_dir / f"{name}.tflite"
        
        assert tflite_file.exists(), f"Missing {name}.tflite"
        
        # Check that file is not empty
        assert tflite_file.stat().st_size > 0, f"{name}.tflite is empty"