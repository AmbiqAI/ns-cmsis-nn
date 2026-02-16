"""
Main TFLite model generation for Helia-Core Tester.
Thin generator that discovers YAML descriptors and generates TFLite models.
"""

import json
import hashlib
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from helia_core_tester.core.discovery import find_descriptors_dir, find_generated_tests_dir, find_repo_root
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops import OP_MAP


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
        # Stable deterministic seed from name (independent of PYTHONHASHSEED)
        seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    op = op_class(desc, seed)
    
    # Build Keras model (skip for ops that generate LiteRT models directly)
    if operator in {"ArgMax", "ArgMin"}:
        model = None
    else:
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
    # Place models in artifacts/generated_tests
    generated_count = 0
    manifest_entries: List[Dict[str, Any]] = []
    for desc in filtered_descriptors:
        try:
            top_generated = find_generated_tests_dir(create=True)
            generate_test(desc, str(top_generated), seed=test_filters.get('seed'))
            test_dir = Path(top_generated) / desc["name"]
            tflite_path = test_dir / f"{desc['name']}.tflite"
            c_sources = sorted([str(p.name) for p in test_dir.glob("*.c")])
            manifest_entries.append({
                "name": desc.get("name"),
                "operator": desc.get("operator"),
                "activation_dtype": desc.get("activation_dtype"),
                "weight_dtype": desc.get("weight_dtype"),
                "path": str(test_dir),
                "tflite": str(tflite_path),
                "c_sources": c_sources,
            })
            generated_count += 1
        except Exception as e:
            print(f"Failed to generate TFLite model for {desc['name']}: {e}")
            # Continue with other models
            continue
            
    print(f"Successfully generated {generated_count} TFLite models")
    if generated_count > 0:
        _write_manifest_and_cmake(manifest_entries, test_filters)
    assert generated_count > 0, "No TFLite models were generated"


def _write_manifest_and_cmake(entries: List[Dict[str, Any]], test_filters: Dict[str, Any]) -> None:
    generated_tests_dir = find_generated_tests_dir(create=True)
    repo_root = find_repo_root()

    manifest = {
        "generated_count": len(entries),
        "filters": {
            "op": test_filters.get("op"),
            "dtype": test_filters.get("dtype"),
            "wtype": test_filters.get("wtype"),
            "name": test_filters.get("name"),
            "limit": test_filters.get("limit"),
            "seed": test_filters.get("seed"),
        },
        "tests": entries,
    }
    manifest_path = generated_tests_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    rel_root = generated_tests_dir.relative_to(repo_root)
    test_dirs = sorted({str(Path(rel_root) / Path(e["name"])) for e in entries})
    cmake_lines = ["set(GENERATED_TEST_DIRS"]
    for d in test_dirs:
        cmake_lines.append(f"  \"{d}\"")
    cmake_lines.append(")")
    cmake_path = generated_tests_dir / "tests.cmake"
    cmake_path.write_text("\n".join(cmake_lines) + "\n")


def test_generated_files_exist():
    """
    Verify that generated TFLite files exist and are valid.
    This should run AFTER test_generation().
    """
    # Don't generate, just validate what test_generation() created
    generated_tests_dir = find_generated_tests_dir(create=False)
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
