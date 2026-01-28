"""
Pytest configuration and fixtures for Helia-Core Tester.
"""

import pytest
import os
import shutil
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--op", action="store", default=None,
                    help="Filter by operator (e.g., FullyConnected)")
    parser.addoption("--dtype", action="store", default=None,
                    help="Filter by activation dtype (S8, S16)")
    parser.addoption("--wtype", action="store", default=None,
                    help="Filter by weight dtype (S8, S4)")
    parser.addoption("--name", action="store", default=None,
                    help="Filter by exact test name")
    parser.addoption("--limit", action="store", type=int, default=None,
                    help="Limit number of tests to run")


def pytest_configure(config):
    """Configure pytest with custom options."""
    # Clean GeneratedTests directory before running (in UnitTest directory)
    generated_tests_dir = Path(__file__).resolve().parents[2] / "GeneratedTests"
    
    if generated_tests_dir.exists():
        print(f"\nCleaning existing GeneratedTests directory...")
        try:
            # Count existing files before deletion
            existing_count = sum(1 for _ in generated_tests_dir.rglob("*.tflite"))
            if existing_count > 0:
                print(f"   Removing {existing_count} existing TFLite model(s)")
            
            shutil.rmtree(generated_tests_dir)
            print(f"Directory cleaned")
        except OSError as e:
            print(f"Warning: Could not remove entire directory, trying individual files...")
            # If rmtree fails, try to remove individual files
            for item in generated_tests_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"   Individual files removed")
    
    # Create fresh directory
    generated_tests_dir.mkdir(exist_ok=True)
    print(f"📁 Created fresh GeneratedTests directory\n")


@pytest.fixture
def test_filters(request):
    """Provide test filters from command line options."""
    return {
        'op': request.config.getoption("--op"),
        'dtype': request.config.getoption("--dtype"),
        'wtype': request.config.getoption("--wtype"),
        'name': request.config.getoption("--name"),
        'limit': request.config.getoption("--limit")
    }
