"""
Repository root discovery utilities.

This module provides robust repository root discovery that works from any
directory within the repository, without relying on __file__ paths.
"""

import os
from pathlib import Path
from typing import Optional

from cmsis_nn_tools.core.errors import RepoRootNotFoundError


# Marker files/directories that indicate the repository root
REPO_MARKERS = [
    "pyproject.toml",  # Tests/CodeGen/pyproject.toml
    "CMakeLists.txt",  # Tests/CodeGen/CMakeLists.txt
    ".git",            # Root .git directory
]


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the repository root by walking up from start_path looking for markers.
    
    The repository root is identified as Tests/CodeGen/ directory, which contains:
    - pyproject.toml
    - CMakeLists.txt
    
    Args:
        start_path: Starting directory (default: current working directory)
        
    Returns:
        Path to the repository root (Tests/CodeGen/)
        
    Raises:
        RepoRootNotFoundError: If repository root cannot be found
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    # Also check environment variable override
    env_root = os.environ.get("CMSIS_NN_REPO_ROOT")
    if env_root:
        env_path = Path(env_root).resolve()
        if _is_repo_root(env_path):
            return env_path
        raise RepoRootNotFoundError(
            f"Environment variable CMSIS_NN_REPO_ROOT points to invalid location: {env_root}"
        )
    
    # Walk up from start_path
    current = start_path.resolve()
    visited = set()
    
    while current != current.parent:  # Stop at filesystem root
        if current in visited:
            break
        visited.add(current)
        
        if _is_repo_root(current):
            return current
        
        current = current.parent
    
    # If we didn't find it, try from the file's location as fallback
    # (but this should rarely be needed)
    raise RepoRootNotFoundError(
        f"Could not find repository root starting from {start_path}. "
        f"Looking for one of: {', '.join(REPO_MARKERS)}. "
        f"Set CMSIS_NN_REPO_ROOT environment variable to override."
    )


def _is_repo_root(path: Path) -> bool:
    """
    Check if a path is the repository root (Tests/CodeGen/).
    
    Args:
        path: Path to check
        
    Returns:
        True if path appears to be the repository root
    """
    if not path.is_dir():
        return False
    
    # Check for key markers
    has_pyproject = (path / "pyproject.toml").exists()
    has_cmake = (path / "CMakeLists.txt").exists()
    
    # Both markers must be present
    if has_pyproject and has_cmake:
        # Additional validation: check if it looks like Tests/CodeGen
        # by checking for expected subdirectories
        expected_dirs = ["descriptors", "GeneratedTests"]
        has_expected = any((path / d).exists() for d in expected_dirs)
        
        # If we have both markers, it's likely the root
        # (expected_dirs check is nice-to-have but not required)
        return True
    
    return False


def find_repo_root_or_cwd() -> Path:
    """
    Find repository root, or return current working directory if not found.
    
    This is a fallback function for cases where we want to be lenient.
    
    Returns:
        Repository root if found, otherwise current working directory
    """
    try:
        return find_repo_root()
    except RepoRootNotFoundError:
        return Path.cwd()


def find_descriptors_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Find the descriptors directory.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
    
    Returns:
        Path to descriptors/ directory at repo root
        
    Raises:
        RepoRootNotFoundError: If repo root cannot be found
        PathNotFoundError: If descriptors directory doesn't exist
    """
    if repo_root is None:
        repo_root = find_repo_root()
    
    descriptors_dir = repo_root / "descriptors"
    if not descriptors_dir.exists():
        from .errors import PathNotFoundError
        raise PathNotFoundError(f"Descriptors directory not found: {descriptors_dir}")
    
    return descriptors_dir


def find_generated_tests_dir(repo_root: Optional[Path] = None, create: bool = True) -> Path:
    """
    Find or create the GeneratedTests directory.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        create: Create directory if it doesn't exist
        
    Returns:
        Path to GeneratedTests/ directory at repo root
        
    Raises:
        RepoRootNotFoundError: If repo root cannot be found
    """
    if repo_root is None:
        repo_root = find_repo_root()
    
    generated_tests_dir = repo_root / "GeneratedTests"
    
    if create and not generated_tests_dir.exists():
        generated_tests_dir.mkdir(parents=True, exist_ok=True)
    
    return generated_tests_dir


def find_tester_templates_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Find the tester templates directory (Jinja2 templates for C/H generation).
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to tester/templates/ under cmsis_nn_tools/generation/tflite_generator/
    """
    if repo_root is None:
        repo_root = find_repo_root()
    
    return (
        repo_root
        / "cmsis_nn_tools"
        / "generation"
        / "tflite_generator"
        / "tester"
        / "templates"
    )


def find_fvp_script_path(repo_root: Optional[Path] = None) -> Path:
    """
    Find the FVP build-and-run script path.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to build_and_run_fvp.py under cmsis_nn_tools/fvp/
    """
    if repo_root is None:
        repo_root = find_repo_root()
    
    return repo_root / "cmsis_nn_tools" / "fvp" / "build_and_run_fvp.py"


def find_setup_dependencies_script(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the setup_dependencies.py script if it exists.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to setup_dependencies.py under cmsis_nn_tools/scripts/, or None if not found
    """
    if repo_root is None:
        repo_root = find_repo_root()
    
    script = repo_root / "cmsis_nn_tools" / "scripts" / "setup_dependencies.py"
    return script if script.exists() else None
