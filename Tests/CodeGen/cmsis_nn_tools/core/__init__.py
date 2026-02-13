"""
Core modules for CMSIS-NN Tools.
"""

from cmsis_nn_tools.core.config import Config
from cmsis_nn_tools.core.discovery import (
    find_repo_root,
    find_repo_root_or_cwd,
    find_descriptors_dir,
    find_generated_tests_dir,
    find_tester_templates_dir,
    find_fvp_script_path,
    find_setup_dependencies_script,
)
from cmsis_nn_tools.core.errors import (
    CMSISNNToolsError,
    RepoRootNotFoundError,
    ConfigurationError,
    PathNotFoundError,
    StepExecutionError,
    BuildError,
    FVPRunError,
    GenerationError,
)

__all__ = [
    "Config",
    "find_repo_root",
    "find_repo_root_or_cwd",
    "find_descriptors_dir",
    "find_generated_tests_dir",
    "find_tester_templates_dir",
    "find_fvp_script_path",
    "find_setup_dependencies_script",
    "CMSISNNToolsError",
    "RepoRootNotFoundError",
    "ConfigurationError",
    "PathNotFoundError",
    "StepExecutionError",
    "BuildError",
    "FVPRunError",
    "GenerationError",
]
