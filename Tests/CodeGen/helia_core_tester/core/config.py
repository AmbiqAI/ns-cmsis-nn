"""
Configuration management for CMSIS-NN Tools.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from helia_core_tester.core.discovery import find_repo_root
from helia_core_tester.core.errors import ConfigurationError, PathNotFoundError

PATH_KEYS = frozenset({
    "project_root", "downloads_dir", "generated_tests_dir",
    "generation_dir", "report_dir",
})


def _discover_generation_dir(project_root: Path) -> Path:
    """Return generation directory (contains test_ops.py, ops/, templates/)."""
    p = project_root / "helia_core_tester" / "generation"
    if (p / "test_ops.py").exists():
        return p
    if p.exists():
        return p
    return p


@dataclass
class Config:
    """Configuration class for CMSIS-NN Tools."""
    
    # Project paths - will be set in __post_init__ using discovery
    project_root: Optional[Path] = None
    config_file: Optional[Path] = None
    downloads_dir: Optional[Path] = None
    generated_tests_dir: Optional[Path] = None
    generation_dir: Optional[Path] = None
    
    # Build configuration
    cpu: str = "cortex-m55"
    optimization: str = "-Ofast"
    jobs: Optional[int] = None
    
    # Test configuration
    timeout: float = 0.0
    fail_fast: bool = True
    verbosity: int = 0  # 0=minimal, 1=progress, 2=commands, 3=debug
    dry_run: bool = False
    plan: bool = False
    
    # Filters
    op_filter: Optional[str] = None
    dtype_filter: Optional[str] = None
    name_filter: Optional[str] = None
    limit: Optional[int] = None
    seed: Optional[int] = 500  # Random seed for test generation (None = use hash of test name)
    
    # Skip options
    skip_generation: bool = False
    skip_conversion: bool = False
    skip_runners: bool = False
    skip_build: bool = False
    skip_run: bool = False

    # Cleanup options
    clean_generated_tests: bool = True
    clean_build: bool = True
    
    # Reporting configuration
    enable_reporting: bool = True
    report_formats: list = field(default_factory=lambda: ["json"])
    report_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Discover repository root if not provided
        if self.project_root is None:
            try:
                self.project_root = find_repo_root()
            except Exception as e:
                raise ConfigurationError(
                    f"Could not discover repository root: {e}. "
                    f"Set project_root explicitly or set CMSIS_NN_REPO_ROOT environment variable."
                ) from e
        else:
            self.project_root = Path(self.project_root).resolve()

        # Load defaults from config file (if present)
        self._load_config_file()
        
        # Set derived paths if not provided
        if self.downloads_dir is None:
            self.downloads_dir = self.project_root / "artifacts" / "downloads"
        else:
            self.downloads_dir = Path(self.downloads_dir).resolve()
        
        if self.generated_tests_dir is None:
            self.generated_tests_dir = self.project_root / "artifacts" / "generated_tests"
        else:
            self.generated_tests_dir = Path(self.generated_tests_dir).resolve()
        
        if self.generation_dir is None:
            self.generation_dir = _discover_generation_dir(self.project_root)
        else:
            self.generation_dir = Path(self.generation_dir).resolve()
        
        # Convert report_dir to Path if needed
        if self.report_dir is None:
            # Default to build directory reports under artifacts
            build_dir = self.project_root / "artifacts" / f"build-{self.cpu}-gcc"
            self.report_dir = build_dir / "reports"
        elif isinstance(self.report_dir, str):
            self.report_dir = Path(self.report_dir)
        else:
            self.report_dir = Path(self.report_dir).resolve()
        
        # Validate verbosity level
        if not 0 <= self.verbosity <= 3:
            raise ValueError(f"verbosity must be between 0 and 3, got {self.verbosity}")
        
        # Set default jobs to CPU count
        if self.jobs is None:
            self.jobs = os.cpu_count() or 4
        
        # Validate paths exist (warn but don't fail for generated directories)
        if not self.project_root.exists():
            raise PathNotFoundError(f"Project root does not exist: {self.project_root}")
        
        # Downloads dir will be created if needed; ensure parent exists
        if not self.downloads_dir.parent.exists():
            self.downloads_dir.parent.mkdir(parents=True, exist_ok=True)

    def _load_config_file(self) -> None:
        """Load defaults from helia_core_tester.toml if present."""
        env_path = os.environ.get("HELIA_CORE_TESTER_CONFIG")
        if env_path:
            self.config_file = Path(env_path).resolve()
        elif self.config_file is None:
            self.config_file = self.project_root / "helia_core_tester.toml"

        if not self.config_file or not self.config_file.exists():
            return

        try:
            try:
                import tomllib  # Python 3.11+
            except ModuleNotFoundError:  # Python 3.8-3.10
                import tomli as tomllib

            data = tomllib.loads(self.config_file.read_text())
            table = data.get("helia_core_tester") or data.get("tool", {}).get("helia_core_tester", {})
            if not isinstance(table, dict):
                return
        except Exception as e:
            raise ConfigurationError(f"Failed to read config file: {self.config_file}: {e}") from e

        for key, value in table.items():
            if not hasattr(self, key):
                continue
            if value is None:
                continue
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_root": str(self.project_root),
            "downloads_dir": str(self.downloads_dir),
            "generated_tests_dir": str(self.generated_tests_dir),
            "generation_dir": str(self.generation_dir),
            "cpu": self.cpu,
            "optimization": self.optimization,
            "jobs": self.jobs,
            "timeout": self.timeout,
            "fail_fast": self.fail_fast,
            "verbosity": self.verbosity,
            "dry_run": self.dry_run,
            "plan": self.plan,
            "op_filter": self.op_filter,
            "dtype_filter": self.dtype_filter,
            "name_filter": self.name_filter,
            "limit": self.limit,
            "seed": self.seed,
            "skip_generation": self.skip_generation,
            "skip_conversion": self.skip_conversion,
            "skip_runners": self.skip_runners,
            "skip_build": self.skip_build,
            "skip_run": self.skip_run,
            "clean_generated_tests": self.clean_generated_tests,
            "clean_build": self.clean_build,
            "enable_reporting": self.enable_reporting,
            "report_formats": self.report_formats,
            "report_dir": str(self.report_dir),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        data = dict(data)
        for key in PATH_KEYS:
            if key in data and isinstance(data[key], str):
                data[key] = Path(data[key])
        return cls(**data)
