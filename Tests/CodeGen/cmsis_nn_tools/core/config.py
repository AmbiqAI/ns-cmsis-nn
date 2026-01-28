"""
Configuration management for Helia-Core Tester.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for Helia-Core Tester."""
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parents[2])
    downloads_dir: Path = field(default_factory=lambda: Path(__file__).parents[2] / "downloads")
    generated_tests_dir: Path = field(default_factory=lambda: Path(__file__).parents[2] / "GeneratedTests")
    tflite_generator_dir: Path = field(default_factory=lambda: Path(__file__).parents[1] / "tflite_generator")
    
    # Build configuration
    cpu: str = "cortex-m55"
    optimization: str = "-Ofast"
    jobs: Optional[int] = None
    
    # Test configuration
    timeout: float = 0.0
    fail_fast: bool = True
    verbosity: int = 0  # 0=minimal, 1=progress, 2=commands, 3=debug
    dry_run: bool = False
    
    # Filters
    op_filter: Optional[str] = None
    dtype_filter: Optional[str] = None
    name_filter: Optional[str] = None
    limit: Optional[int] = None
    
    # Skip options
    skip_generation: bool = False
    skip_conversion: bool = False
    skip_runners: bool = False
    skip_build: bool = False
    skip_run: bool = False
    
    # Reporting configuration
    enable_reporting: bool = True
    report_formats: list = field(default_factory=lambda: ["json"])
    report_dir: Path = field(default_factory=lambda: Path("reports"))
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects if needed
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.downloads_dir, str):
            self.downloads_dir = Path(self.downloads_dir)
        if isinstance(self.generated_tests_dir, str):
            self.generated_tests_dir = Path(self.generated_tests_dir)
        if isinstance(self.tflite_generator_dir, str):
            self.tflite_generator_dir = Path(self.tflite_generator_dir)
        
        # Validate verbosity level
        if not 0 <= self.verbosity <= 3:
            raise ValueError(f"verbosity must be between 0 and 3, got {self.verbosity}")
        
        # Set default jobs to CPU count
        if self.jobs is None:
            self.jobs = os.cpu_count() or 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_root": str(self.project_root),
            "downloads_dir": str(self.downloads_dir),
            "generated_tests_dir": str(self.generated_tests_dir),
            "tflite_generator_dir": str(self.tflite_generator_dir),
            "cpu": self.cpu,
            "optimization": self.optimization,
            "jobs": self.jobs,
            "timeout": self.timeout,
            "fail_fast": self.fail_fast,
            "verbosity": self.verbosity,
            "dry_run": self.dry_run,
            "op_filter": self.op_filter,
            "dtype_filter": self.dtype_filter,
            "name_filter": self.name_filter,
            "limit": self.limit,
            "skip_generation": self.skip_generation,
            "skip_conversion": self.skip_conversion,
            "skip_runners": self.skip_runners,
            "skip_build": self.skip_build,
            "skip_run": self.skip_run,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**data)
