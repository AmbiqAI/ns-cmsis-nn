"""
CMake build step for FVP.
"""

import subprocess
import sys
from pathlib import Path

from helia_core_tester.core.steps.base import StepBase, StepResult, StepStatus
from helia_core_tester.core.errors import BuildError, PathNotFoundError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.discovery import find_fvp_script_path
from helia_core_tester.utils.command_runner import run_command


class BuildStep(StepBase):
    """Step for building FVP executables."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "build"
    
    def should_skip(self) -> bool:
        """Check if build should be skipped."""
        return self.config.skip_build
    
    def validate(self) -> str | None:
        """Validate prerequisites for build."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        return None
    
    def execute(self) -> StepResult:
        """Execute CMake build for FVP."""
        if self.should_skip():
            return StepResult(
                status=StepStatus.SKIPPED,
                message="Build skipped (--skip-build)"
            )
        
        if self.config.dry_run:
            return self.dry_run()
        
        error = self.validate()
        if error:
            return StepResult(
                status=StepStatus.FAILED,
                message=error,
                error=PathNotFoundError(error)
            )
        
        if self.config.verbosity >= 1:
            self.logger.info(f"Building for FVP ({self.config.cpu})")
        
        script_path = find_fvp_script_path(self.config.project_root)
        
        cmd = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        
        if self.config.jobs:
            cmd.extend(["--jobs", str(self.config.jobs)])
        
        # Map verbosity: 0-1 = quiet, 2-3 = verbose
        if self.config.verbosity <= 1:
            cmd.append("--quiet")
        else:
            cmd.extend(["--verbosity", str(self.config.verbosity)])
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            
            run_command(cmd, cwd=self.config.project_root, verbosity=self.config.verbosity)
            
            if self.config.verbosity >= 1:
                self.logger.info(f"Successfully built for {self.config.cpu}")
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message=f"Successfully built for {self.config.cpu}"
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to build for FVP (exit code {e.returncode})"
            self.logger.error(error_msg)
            
            # Try to capture more error details
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.stdout:
                    self.logger.error(f"stdout: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"stderr: {result.stderr}")
            except Exception:
                pass
            
            return StepResult(
                status=StepStatus.FAILED,
                message=error_msg,
                error=BuildError(error_msg)
            )
        except FileNotFoundError as e:
            error_msg = f"Failed to build for FVP: {e}"
            self.logger.error(error_msg)
            build_error = BuildError(error_msg)
            build_error.__cause__ = e
            return StepResult(
                status=StepStatus.FAILED,
                message=error_msg,
                error=build_error
            )
    
    def dry_run(self) -> StepResult:
        """Dry run of build step."""
        cmd_preview = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        if self.config.jobs:
            cmd_preview.extend(["--jobs", str(self.config.jobs)])
        
        return StepResult(
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: {' '.join(cmd_preview)}"
        )
