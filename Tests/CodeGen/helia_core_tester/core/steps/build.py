"""
CMake build step for FVP.
"""

import subprocess
import sys
from pathlib import Path

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import BuildError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.discovery import find_fvp_script_path, find_build_dir
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

    def _do_execute(self) -> StepResult:
        """Execute CMake build for FVP."""
        if self.config.verbosity >= 1:
            self.logger.info(f"Building for FVP ({self.config.cpu})")
        script_path = find_fvp_script_path(self.config.project_root)
        cmd = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        if not self.config.clean_build:
            cmd.append("--no-clean-build")
        
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
            
            run_command(
                cmd,
                cwd=self.config.project_root,
                verbosity=self.config.verbosity,
                capture_output=True
            )
            
            if self.config.verbosity >= 1:
                self.logger.info(f"Successfully built for {self.config.cpu}")
            
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message=f"Successfully built for {self.config.cpu}",
                outputs={"build_dir": str(find_build_dir(self.config.cpu, self.config.project_root))},
                details={"command": cmd},
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to build for FVP (exit code {e.returncode})"
            self.logger.error(error_msg)
            
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=BuildError(error_msg),
                outputs={"build_dir": str(find_build_dir(self.config.cpu, self.config.project_root))},
                details={"command": cmd},
            )
        except FileNotFoundError as e:
            error_msg = f"Failed to build for FVP: {e}"
            self.logger.error(error_msg)
            build_error = BuildError(error_msg)
            build_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=build_error,
                outputs={"build_dir": str(find_build_dir(self.config.cpu, self.config.project_root))},
                details={"command": cmd},
            )

    def dry_run(self) -> StepResult:
        """Dry run of build step."""
        cmd_preview = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        if not self.config.clean_build:
            cmd_preview.append("--no-clean-build")
        if self.config.jobs:
            cmd_preview.extend(["--jobs", str(self.config.jobs)])
        
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: {' '.join(cmd_preview)}",
            outputs={"build_dir": str(find_build_dir(self.config.cpu, self.config.project_root))},
        )

    def _plan_details(self) -> StepPlan:
        cmd = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        if not self.config.clean_build:
            cmd.append("--no-clean-build")
        if self.config.jobs:
            cmd.extend(["--jobs", str(self.config.jobs)])
        if self.config.verbosity <= 1:
            cmd.append("--quiet")
        else:
            cmd.extend(["--verbosity", str(self.config.verbosity)])
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=[cmd],
            outputs={"build_dir": str(find_build_dir(self.config.cpu, self.config.project_root))}
        )
