"""
FVP test execution step.
"""

import subprocess
import sys
from pathlib import Path

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import FVPRunError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.discovery import find_fvp_script_path, find_build_dir
from helia_core_tester.utils.command_runner import run_command


class RunStep(StepBase):
    """Step for running tests on FVP."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "run"
    
    def should_skip(self) -> bool:
        """Check if test execution should be skipped."""
        return self.config.skip_run
    
    def validate(self) -> str | None:
        """Validate prerequisites for test execution."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        build_dir = find_build_dir(self.config.cpu, self.config.project_root)
        if not build_dir.exists():
            return f"Build directory not found: {build_dir}. Run 'build' step first."
        return None

    def plan_validate(self) -> str | None:
        """Lenient validation for plan mode."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        return None

    def _do_execute(self) -> StepResult:
        """Execute FVP test execution."""
        if self.config.verbosity >= 1:
            self.logger.info("Running tests on FVP")
        cmd = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--no-build",
        ]
        
        if self.config.timeout > 0:
            cmd.extend(["--timeout-run", str(self.config.timeout)])
        if not self.config.fail_fast:
            cmd.append("--no-fail-fast")
        
        # Pass verbosity
        cmd.extend(["--verbosity", str(self.config.verbosity)])
        
        # Reporting options
        if hasattr(self.config, 'enable_reporting'):
            if not self.config.enable_reporting:
                cmd.append("--no-report")
            if hasattr(self.config, 'report_formats') and self.config.report_formats:
                cmd.extend(["--report-formats"] + self.config.report_formats)
            if hasattr(self.config, 'report_dir') and self.config.report_dir:
                cmd.extend(["--report-dir", str(self.config.report_dir)])
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
                self.logger.info("=" * 60)
            
            run_command(
                cmd,
                cwd=self.config.project_root,
                verbosity=self.config.verbosity,
                stream_output=True
            )
            
            if self.config.verbosity >= 2:
                self.logger.info("=" * 60)
            
            if self.config.verbosity >= 1:
                self.logger.info("All tests completed successfully")
            
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message="All tests completed successfully",
                outputs={
                    "build_dir": str(find_build_dir(self.config.cpu, self.config.project_root)),
                    "report_dir": str(self.config.report_dir) if self.config.report_dir else "",
                },
                details={"command": cmd},
            )
        except subprocess.CalledProcessError as e:
            if self.config.verbosity >= 2:
                self.logger.error("=" * 60)
            
            error_msg = f"Some tests failed (exit code: {e.returncode})"
            self.logger.error(error_msg)
            
            fvp_error = FVPRunError(error_msg)
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=fvp_error,
                outputs={
                    "build_dir": str(find_build_dir(self.config.cpu, self.config.project_root)),
                    "report_dir": str(self.config.report_dir) if self.config.report_dir else "",
                },
                details={"command": cmd},
            )
        except FileNotFoundError as e:
            error_msg = f"Failed to run tests: {e}"
            self.logger.error(error_msg)
            fvp_error = FVPRunError(error_msg)
            fvp_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=fvp_error,
                outputs={
                    "build_dir": str(find_build_dir(self.config.cpu, self.config.project_root)),
                    "report_dir": str(self.config.report_dir) if self.config.report_dir else "",
                },
                details={"command": cmd},
            )

    def dry_run(self) -> StepResult:
        """Dry run of run step."""
        cmd_preview = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,   
            "--no-build",
        ]
        if self.config.timeout > 0:
            cmd_preview.extend(["--timeout-run", str(self.config.timeout)])
        if not self.config.fail_fast:
            cmd_preview.append("--no-fail-fast")
        
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: {' '.join(cmd_preview)}",
            outputs={
                "build_dir": str(find_build_dir(self.config.cpu, self.config.project_root)),
                "report_dir": str(self.config.report_dir) if self.config.report_dir else "",
            },
        )

    def _plan_details(self) -> StepPlan:
        cmd = [
            sys.executable, "-m", "helia_core_tester.fvp.build_and_run_fvp",
            "--cpu", self.config.cpu,
            "--no-build",
        ]
        if self.config.timeout > 0:
            cmd.extend(["--timeout-run", str(self.config.timeout)])
        if not self.config.fail_fast:
            cmd.append("--no-fail-fast")
        cmd.extend(["--verbosity", str(self.config.verbosity)])
        if hasattr(self.config, 'enable_reporting'):
            if not self.config.enable_reporting:
                cmd.append("--no-report")
            if hasattr(self.config, 'report_formats') and self.config.report_formats:
                cmd.extend(["--report-formats"] + self.config.report_formats)
            if hasattr(self.config, 'report_dir') and self.config.report_dir:
                cmd.extend(["--report-dir", str(self.config.report_dir)])
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=[cmd],
            outputs={
                "build_dir": str(find_build_dir(self.config.cpu, self.config.project_root)),
                "report_dir": str(self.config.report_dir) if self.config.report_dir else "",
            }
        )
