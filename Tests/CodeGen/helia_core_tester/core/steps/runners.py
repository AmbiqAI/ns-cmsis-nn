"""
Unity test runner generation step.
"""

import subprocess
from pathlib import Path

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import StepExecutionError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.utils.command_runner import run_command


class RunnersStep(StepBase):
    """Step for generating Unity test runners."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "runners"
    
    def should_skip(self) -> bool:
        """Check if runner generation should be skipped."""
        return self.config.skip_runners

    def _script_path(self) -> Path:
        """Path to the test runner generation script (single definition)."""
        return self.config.project_root / "helia_core_tester" / "scripts" / "generate_test_runners.py"
    
    def validate(self) -> str | None:
        """Validate prerequisites for runner generation."""
        if not self._script_path().exists():
            return f"Test runner script not found: {self._script_path()}"
        if not self.config.generated_tests_dir.exists():
            return f"Generated tests directory not found: {self.config.generated_tests_dir}"
        return None

    def plan_validate(self) -> str | None:
        """Lenient validation for plan mode."""
        if not self._script_path().exists():
            return f"Test runner script not found: {self._script_path()}"
        return None

    def _do_execute(self) -> StepResult:
        """Execute Unity test runner generation."""
        if self.config.verbosity >= 1:
            self.logger.info("Generating Unity test runners for all test directories")
        script_path = self._script_path()
        model_headers = list(self.config.generated_tests_dir.rglob("includes/*_model.h"))
        template_headers = list(self.config.generated_tests_dir.rglob("includes/*.h"))
        all_headers = list(set(model_headers + template_headers))
        if not all_headers:
            if self.config.verbosity >= 1:
                self.logger.warning("No model headers found - test runners will be generated after conversion")
            return StepResult(
                name=self.name,
                status=StepStatus.SKIPPED,
                message="No model headers found, skipping runner generation (will run after conversion)",
                outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
                details={"checked_headers": 0},
            )
        cmd = ["python3", str(script_path), "--root", str(self.config.generated_tests_dir)]
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            run_command(cmd, verbosity=self.config.verbosity)
            if self.config.verbosity >= 1:
                self.logger.info("Test runners generated successfully")
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message="Test runners generated successfully",
                outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
                details={"command": cmd, "headers_found": len(all_headers)},
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = f"Failed to generate test runners: {e}"
            self.logger.error(error_msg)
            exec_error = StepExecutionError(error_msg)
            exec_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=exec_error,
                outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
                details={"command": cmd},
            )

    def dry_run(self) -> StepResult:
        """Dry run of runner generation step."""
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: python3 {self._script_path()} --root {self.config.generated_tests_dir}",
            outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
        )

    def _plan_details(self) -> StepPlan:
        cmd = ["python3", str(self._script_path()), "--root", str(self.config.generated_tests_dir)]
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=[cmd],
            outputs={"generated_tests_dir": str(self.config.generated_tests_dir)}
        )
