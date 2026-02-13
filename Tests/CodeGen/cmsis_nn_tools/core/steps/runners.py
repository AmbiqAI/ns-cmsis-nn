"""
Unity test runner generation step.
"""

import subprocess
from pathlib import Path

from cmsis_nn_tools.core.steps.base import StepBase, StepResult, StepStatus
from cmsis_nn_tools.core.errors import StepExecutionError, PathNotFoundError
from cmsis_nn_tools.core.logging import get_logger
from cmsis_nn_tools.utils.command_runner import run_command


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
    
    def validate(self) -> str | None:
        """Validate prerequisites for runner generation."""
        script_path = self.config.project_root / "cmsis_nn_tools" / "scripts" / "generate_test_runners.py"
        
        if not script_path.exists():
            return f"Test runner script not found: {script_path}"
        
        if not self.config.generated_tests_dir.exists():
            return f"Generated tests directory not found: {self.config.generated_tests_dir}"
        
        return None
    
    def execute(self) -> StepResult:
        """Execute Unity test runner generation."""
        if self.should_skip():
            return StepResult(
                status=StepStatus.SKIPPED,
                message="Runner generation skipped (--skip-runners)"
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
            self.logger.info("Generating Unity test runners for all test directories")
        
        script_path = self.config.project_root / "cmsis_nn_tools" / "scripts" / "generate_test_runners.py"
        
        # Check for model headers
        model_headers = list(self.config.generated_tests_dir.rglob("includes/*_model.h"))
        template_headers = list(self.config.generated_tests_dir.rglob("includes/*.h"))
        all_headers = list(set(model_headers + template_headers))
        
        if not all_headers:
            if self.config.verbosity >= 1:
                self.logger.warning("No model headers found - test runners will be generated after conversion")
            return StepResult(
                status=StepStatus.SKIPPED,
                message="No model headers found, skipping runner generation (will run after conversion)"
            )
        
        cmd = ["python3", str(script_path), "--root", str(self.config.generated_tests_dir)]
        if self.config.dry_run:
            cmd.append("--dry-run")
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            
            run_command(cmd, verbosity=self.config.verbosity)
            
            if self.config.verbosity >= 1:
                self.logger.info("Test runners generated successfully")
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Test runners generated successfully"
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = f"Failed to generate test runners: {e}"
            self.logger.error(error_msg)
            exec_error = StepExecutionError(error_msg)
            exec_error.__cause__ = e
            return StepResult(
                status=StepStatus.FAILED,
                message=error_msg,
                error=exec_error
            )
    
    def dry_run(self) -> StepResult:
        """Dry run of runner generation step."""
        script_path = self.config.project_root / "cmsis_nn_tools" / "scripts" / "generate_test_runners.py"
        
        return StepResult(
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: python3 {script_path} --root {self.config.generated_tests_dir}"
        )
