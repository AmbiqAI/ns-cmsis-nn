"""
TFLite model generation step.
"""

import subprocess
from pathlib import Path

from cmsis_nn_tools.core.steps.base import StepBase, StepResult, StepStatus
from cmsis_nn_tools.core.errors import GenerationError, PathNotFoundError
from cmsis_nn_tools.core.logging import get_logger
from cmsis_nn_tools.utils.command_runner import run_command


class GenerateStep(StepBase):
    """Step for generating TFLite models."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "generate"
    
    def should_skip(self) -> bool:
        """Check if generation should be skipped."""
        return self.config.skip_generation
    
    def validate(self) -> str | None:
        """Validate prerequisites for generation."""
        if not self.config.tflite_generator_dir.exists():
            return f"TFLite generator directory not found: {self.config.tflite_generator_dir}"
        return None
    
    def execute(self) -> StepResult:
        """Execute TFLite model generation."""
        if self.should_skip():
            return StepResult(
                status=StepStatus.SKIPPED,
                message="Generation skipped (--skip-generation)"
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
            self.logger.info("Generating TensorFlow Lite models using pytest")
        
        # Build pytest command
        cmd = ["pytest", "test_ops.py::test_generation", "-v"]
        
        if self.config.op_filter:
            cmd.extend(["--op", self.config.op_filter])
        if self.config.dtype_filter:
            cmd.extend(["--dtype", self.config.dtype_filter])
        if self.config.name_filter:
            cmd.extend(["--name", self.config.name_filter])
        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])
        if self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            
            run_command(
                cmd,
                cwd=self.config.tflite_generator_dir,
                verbosity=self.config.verbosity
            )
            
            if self.config.verbosity >= 1:
                self.logger.info("TFLite models generated successfully")
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="TFLite models generated successfully"
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = f"Failed to generate TFLite models: {e}"
            self.logger.error(error_msg)
            gen_error = GenerationError(error_msg)
            gen_error.__cause__ = e
            return StepResult(
                status=StepStatus.FAILED,
                message=error_msg,
                error=gen_error
            )
    
    def dry_run(self) -> StepResult:
        """Dry run of generation step."""
        cmd_preview = ["pytest", "test_ops.py::test_generation", "-v"]
        if self.config.op_filter:
            cmd_preview.extend(["--op", self.config.op_filter])
        if self.config.dtype_filter:
            cmd_preview.extend(["--dtype", self.config.dtype_filter])
        if self.config.name_filter:
            cmd_preview.extend(["--name", self.config.name_filter])
        if self.config.limit:
            cmd_preview.extend(["--limit", str(self.config.limit)])
        
        return StepResult(
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run: {' '.join(cmd_preview)} in {self.config.tflite_generator_dir}"
        )
