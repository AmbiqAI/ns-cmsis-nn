"""
Cleanup step for removing build artifacts.
"""

import shutil
from pathlib import Path

from cmsis_nn_tools.core.steps.base import StepBase, StepResult, StepStatus
from cmsis_nn_tools.core.errors import StepExecutionError
from cmsis_nn_tools.core.logging import get_logger


class CleanStep(StepBase):
    """Step for cleaning up build artifacts and reports."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "clean"
    
    def should_skip(self) -> bool:
        """Clean step is never skipped by default."""
        return False
    
    def validate(self) -> str | None:
        """No validation needed for clean step."""
        return None
    
    def execute(self) -> StepResult:
        """Execute cleanup of build artifacts."""
        if self.config.dry_run:
            return self.dry_run()
        
        if self.config.verbosity >= 1:
            self.logger.info("Cleaning up build artifacts and reports")
        
        cleaned_items = []
        
        try:
            # Clean up build directories
            project_root = self.config.project_root
            build_pattern = "build-*-gcc"
            
            for build_dir in project_root.glob(build_pattern):
                if build_dir.is_dir():
                    if self.config.verbosity >= 1:
                        self.logger.info(f"Removing build directory: {build_dir}")
                    shutil.rmtree(build_dir, ignore_errors=True)
                    cleaned_items.append(str(build_dir))
            
            # Clean up reports directory
            if self.config.report_dir.exists():
                if self.config.verbosity >= 1:
                    self.logger.info(f"Removing reports directory: {self.config.report_dir}")
                shutil.rmtree(self.config.report_dir, ignore_errors=True)
                cleaned_items.append(str(self.config.report_dir))
            
            if cleaned_items:
                message = f"Cleaned {len(cleaned_items)} item(s): {', '.join(cleaned_items)}"
            else:
                message = "No artifacts to clean"
            
            if self.config.verbosity >= 1:
                self.logger.info(message)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message=message
            )
        except Exception as e:
            error_msg = f"Failed to clean artifacts: {e}"
            self.logger.error(error_msg)
            exec_error = StepExecutionError(error_msg)
            exec_error.__cause__ = e
            return StepResult(
                status=StepStatus.FAILED,
                message=error_msg,
                error=exec_error
            )
    
    def dry_run(self) -> StepResult:
        """Dry run of clean step."""
        cleaned_items = []
        
        # Check what would be cleaned
        project_root = self.config.project_root
        build_pattern = "build-*-gcc"
        
        for build_dir in project_root.glob(build_pattern):
            if build_dir.is_dir():
                cleaned_items.append(str(build_dir))
        
        if self.config.report_dir.exists():
            cleaned_items.append(str(self.config.report_dir))
        
        if cleaned_items:
            message = f"DRY RUN: Would clean {len(cleaned_items)} item(s): {', '.join(cleaned_items)}"
        else:
            message = "DRY RUN: No artifacts to clean"
        
        return StepResult(
            status=StepStatus.SKIPPED,
            message=message
        )
