"""
Main pipeline orchestration for CMSIS-NN Tools.

This module orchestrates the execution of pipeline steps in the correct order.
"""

import time

from cmsis_nn_tools.core.config import Config
from cmsis_nn_tools.core.logging import get_logger
from cmsis_nn_tools.core.steps import (
    GenerateStep,
    RunnersStep,
    BuildStep,
    RunStep,
    CleanStep,
    StepStatus,
)


class FullTestPipeline:
    """Main pipeline class for CMSIS-NN Tools workflow."""
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def run(self) -> bool:
        """
        Run the complete test pipeline.
        
        The pipeline executes steps in order:
        1. Generate TFLite models (if not skipped)
        2. Generate Unity test runners (if not skipped, after conversion check)
        3. Build FVP executables (if not skipped)
        4. Run tests on FVP (if not skipped)
        
        Returns:
            True if all steps succeeded, False otherwise
        """
        if self.config.verbosity >= 1:
            self.logger.info("Starting CMSIS-NN Tools Full Test Pipeline")
        
        if self.config.dry_run:
            self.logger.warning("DRY RUN MODE - No actual changes will be made")
        
        start_time = time.time()
        overall_success = True
        
        try:
            # Step 1: Generate TFLite models
            if not self.config.skip_generation:
                step = GenerateStep(self.config)
                result = step.execute()
                
                if result.success:
                    if self.config.verbosity >= 1:
                        self.logger.info("✓ Generation step completed")
                elif result.skipped:
                    if self.config.verbosity >= 1:
                        self.logger.info(f"⊘ Generation step skipped: {result.message}")
                else:
                    overall_success = False
                    self.logger.error(f"✗ Generation step failed: {result.message}")
                    if self.config.fail_fast:
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping TFLite model generation (--skip-generation)")
            
            # Step 2: Conversion is handled during generation (template-based)
            # This step is kept for backward compatibility but is mostly a no-op
            # since C files are generated via templates during TFLite generation
            if not self.config.skip_conversion and overall_success:
                if self.config.verbosity >= 1:
                    self.logger.info("Conversion step: C files are generated via templates during generation")
                # No separate conversion step needed - templates handle it
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping TFLite to C conversion")
            
            # Step 3: Generate Unity test runners
            # Try once before build, and again after if needed
            runners_generated = False
            if not self.config.skip_runners and overall_success:
                step = RunnersStep(self.config)
                result = step.execute()
                
                if result.success:
                    runners_generated = True
                    if self.config.verbosity >= 1:
                        self.logger.info("✓ Test runners generated")
                elif result.skipped:
                    if self.config.verbosity >= 1:
                        self.logger.info(f"⊘ Test runners skipped: {result.message}")
                    # Will try again after conversion if needed
                else:
                    overall_success = False
                    self.logger.error(f"✗ Test runner generation failed: {result.message}")
                    if self.config.fail_fast:
                        return False
            
            # Step 3.5: Try generating runners again after conversion if they weren't generated
            if not runners_generated and not self.config.skip_runners and overall_success and not self.config.skip_conversion:
                if self.config.verbosity >= 1:
                    self.logger.info("Attempting to generate test runners after conversion...")
                step = RunnersStep(self.config)
                result = step.execute()
                
                if result.success:
                    if self.config.verbosity >= 1:
                        self.logger.info("✓ Test runners generated after conversion")
                elif not result.skipped:
                    overall_success = False
                    self.logger.error(f"✗ Test runner generation failed: {result.message}")
                    if self.config.fail_fast:
                        return False
            
            # Step 4: Build FVP executables
            if not self.config.skip_build and overall_success:
                step = BuildStep(self.config)
                result = step.execute()
                
                if result.success:
                    if self.config.verbosity >= 1:
                        self.logger.info(f"✓ Build completed for {self.config.cpu}")
                elif result.skipped:
                    if self.config.verbosity >= 1:
                        self.logger.info(f"⊘ Build skipped: {result.message}")
                else:
                    overall_success = False
                    self.logger.error(f"✗ Build failed: {result.message}")
                    if self.config.fail_fast:
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping FVP build")
            
            # Step 5: Run tests on FVP
            if not self.config.skip_run and overall_success:
                step = RunStep(self.config)
                result = step.execute()
                
                if result.success:
                    if self.config.verbosity >= 1:
                        self.logger.info("✓ Test execution completed")
                elif result.skipped:
                    if self.config.verbosity >= 1:
                        self.logger.info(f"⊘ Test execution skipped: {result.message}")
                else:
                    overall_success = False
                    self.logger.error(f"✗ Test execution failed: {result.message}")
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping FVP test execution")
            
            end_time = time.time()
            duration = end_time - start_time
            
            if overall_success:
                if self.config.verbosity >= 1:
                    self.logger.info(f"Pipeline completed successfully in {duration:.1f} seconds")
                return True
            else:
                self.logger.error(f"Pipeline failed after {duration:.1f} seconds")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}", exc_info=self.config.verbosity >= 3)
            return False
