"""
Main pipeline orchestration for CMSIS-NN Tools.

This module orchestrates the execution of pipeline steps in the correct order.
"""

import time

from helia_core_tester.core.config import Config
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.steps import (
    GenerateStep,
    RunnersStep,
    BuildStep,
    RunStep,
    CleanStep,
    StepStatus,
)
from helia_core_tester.core.steps.base import StepPlan


def _run_step(step, logger, verbosity: int, fail_fast: bool):
    """Execute a step, log result, return (success, should_stop)."""
    result = step.execute()
    if result.success:
        if verbosity >= 1:
            duration = f" ({result.duration_sec:.2f}s)" if result.duration_sec is not None else ""
            logger.info(f"✓ {step.name} step completed{duration}")
        return True, False
    if result.skipped:
        if verbosity >= 1:
            logger.info(f"⊘ {step.name} step skipped: {result.message}")
        return True, False
    logger.error(f"✗ {step.name} step failed: {result.message}")
    return False, fail_fast


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

        if self.config.plan:
            self.print_plan()
            return True

        if self.config.dry_run:
            self.logger.warning("DRY RUN MODE - No actual changes will be made")
        
        start_time = time.time()
        overall_success = True
        
        try:
            if self.config.skip_generation:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping TFLite model generation (--skip-generation)")
            elif overall_success:
                success, stop = _run_step(
                    GenerateStep(self.config), self.logger,
                    self.config.verbosity, self.config.fail_fast
                )
                overall_success = success
                if stop:
                    return False

            if self.config.verbosity >= 1:
                if not self.config.skip_conversion and overall_success:
                    self.logger.info("Conversion step: C files are generated via templates during generation")
                else:
                    self.logger.info("Skipping TFLite to C conversion")

            runners_generated = False
            if self.config.skip_runners:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping test runner generation")
            elif overall_success:
                success, stop = _run_step(
                    RunnersStep(self.config), self.logger,
                    self.config.verbosity, self.config.fail_fast
                )
                runners_generated = success
                overall_success = success
                if stop:
                    return False

            if not runners_generated and not self.config.skip_runners and overall_success and not self.config.skip_conversion:
                if self.config.verbosity >= 1:
                    self.logger.info("Attempting to generate test runners after conversion...")
                success, stop = _run_step(
                    RunnersStep(self.config), self.logger,
                    self.config.verbosity, self.config.fail_fast
                )
                if not success:
                    overall_success = False
                if stop:
                    return False

            if self.config.skip_build:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping FVP build")
            elif overall_success:
                success, stop = _run_step(
                    BuildStep(self.config), self.logger,
                    self.config.verbosity, self.config.fail_fast
                )
                overall_success = success
                if stop:
                    return False

            if self.config.skip_run:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping FVP test execution")
            elif overall_success:
                success, stop = _run_step(
                    RunStep(self.config), self.logger,
                    self.config.verbosity, self.config.fail_fast
                )
                overall_success = success
            
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

    def build_plan(self) -> list[StepPlan]:
        """Build a plan for the pipeline without executing steps."""
        plans: list[StepPlan] = []

        if self.config.skip_generation:
            plans.append(StepPlan(name="generate", will_run=False, reason="skipped by config"))
        else:
            plans.append(GenerateStep(self.config).plan())

        if self.config.skip_conversion:
            plans.append(StepPlan(name="conversion", will_run=False, reason="skipped by config"))
        else:
            plans.append(
                StepPlan(
                    name="conversion",
                    will_run=True,
                    reason="handled during generation (templates)",
                    commands=[],
                    outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
                )
            )

        if self.config.skip_runners:
            plans.append(StepPlan(name="runners", will_run=False, reason="skipped by config"))
        else:
            plans.append(RunnersStep(self.config).plan())
            if not self.config.skip_conversion:
                plans.append(
                    StepPlan(
                        name="runners (post-conversion)",
                        will_run=True,
                        reason="retry if headers were missing before conversion",
                        commands=[],
                        outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
                        details={"retry": True},
                    )
                )

        if self.config.skip_build:
            plans.append(StepPlan(name="build", will_run=False, reason="skipped by config"))
        else:
            plans.append(BuildStep(self.config).plan())

        if self.config.skip_run:
            plans.append(StepPlan(name="run", will_run=False, reason="skipped by config"))
        else:
            plans.append(RunStep(self.config).plan())

        return plans

    def print_plan(self) -> None:
        """Print a human-readable execution plan."""
        plans = self.build_plan()
        self.logger.info("Plan:")
        for idx, plan in enumerate(plans, start=1):
            will = "will run" if plan.will_run else "skipped"
            self.logger.info(f"{idx}. {plan.name}: {will} ({plan.reason})")
            if plan.commands:
                for cmd in plan.commands:
                    self.logger.info(f"   cmd: {' '.join(cmd)}")
            if plan.outputs:
                outputs = ", ".join(f"{k}={v}" for k, v in plan.outputs.items() if v)
                if outputs:
                    self.logger.info(f"   outputs: {outputs}")
