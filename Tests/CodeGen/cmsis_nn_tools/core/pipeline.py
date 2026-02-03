"""
Main pipeline orchestration for Helia-Core Tester.
"""

import shutil
import subprocess
import time
from pathlib import Path

from .config import Config
from .logger import get_logger
from ..utils.command_runner import run_command


class FullTestPipeline:
    """Main pipeline class for Helia-Core Tester workflow."""
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def _get_platform_name(self, cpu: str) -> str:
        """
        
        Args:
            cpu: CPU type (e.g., 'cortex-m4', 'cortex-m55')
            
        Returns:
            Platform name for helia-aot
        """
        platform_mapping = {
            'cortex-m4': 'apollo4l_evb',
            'cortex-m55': 'apollo510_evb',
        }
        return platform_mapping.get(cpu, 'apollo4l_evb')
    
    def _cleanup_directories(self) -> None:
        """
        Clean up previous build and reports directories.
        """
        if self.config.dry_run:
            self.logger.info("DRY RUN: Would clean up build and reports directories")
            return
        
        # Clean up build directories
        project_root = self.config.project_root
        build_pattern = f"build-*-gcc"
        for build_dir in project_root.glob(build_pattern):
            if build_dir.is_dir():
                if self.config.verbosity >= 1:
                    self.logger.info(f"Removing previous build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
        
        # Clean up reports directory
        if self.config.report_dir.exists():
            if self.config.verbosity >= 1:
                self.logger.info(f"Removing previous reports directory: {self.config.report_dir}")
            shutil.rmtree(self.config.report_dir, ignore_errors=True)
    
    def run(self) -> bool:
        """
        Run the complete test pipeline.
        
        Returns:
            True if all steps succeeded, False otherwise
        """
        if self.config.verbosity >= 1:
            self.logger.info("Starting Helia-Core Tester Full Test Pipeline")
        
        if self.config.dry_run:
            self.logger.warning("DRY RUN MODE - No actual changes will be made")
        
        # Clean up previous build and reports directories
        self._cleanup_directories()
        
        start_time = time.time()
        overall_success = True
        
        try:
            if not self.config.skip_generation:
                success = self._step1_generate_tflite_models()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in TFLite generation")
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping TFLite model generation")
            
            if not self.config.skip_conversion and overall_success:
                success = self._step2_convert_tflite_models()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in TFLite conversion")
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping TFLite to C conversion")
            
            if not self.config.skip_runners and overall_success:
                success = self._step3_generate_test_runners()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in test runner generation")
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping test runner generation")
            
            if not self.config.skip_runners and overall_success and not self.config.skip_conversion:
                success = self._step3_5_generate_test_runners_after_conversion()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in test runner generation")
                        return False
            
            if not self.config.skip_build and overall_success:
                success = self._step4_build_fvp()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in FVP build")
                        return False
            else:
                if self.config.verbosity >= 1:
                    self.logger.info("Skipping FVP build")
            
            if not self.config.skip_run and overall_success:
                success = self._step5_run_tests()
                if not success:
                    overall_success = False
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
            self.logger.error(f"Pipeline failed with exception: {e}")
            return False
    
    def _step1_generate_tflite_models(self) -> bool:
        """Step 1: Generate TFLite models."""
        if self.config.verbosity >= 1:
            self.logger.info("Step 1/5: Generate TFLite Models")
        if self.config.verbosity >= 1:
            self.logger.info("Generating TensorFlow Lite models using pytest in tflite generator")
        
        if not self.config.tflite_generator_dir.exists():
            self.logger.error(f"tflite generator directory not found: {self.config.tflite_generator_dir.absolute()}")
            return False
        
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
            run_command(cmd, cwd=self.config.tflite_generator_dir, verbosity=self.config.verbosity)
            if self.config.verbosity >= 1:
                self.logger.info("TFLite models generated successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Failed to generate TFLite models: {e}")
            return False
    
    def _step2_convert_tflite_models(self) -> bool:
        """Step 2: Convert TFLite models to C modules."""
        if self.config.verbosity >= 1:
            self.logger.info("Step 2/5: Convert TFLite to C Modules")
        if self.config.verbosity >= 1:
            self.logger.info("Converting TensorFlow Lite models to standalone C inference modules")
        
        if not self.config.generated_tests_dir.exists():
            self.logger.error(f"Generated tests directory not found: {self.config.generated_tests_dir}")
            return False
        
        tflite_files = list(self.config.generated_tests_dir.rglob("*.tflite"))
        if not tflite_files:
            self.logger.error(f"No TFLite files found in {self.config.generated_tests_dir}")
            return False
        
        if self.config.verbosity >= 1:
            self.logger.info(f"Found {len(tflite_files)} TFLite files to convert")
        
        if self.config.dry_run:
            if self.config.verbosity >= 1:
                self.logger.info("Dry run mode - would convert the following files:")
            for tflite_file in tflite_files:
                if self.config.verbosity >= 1:
                    self.logger.info(f"  - {tflite_file}")
            return True

        platform_name = self._get_platform_name(self.config.cpu)
        if self.config.verbosity >= 1:
            self.logger.info(f"Using platform: {platform_name} for CPU: {self.config.cpu}")
    
        success_count = 0
        skipped_count = 0
        failed_count = 0
        for tflite_file in tflite_files:
            output_dir = tflite_file.parent
            module_name = tflite_file.stem
            
            # Check if C files were already generated by our template system
            # Look for generated C files in the output directory or headers in includes
            includes_api_dir = output_dir / "includes"
            
            # Check for any generated C files (could be conv2d.c, pooling.c, etc.)
            generated_c_files = list(output_dir.glob(f"{module_name}_*.c"))
            generated_h_files = list(includes_api_dir.glob(f"{module_name}_*.h")) if includes_api_dir.exists() else []
            
            # Check if we have generated C files (for operators that support direct generation)
            if generated_c_files or generated_h_files:
                if self.config.verbosity >= 1:
                    self.logger.info(f"C files already generated via templates")
                skipped_count += 1
                success_count += 1  # Count as success since files exist
                continue
            # If C files don't exist, mark as conversion failed
            if self.config.verbosity >= 1:
                self.logger.warning(f"Conversion failed for {module_name}: C files not generated (operator may not have generate_c_files implemented)")
            failed_count += 1
            continue
        
        if self.config.verbosity >= 1:
            self.logger.info(f"Conversion complete: {success_count} succeeded, {skipped_count} skipped (already generated), {failed_count} failed (no C generation)")
        
        if success_count == 0 and failed_count > 0:
            self.logger.error("No TFLite files were successfully converted")
            return False
        
        if self.config.verbosity >= 1:
            if skipped_count > 0:
                self.logger.info(f"Successfully processed {success_count}/{len(tflite_files)} TFLite files ({skipped_count} skipped - C files already generated")
            else:
                self.logger.info(f"Successfully converted {success_count}/{len(tflite_files)} TFLite files")
        return True
    
    def _step3_generate_test_runners(self) -> bool:
        """Step 3: Generate Unity test runners."""
        if self.config.verbosity >= 1:
            self.logger.info("Step 3/5: Generate Test Runners")
        if self.config.verbosity >= 1:
            self.logger.info("Generating Unity test runners for all test directories")
        
        script_path = self.config.project_root / "cmsis_nn_tools" / "scripts" / "generate_test_runners.py"
        if not script_path.exists():
            self.logger.error(f"Test runner script not found: {script_path}")
            return False
        
        model_headers = list(self.config.generated_tests_dir.rglob("includes/*_model.h"))
        template_headers = list(self.config.generated_tests_dir.rglob("includes/*.h"))
        all_headers = list(set(model_headers + template_headers))
        
        if not all_headers:
            self.logger.warning("No model headers found - test runners will be generated after TFLite conversion")
            if self.config.verbosity >= 1:
                self.logger.info("This step will be skipped for now and run after conversion")
            return True
        
        cmd = ["python3", str(script_path), "--root", str(self.config.generated_tests_dir)]
        if self.config.dry_run:
            cmd.append("--dry-run")
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            run_command(cmd, verbosity=self.config.verbosity)
            if self.config.verbosity >= 1:
                self.logger.info("Test runners generated successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Failed to generate test runners: {e}")
            return False
    
    def _step3_5_generate_test_runners_after_conversion(self) -> bool:
        """Step 3.5: Generate test runners after conversion if needed."""
        model_headers = list(self.config.generated_tests_dir.rglob("includes/*_model.h"))
        template_headers = list(self.config.generated_tests_dir.rglob("includes/*.h"))
        all_headers = list(set(model_headers + template_headers))
        
        if not all_headers:
            self.logger.info("Generating test runners after TFLite conversion...")
            return self._step3_generate_test_runners()
        return True
    
    def _step4_build_fvp(self) -> bool:
        """Step 4: Build for FVP."""
        if self.config.verbosity >= 1:
            self.logger.info("Step 4/5: Build for FVP")
        
        if self.config.dry_run:
            if self.config.verbosity >= 1:
                self.logger.info("Dry run mode - would build for FVP")
            return True
        
        script_path = self.config.project_root / "cmsis_nn_tools" / "build_and_run_fvp.py"
        if not script_path.exists():
            self.logger.error(f"Build script not found: {script_path}")
            return False
        
        cmd = [
            "python3", str(script_path),
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
        ]
        
        if self.config.jobs:
            cmd.extend(["--jobs", str(self.config.jobs)])
        # Map verbosity to quiet: verbosity 0-1 = quiet, 2-3 = not quiet
        if self.config.verbosity <= 1:
            cmd.append("--quiet")
        
        try:
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            run_command(cmd, verbosity=self.config.verbosity)
            if self.config.verbosity >= 1:
                self.logger.info(f"Successfully built for {self.config.cpu}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build for FVP (exit code {e.returncode})")
            try:
                result = subprocess.run(
                    cmd,
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
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Failed to build for FVP: {e}")
            return False
    
    def _step5_run_tests(self) -> bool:
        """Step 5: Run tests on FVP."""
        if self.config.verbosity >= 1:
            self.logger.info("Step 5/5: Run Tests on FVP")
        
        if self.config.dry_run:
            if self.config.verbosity >= 1:
                self.logger.info("Dry run mode - would run tests on FVP")
            return True
        
        script_path = self.config.project_root / "cmsis_nn_tools" / "build_and_run_fvp.py"
        if not script_path.exists():
            self.logger.error(f"Build script not found: {script_path}")
            return False
        
        cmd = [
            "python3", str(script_path),
            "--cpu", self.config.cpu,
            "--no-build",
        ]
        
        if self.config.timeout > 0:
            cmd.extend(["--timeout-run", str(self.config.timeout)])
        if not self.config.fail_fast:
            cmd.append("--no-fail-fast")
        
        # Pass verbosity to build_and_run_fvp.py
        cmd.extend(["--verbosity", str(self.config.verbosity)])
        
        if hasattr(self.config, 'enable_reporting'):
            if not self.config.enable_reporting:
                cmd.append("--no-report")
            if hasattr(self.config, 'report_formats') and self.config.report_formats:
                cmd.extend(["--report-formats"] + self.config.report_formats)
            if hasattr(self.config, 'report_dir') and self.config.report_dir:
                cmd.extend(["--report-dir", str(self.config.report_dir)])
        
        try:
            if self.config.verbosity >= 1:
                self.logger.info("Running tests on FVP")
            if self.config.verbosity >= 2:
                self.logger.info(f"Running command: {' '.join(cmd)}")
            if self.config.verbosity >= 2:
                self.logger.info("=" * 60)
            
            subprocess.run(
                cmd,
                check=True,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if self.config.verbosity >= 2:
                self.logger.info("=" * 60)
            if self.config.verbosity >= 1:
                self.logger.info("All tests completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            if self.config.verbosity >= 2:
                self.logger.error("=" * 60)
            self.logger.error(f"Some tests failed (exit code: {e.returncode})")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Failed to run tests: {e}")
            return False
