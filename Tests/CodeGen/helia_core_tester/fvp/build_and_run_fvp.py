#!/usr/bin/env python3
"""
build_and_run_fvp.py — Python replacement for CMSIS-NN UnitTest build+run on FVP Corstone-300.

Examples:
  # vanilla (downloads in ./artifacts/downloads, GCC toolchain from downloads, FVP from downloads)
  python3 python_scripts/build_and_run_fvp.py

  # multiple CPUs, less optimization, quiet logs
  python3 python_scripts/build_and_run_fvp.py -c cortex-m3,cortex-m55 -o "-O2" -q

  # skip downloads/setup (assume paths present), override paths, pass extra CMake defs
  python3 python_scripts/build_and_run_fvp.py -e -u ./artifacts/downloads/ethos-u-core-platform -C ./artifacts/downloads/CMSIS_5 \
    -D CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=ON

  # use Arm Compiler, custom generator, increased timeouts
  python3 python_scripts/build_and_run_fvp.py -a --generator Ninja --timeout-run 180

Notes:
- This script mirrors flags of ns-cmsis-nn/Tests/UnitTest/build_and_run_tests.sh where sensible.
- It expects Linux (same as the bash script).
"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Import reporting and discovery (package imports only)
from helia_core_tester.core.discovery import (
    find_repo_root,
    find_setup_dependencies_script,
    find_descriptors_dir,
    find_generated_tests_dir,
    find_build_dir,
)
from helia_core_tester.reporting.models import TestResult, TestStatus
from helia_core_tester.reporting.parser import TestResultParser

repo_root = find_repo_root()
ARTIFACTS_DIR = repo_root / "artifacts"
DEFAULT_DL = ARTIFACTS_DIR / "downloads"
DEFAULT_SOURCE = repo_root

FVP_EXE_NAME = "FVP_Corstone_SSE-300_Ethos-U55"
FVP_DIR_X86 = "Linux64_GCC-9.3"
FVP_DIR_AARCH64 = "Linux64_armv8l_GCC-9.3"


def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def arch_tag() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    die(f"Unsupported architecture: {m}")


def ensure_exe_on_path(name: str) -> Optional[str]:
    return shutil.which(name)


def call_setup_dependencies(downloads_dir: Path) -> None:
    setup = find_setup_dependencies_script(repo_root)
    if not setup or not setup.exists():
        print("No setup_dependencies.py found; skipping dependency setup.")
        return
    print("Ensuring dependencies via setup_dependencies.py")
    rc = subprocess.call(
        [sys.executable, str(setup), "--downloads-dir", str(downloads_dir)],
        cwd=str(repo_root),
    )
    if rc != 0:
        die(f"Dependency setup failed (rc={rc})")


def prepend_path(p: Path, env: dict) -> None:
    env["PATH"] = str(p) + os.pathsep + env.get("PATH", "")


def detect_paths(args) -> dict:
    # base
    env = os.environ.copy()
    arch = arch_tag()

    # downloads root
    dl = args.downloads_dir.resolve()

    # ethos-u core platform
    ethos = Path(args.ethos_path).resolve() if args.ethos_path else (dl / "ethos-u-core-platform")
    if not ethos.exists():
        die(f"Ethos-U core platform not found: {ethos}. Run without -e or point -u to a valid path.")

    # cmsis5
    cmsis5 = Path(args.cmsis5_path).resolve() if args.cmsis5_path else (dl / "CMSIS_5")
    if not cmsis5.exists():
        die(f"CMSIS_5 not found: {cmsis5}. Run without -e or point -C to a valid path.")

    # toolchain file & compiler tag
    if args.use_arm_compiler:
        toolchain_file = ethos / "cmake" / "toolchain" / "armclang.cmake"
        compiler_tag = "arm-compiler"
    else:
        toolchain_file = ethos / "cmake" / "toolchain" / "arm-none-eabi-gcc.cmake"
        compiler_tag = "gcc"
        if not args.no_gcc_from_download:
            gcc_bin = dl / "arm_gcc_download" / "bin"
            if not gcc_bin.exists():
                die(f"GCC toolchain not found at {gcc_bin}. Run without -e or install gcc on PATH.")
            prepend_path(gcc_bin, env)

    if not toolchain_file.exists():
        die(f"Toolchain file missing: {toolchain_file}")

    # FVP
    if arch == "x86_64":
        fvp_dir = dl / "corstone300_download" / "models" / FVP_DIR_X86
    elif arch == "aarch64":
        fvp_dir = dl / "corstone300_download" / "models" / FVP_DIR_AARCH64
    else:
        die(f"Unsupported architecture for FVP: {arch}")
    fvp_exe: Optional[Path] = None
    if not args.no_fvp_from_download:
        fvp_exe_candidate = fvp_dir / FVP_EXE_NAME
        if not fvp_exe_candidate.exists():
            die(f"FVP not found at {fvp_exe_candidate}. Run with -f to use a system FVP on PATH.")
        prepend_path(fvp_dir, env)
        fvp_exe = fvp_exe_candidate
    else:
        from_path = ensure_exe_on_path(FVP_EXE_NAME)
        if not from_path:
            die(f"{FVP_EXE_NAME} not on PATH (use downloads or add it).")
        fvp_exe = Path(from_path)

    return {
        "env": env,
        "dl": dl,
        "ethos": ethos,
        "cmsis5": cmsis5,
        "toolchain_file": toolchain_file,
        "compiler_tag": compiler_tag,
        "fvp_exe": fvp_exe,
    }


def _get_git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True
        ).strip()
    except Exception:
        return None


def cmake_configure(source_dir: Path, build_dir: Path, toolchain_file: Path, cpu: str,
                    cmsis5: Path, optimization: str, extra_defs: List[str], generator: Optional[str],
                    verbosity: int, env: dict) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        cmake_cache.unlink()
    
    cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DTARGET_CPU={cpu}",
        f"-DCMSIS_PATH={cmsis5}",
        f"-DCMSIS_OPTIMIZATION_LEVEL={optimization}",
    ] + [f"-D{d}" for d in extra_defs]
    
    if generator:
        cmd += ["-G", generator]
    if verbosity >= 2:
        print(f"Configure: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        die(f"CMake configure failed for {cpu} (rc={rc})")


def cmake_build(build_dir: Path, verbosity: int, env: dict, jobs: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir)]
    if jobs and jobs > 0:
        # Pass through to the underlying build tool (works for Make/Ninja)
        cmd += ["--", f"-j{jobs}"]
    if verbosity >= 2:
        print(f"Build: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        die(f"CMake build failed (rc={rc})")


def find_elves(build_dir: Path) -> List[Path]:
    # First try to find ELF files in a 'tests' subdirectory
    tests_dir = build_dir / "tests"
    if tests_dir.exists():
        return [p for p in tests_dir.rglob("*.elf") if p.is_file()]
    # Fallback to searching the entire build directory
    return [p for p in build_dir.rglob("*.elf") if p.is_file()]


def run_fvp_with_reporting(fvp_exe: Path, elf: Path, timeout: float, verbosity: int, extra_args: List[str], env: dict, cpu: str, descriptor_name: Optional[str] = None) -> TestResult:
    """
    Run FVP with comprehensive result reporting.
    
    Returns:
        TestResult object with detailed test information
    """
    start_time = time.time()
    parser = TestResultParser()
    
    args = [
        str(fvp_exe),
        "-C", "mps3_board.uart0.shutdown_on_eot=1",
        "-C", "mps3_board.visualisation.disable-visualisation=1",
        "-C", "mps3_board.telnetterminal0.start_telnet=0",
        "-C", "mps3_board.uart0.out_file=-",
        "-C", "mps3_board.uart0.unbuffered_output=1",
    ] + extra_args + [str(elf)]
    
    if verbosity >= 2:
        print(f"Run: {' '.join(args)}")
    
    try:
        proc = subprocess.run(
            args,
            cwd=str(repo_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=None if timeout <= 0 else timeout,
        )
        exit_code = proc.returncode
        output = proc.stdout or ""
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f"TIMEOUT running {elf}", file=sys.stderr)
        
        # Create timeout result
        return TestResult(
            test_name=elf.stem,
            status=TestStatus.TIMEOUT,
            duration=duration,
            cpu=cpu,
            elf_path=elf,
            failure_reason="Test execution timed out",
            timestamp=datetime.now(),
            exit_code=124,
            error_type="timeout",
            descriptor_name=descriptor_name or elf.stem
        )
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"ERROR running {elf}: {e}", file=sys.stderr)
        
        # Create error result
        return TestResult(
            test_name=elf.stem,
            status=TestStatus.ERROR,
            duration=duration,
            cpu=cpu,
            elf_path=elf,
            failure_reason=f"Execution error: {str(e)}",
            timestamp=datetime.now(),
            exit_code=-1,
            error_type="crash",
            descriptor_name=descriptor_name or elf.stem
        )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Parse the output to create TestResult
    result = parser.parse_fvp_output(output, elf, cpu, duration, exit_code, descriptor_name=descriptor_name)
    
    # Display results based on verbosity
    if result.status == TestStatus.PASS:
        # Level 0: Only test name and pass/fail
        print(f"PASS: {elf}")
        if verbosity >= 3:
            sys.stdout.write(output)
            sys.stdout.flush()
        elif verbosity >= 1:
            # Level 1: Add summary
            pass  # Already printed PASS
    else:
        # Always show failures
        print(f"FAIL: {elf}")
        if verbosity >= 1:
            print("=" * 60)
            print("FAILURE DETAILS:")
            print("=" * 60)
            
            if result.failure_reason:
                print(f"Reason: {result.failure_reason}")
                print()
            
            # Show output differences based on verbosity
            if verbosity >= 2:
                # Level 2+: Show expected vs actual output
                if result.expected_output or result.actual_output:
                    print("OUTPUT COMPARISON:")
                    if result.expected_output:
                        print(f"  Expected (Golden): {result.expected_output}")
                    if result.actual_output:
                        print(f"  Actual (Got):     {result.actual_output}")
                    print()
                
                # Show detailed differences
                if result.output_differences:
                    print("DETAILED DIFFERENCES:")
                    max_diffs = 20 if verbosity < 3 else len(result.output_differences)
                    for diff in result.output_differences[:max_diffs]:
                        print(f"  {diff}")
                    if len(result.output_differences) > max_diffs:
                        print(f"  ... ({len(result.output_differences) - max_diffs} more differences)")
                    print()
            
            # Show relevant output lines
            max_lines = 20 if verbosity < 3 else len(result.output_lines)
            if result.output_lines:
                print("TEST OUTPUT:")
                for line in result.output_lines[:max_lines]:
                    print(f"  {line}")
                if len(result.output_lines) > max_lines and verbosity < 3:
                    print("  ... (truncated)")
                print()
            
            if verbosity >= 1:
                print("=" * 60)
    
    return result

def parse_cpus(cpu_str: str) -> List[str]:
    return [c.strip() for c in cpu_str.split(",") if c.strip()]




def run_tests_with_reporting(cpus: List[str], 
                           source_dir: Path,
                           toolchain_file: Path,
                           cmsis5: Path,
                           fvp_exe: Path,
                           args,
                           env: dict,
                           enable_reporting: bool) -> Tuple[List[TestResult], bool]:
    """
    Run tests with comprehensive reporting.
    
    Returns:
        Tuple of (list of TestResult objects, overall success)
    """
    from helia_core_tester.reporting.generator import ReportGenerator
    from helia_core_tester.reporting.models import TestReport, TestStatus, DescriptorResult
    from helia_core_tester.reporting.descriptor_tracker import DescriptorTracker
    
    all_results = []
    any_fail = False
    start_time = datetime.now()
    
    # Get verbosity from args (default to 0 if not provided)
    verbosity = getattr(args, 'verbosity', 0)
    
    # Get report directory from args
    report_dir = getattr(args, 'report_dir', ARTIFACTS_DIR / "reports")
    
    # Clean up previous build directories (only if we're going to build)
    # If --no-build or --no-clean-build is set, keep the existing build directory
    if not args.no_build and not args.no_clean_build:
        for cpu in cpus:
            build_dir = find_build_dir(cpu, source_dir)
            if build_dir.exists():
                if verbosity >= 1:
                    print(f"Removing previous build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
    
    # Clean up previous reports directory
    if report_dir.exists():
        if verbosity >= 1:
            print(f"Removing previous reports directory: {report_dir}")
        shutil.rmtree(report_dir, ignore_errors=True)
    
    generator = ReportGenerator(output_dir=report_dir) if enable_reporting else None
    
    # Initialize descriptor tracking
    descriptors_dir = find_descriptors_dir()
    generated_tests_dir = find_generated_tests_dir(create=False)
    tracker = DescriptorTracker(descriptors_dir)
    all_descriptors_dict = tracker.load_all_descriptors()
    
    for cpu in cpus:
        if verbosity >= 1:
            print(f"\nTarget: {cpu} (gcc)")
        build_dir = find_build_dir(cpu, source_dir)
        
        if not args.no_build:
            # Build first - use the env passed in
            cmake_configure(
                source_dir=source_dir,
                build_dir=build_dir,
                toolchain_file=toolchain_file,
                cpu=cpu,
                cmsis5=cmsis5,
                optimization=args.opt,
                extra_defs=args.cmake_def,
                generator=args.generator,
                verbosity=verbosity,
                env=env,
            )
            cmake_build(build_dir=build_dir, verbosity=verbosity, env=env, jobs=args.jobs)
        
        if args.no_run:
            continue
        
        elves = find_elves(build_dir)
        if not elves:
            if verbosity >= 1:
                print(f"(no .elf found under {build_dir}, nothing to run)")
            continue
        
        # Run tests for this CPU
        cpu_results = []
        for elf in sorted(elves):
            # Try to map test name to descriptor
            test_name = elf.stem
            descriptor = tracker.map_test_to_descriptor(test_name, all_descriptors_dict)
            descriptor_name = descriptor.get('name') if descriptor else test_name
            
            result = run_fvp_with_reporting(
                fvp_exe=fvp_exe, 
                elf=elf, 
                timeout=args.timeout_run,
                verbosity=verbosity, 
                extra_args=args.fvp_arg, 
                env=env,
                cpu=cpu,
                descriptor_name=descriptor_name
            )
            cpu_results.append(result)
            all_results.append(result)
            
            if result.status != TestStatus.PASS:
                any_fail = True
                if args.fail_fast:
                    if verbosity >= 1:
                        print("Stopping early due to failure (--fail-fast).")
                    break
        
        if any_fail and args.fail_fast:
            break
    
    # Generate report
    end_time = datetime.now()
    
    # Build descriptor_results: map each descriptor to its status and test result
    descriptor_results = {}
    
    # First, map test results to descriptors by descriptor_name
    test_result_map = {}
    for result in all_results:
        desc_name = result.descriptor_name or result.test_name
        # If we already have a result for this descriptor, keep the one with better status
        if desc_name not in test_result_map or result.status == TestStatus.PASS:
            test_result_map[desc_name] = result
    
    # Get set of descriptors that were actually generated/run
    # This includes descriptors that have:
    # 1. A test result (were run)
    # 2. A generated TFLite file (were generated)
    # 3. A build artifact (ELF file)
    # This ensures total_tests reflects only the filtered descriptors
    active_descriptors = set()
    
    # Add descriptors from test results
    for result in all_results:
        desc_name = result.descriptor_name or result.test_name
        active_descriptors.add(desc_name)
    
    # Add descriptors that have generated artifacts
    primary_build_dir = find_build_dir(cpus[0], source_dir) if cpus else find_build_dir("cortex-m55", source_dir)
    for desc_name in all_descriptors_dict.keys():
        # Check for TFLite file (generation stage)
        tflite_file = generated_tests_dir / desc_name / f"{desc_name}.tflite"
        # Check for model header (conversion stage) - new template system uses operator-specific names
        includes_dir = generated_tests_dir / desc_name / "includes"
        model_headers = list(includes_dir.glob(f"{desc_name}_*.h")) if includes_dir.exists() else []
        model_header_old = generated_tests_dir / desc_name / "includes" / f"{desc_name}_model.h"
        has_model_header = len(model_headers) > 0 or model_header_old.exists()
        # Check for ELF file (build stage)
        elf_path = primary_build_dir / "tests" / f"{desc_name}.elf"
        
        if tflite_file.exists() or has_model_header or elf_path.exists():
            active_descriptors.add(desc_name)
    
    # Process only active descriptors (those that were actually generated/run)
    for desc_name in active_descriptors:
        desc_content = all_descriptors_dict.get(desc_name)
        if not desc_content:
            # Descriptor not in loaded descriptors - skip
            continue
        
        test_result = test_result_map.get(desc_name)
        
        # Determine status
        status, failure_stage, failure_reason = tracker.determine_descriptor_status(
            descriptor_name=desc_name,
            test_result=test_result,
            build_dir=primary_build_dir,
            generated_tests_dir=generated_tests_dir
        )
        
        # Get descriptor path
        desc_path = tracker.get_descriptor_path(desc_name)
        
        # Create DescriptorResult
        descriptor_results[desc_name] = DescriptorResult(
            descriptor_name=desc_name,
            descriptor_path=desc_path,
            descriptor_content=desc_content,
            status=status,
            test_result=test_result,
            failure_stage=failure_stage,
            failure_reason=failure_reason
        )
    
    # Handle descriptors that might have been in test results but not in loaded descriptors
    # (edge case: test name doesn't match descriptor name exactly)
    for result in all_results:
        if result.descriptor_name and result.descriptor_name not in descriptor_results:
            # Try to find matching descriptor
            desc = tracker.map_test_to_descriptor(result.test_name, all_descriptors_dict)
            if desc:
                desc_name = desc.get('name', result.descriptor_name)
                if desc_name not in descriptor_results:
                    # Add to active descriptors if not already there
                    active_descriptors.add(desc_name)
                    primary_build_dir = find_build_dir(cpus[0], source_dir) if cpus else find_build_dir("cortex-m55", source_dir)
                    status, failure_stage, failure_reason = tracker.determine_descriptor_status(
                        descriptor_name=desc_name,
                        test_result=result,
                        build_dir=primary_build_dir,
                        generated_tests_dir=generated_tests_dir
                    )
                    desc_path = tracker.get_descriptor_path(desc_name)
                    descriptor_results[desc_name] = DescriptorResult(
                        descriptor_name=desc_name,
                        descriptor_path=desc_path,
                        descriptor_content=desc,
                        status=status,
                        test_result=result,
                        failure_stage=failure_stage,
                        failure_reason=failure_reason
                    )
    
    # Create test report with descriptor_results
    metadata = {
        "cpu": ",".join(cpus),
        "optimization": args.opt,
        "compiler": "arm-compiler" if args.use_arm_compiler else "gcc",
        "toolchain_file": str(toolchain_file),
        "cmsis5_path": str(cmsis5),
        "fvp_exe": str(fvp_exe),
        "downloads_dir": str(args.downloads_dir),
        "source_dir": str(source_dir),
        "git_sha": _get_git_sha(source_dir),
    }
    report = TestReport(
        run_id=f"run_{start_time.strftime('%Y%m%d_%H%M%S')}",
        start_time=start_time,
        end_time=end_time,
        cpu=",".join(cpus),
        descriptor_results=descriptor_results,
        all_descriptors=list(all_descriptors_dict.values()),
        project_root=source_dir,  # For making paths relative in JSON output
        metadata=metadata,
    )
    
    # Generate reports in requested formats (defaults to JSON if none specified)
    report_formats = getattr(args, 'report_formats', None) or ["json"]
    generated_files = {}
    if enable_reporting and generator:
        generated_files = generator.generate_reports(report, report_formats)
    verbosity = getattr(args, 'verbosity', 0)
    if not getattr(args, "quiet", False):
        print(
            f"Summary: total={report.total_tests} "
            f"passed={report.passed} failed={report.failed} skipped={report.skipped} "
            f"duration={report.duration:.2f}s"
        )
    if verbosity >= 1:
        for format_type, file_path in generated_files.items():
            print(f"{format_type.upper()} report generated: {file_path}")
    
    return all_results, not any_fail


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build and run helia-core kernels unit tests on FVP Corstone-300 (Python).")
    ap.add_argument("-c", "--cpu", default="cortex-m55", help="Comma-separated cores, e.g. cortex-m3,cortex-m55")
    ap.add_argument("-o", "--opt", default="-Ofast", help="Optimization level passed via CMSIS_OPTIMIZATION_LEVEL")
    ap.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], default=0,
                   help="Output verbosity level (0=minimal, 1=progress, 2=commands, 3=debug)")
    ap.add_argument("-b", "--no-build", action="store_true", help="Skip build (only run)")
    ap.add_argument("-r", "--no-run", action="store_true", help="Skip run (only build)")
    ap.add_argument("--no-clean-build", action="store_true", help="Do not delete existing build directories before build")
    ap.add_argument("-e", "--no-setup", action="store_true", help="Skip dependency setup")
    ap.add_argument("-a", "--use-arm-compiler", action="store_true", help="Use Arm Compiler (default: GCC)")
    ap.add_argument("-p", "--no-venv", action="store_true", help="(Kept for parity; no effect on CMake build)")
    ap.add_argument("-f", "--no-fvp-from-download", action="store_true", help="Do NOT use downloaded FVP; use FVP from PATH")
    ap.add_argument("-g", "--no-gcc-from-download", action="store_true", help="Do NOT use downloaded GCC; use system GCC")
    ap.add_argument("-u", "--ethos-path", type=Path, help="Override ethos-u-core-platform path")
    ap.add_argument("-C", "--cmsis5-path", type=Path, help="Override CMSIS_5 path")
    ap.add_argument("-D", "--cmake-def", action="append", default=[], help="Extra -DVAR=VAL for CMake (repeatable)")
    ap.add_argument("--downloads-dir", type=Path, default=DEFAULT_DL, help="Downloads directory (default: ./artifacts/downloads)")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE, help="CMake source dir (UnitTest root)")
    ap.add_argument("--generator", help="CMake generator (e.g. Ninja)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Parallel build jobs")
    ap.add_argument("--timeout-run", type=float, default=0.0, help="Per-test timeout in seconds (0 = none)")
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False, help="Stop on first failure")
    ap.add_argument("--fvp-arg", action="append", default=[], help="Extra args to pass to the FVP (repeatable)")
    # Reporting options
    ap.add_argument("--no-report", action="store_true", help="Disable comprehensive test reporting (enabled by default)")
    ap.add_argument("--report-formats", nargs="+", choices=["json", "html", "md", "junit"], default=["json"], 
                   help="Report formats to generate (default: json)")
    ap.add_argument("--report-dir", type=Path, default=ARTIFACTS_DIR / "reports", help="Directory to save reports (default: ./artifacts/reports)")
    ap.add_argument("--quiet", action="store_true", help="Quiet mode (no output)")
    args = ap.parse_args(argv)

    if not is_linux():
        die("This script supports Linux only (matching the original bash script).")

    # Optional setup of downloads
    if not args.no_setup:
        call_setup_dependencies(args.downloads_dir)

    # Resolve paths / env
    ctx = detect_paths(args)
    env = ctx["env"]
    toolchain_file = ctx["toolchain_file"]
    compiler_tag = ctx["compiler_tag"]
    fvp_exe = ctx["fvp_exe"]
    cmsis5 = ctx["cmsis5"]
    source_dir = args.source_dir.resolve()

    if not source_dir.exists():
        die(f"CMake source dir not found: {source_dir}")

    cpus = parse_cpus(args.cpu)
    
    enable_reporting = not args.no_report
    results, success = run_tests_with_reporting(
        cpus=cpus,
        source_dir=source_dir,
        toolchain_file=toolchain_file,
        cmsis5=cmsis5,
        fvp_exe=fvp_exe,
        args=args,
        env=env,
        enable_reporting=enable_reporting
    )
    
    verbosity = getattr(args, 'verbosity', 0)
    if success:
        if verbosity >= 1:
            print("\nAll requested builds/runs completed successfully.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
