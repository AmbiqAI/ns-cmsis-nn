#!/usr/bin/env python3
"""
Generate Unity test runners for embedded ML model tests.

This script scans a test directory tree for model headers (includes/*_model.h)
and generates a test_runner.c in each directory that contains model headers.
Each runner supports both FVP (correctness) and PMU/DWT (performance) backends,
selected at build time via TEST_BACKEND macro.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List


def discover_test_dirs(root: Path) -> List[Path]:
    """
    Discover all test directories containing includes/*_model.h files.

    A directory qualifies as a test directory if it contains an includes/
    subdirectory with at least one *_model.h file.

    Args:
        root: Root directory to scan recursively.

    Returns:
        List of Path objects representing test directories (sorted for determinism).
    """
    test_dirs = []
    for includes_api_dir in root.rglob("includes"):
        if not includes_api_dir.is_dir():
            continue
        # Check if this includes directory contains at least one *_model.h
        model_headers = list(includes_api_dir.glob("*.h"))
        if model_headers:
            # The test directory is the parent of includes
            test_dir = includes_api_dir.parent
            test_dirs.append(test_dir)
    
    return sorted(test_dirs)


def find_model_headers(test_dir: Path) -> List[Path]:
    """
    Find all model headers in the includes/ subdirectory of a test directory.

    Args:
        test_dir: The test directory to search.

    Returns:
        List of Path objects representing *_model.h files (sorted for determinism).
    """
    includes_api = test_dir / "includes"
    if not includes_api.is_dir():
        return []
    
    model_headers = list(includes_api.glob("*_test_case.h"))
    return sorted(model_headers)


def extract_model_prefix(header_path: Path) -> str:
    """
    Extract the model prefix from a model header filename.

    For a file named <prefix>_model.h, returns <prefix>.

    Args:
        header_path: Path to the model header file.

    Returns:
        The model prefix string.

    Raises:
        ValueError: If the filename does not match the *_model.h pattern.
    """
    filename = header_path.name
    if not filename.endswith("_test_case.h"):
        raise ValueError(f"Header filename does not match *_test_case.h pattern: {filename}")
    
    prefix = filename[:-len("_test_case.h")]
    return prefix


def rel_include(from_file: Path, to_header: Path) -> str:
    """
    Compute the relative include path from a source file to a header file.

    Args:
        from_file: Path to the source file that will contain the #include.
        to_header: Path to the header file to be included.

    Returns:
        Relative path string with POSIX-style separators (/).
    """
    from_dir = from_file.parent
    try:
        rel_path = to_header.resolve().relative_to(from_dir.resolve())
    except ValueError:
        # If relative_to fails, use os.path.relpath as fallback
        rel_path = Path(os.path.relpath(to_header.resolve(), from_dir.resolve()))
    
    # Ensure POSIX-style separators
    return rel_path.as_posix()


def render_runner(test_dir: Path, headers: List[Path], root_path: Path, runner_name: str) -> str:
    """
    Render the complete test_runner.c content for a test directory.

    Args:
        test_dir: The test directory containing the model headers.
        headers: List of model header file paths.
        root_path: The root directory that was scanned (for banner).
        runner_name: Name of the runner file (for computing relative includes).

    Returns:
        The complete C source code as a string.
    """
    runner_path = test_dir / runner_name
    
    # Generate include block
    include_lines = []
    for header in headers:
        rel_path = rel_include(runner_path, header)
        include_lines.append(f'#include "{rel_path}"')
    include_block = "\n".join(include_lines)
    
    # Generate model-specific blocks
    model_blocks = []
    run_test_lines = []
    
    for header in headers:
        prefix = extract_model_prefix(header)
        
        model_block = f"""
static {prefix}_model_context_t ctx__{prefix};

#if (TEST_BACKEND == BACKEND_FVP)
static int32_t {prefix}_run_fvp({prefix}_model_context_t* c)
{{
    return {prefix}_test_case_run();
}}
#endif

#if (TEST_BACKEND == BACKEND_PMU)
static int32_t {prefix}_run_pmu({prefix}_model_context_t* c, uint32_t* out_cycles)
{{
    int rc = pmu_init();
    TEST_ASSERT_EQUAL_MESSAGE(0, rc, "PMU/DWT not available on this build/board.");

    pmu_reset_start();
    int32_t status = {prefix}_model_run(c);
    uint32_t cycles = pmu_stop_read_cycles();

    if (out_cycles) {{ *out_cycles = cycles; }}
    return status;
}}
#endif

void test__{prefix}(void)
{{
    int32_t status = {prefix}_test_case_init();
    TEST_ASSERT_EQUAL_INT32_MESSAGE(0, status, "Model init failed: {prefix}");

#if (TEST_BACKEND == BACKEND_FVP)
    status = {prefix}_run_fvp(&ctx__{prefix});
    TEST_ASSERT_EQUAL_INT32_MESSAGE(0, status, "Model run (FVP) failed: {prefix}");
#elif (TEST_BACKEND == BACKEND_PMU)
    uint32_t cycles = 0;
    status = {prefix}_run_pmu(&ctx__{prefix}, &cycles);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(0, status, "Model run (PMU) failed: {prefix}");
    printf("[PERF] {prefix}: %lu cycles\\n", (unsigned long)cycles);
#endif
}}
"""
        model_blocks.append(model_block)
        run_test_lines.append(f"    RUN_TEST(test__{prefix});")
    
    model_blocks_str = "\n".join(model_blocks)
    run_test_lines_str = "\n".join(run_test_lines)
    
    # Render the complete template
    content = f"""// AUTOGENERATED FILE - DO NOT EDIT.
// Generated by scripts/generate_test_runners.py
// tests root: {root_path}
// output dir:  {test_dir}

#include "unity.h"
#include <stdint.h>
#include <stdio.h>

// One include per discovered model header in this directory:
{include_block}

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

// ---------- Backend selection ----------
#define BACKEND_FVP 1
#define BACKEND_PMU 2

#ifndef TEST_BACKEND
#  ifdef USING_FVP_CORSTONE_300
#    define TEST_BACKEND BACKEND_FVP
#  else
#    define TEST_BACKEND BACKEND_PMU
#  endif
#endif

#if (TEST_BACKEND != BACKEND_FVP) && (TEST_BACKEND != BACKEND_PMU)
#  error "Invalid TEST_BACKEND selection. Use BACKEND_FVP or BACKEND_PMU."
#endif

// ---------- PMU (DWT) helpers ----------
#if (TEST_BACKEND == BACKEND_PMU)
#  include "cmsis.h"

static inline int pmu_init(void)
{{
#if defined(DWT) && defined(CoreDebug) && \\
    defined(DWT_CTRL_CYCCNTENA_Msk) && defined(CoreDebug_DEMCR_TRCENA_Msk)
    if ((CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk) == 0U) {{
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    }}
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    return 0;
#else
    return -1;
#endif
}}

static inline void pmu_reset_start(void)
{{
#if defined(DWT)
    DWT->CYCCNT = 0U;
#endif
    __DSB(); __ISB();
}}

static inline uint32_t pmu_stop_read_cycles(void)
{{
    __DSB(); __ISB();
#if defined(DWT)
    return DWT->CYCCNT;
#else
    return 0U;
#endif
}}
#endif // BACKEND_PMU

// ---------- setUp/tearDown ----------
void setUp(void)
{{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
}}

void tearDown(void) {{}}

// ---------- Model-specific state & wrappers ----------
// For each discovered model, emit:
//   static {{MODEL_PREFIX}}_model_context_t ctx__{{MODEL_PREFIX}};
//   static int32_t {{MODEL_PREFIX}}_run_fvp(...);
//   static int32_t {{MODEL_PREFIX}}_run_pmu(...);
//   void test__{{MODEL_PREFIX}}(void) {{ ... }}
{model_blocks_str}

// ---------- Test main ----------
int main(void)
{{
    UNITY_BEGIN();
{run_test_lines_str}
    return UNITY_END();
}}
"""
    
    # Normalize newlines
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    return content


def write_if_changed(path: Path, content: str, dry_run: bool = False) -> bool:
    """
    Write content to file only if it differs from existing content.

    Uses atomic write via temporary file to ensure consistency.

    Args:
        path: Target file path.
        content: Content to write.
        dry_run: If True, do not actually write the file.

    Returns:
        True if the file was written (or would be written in dry-run mode), False if unchanged.
    """
    # Compute hash of new content
    new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    # Check if file exists and compute hash of existing content
    if path.exists():
        existing_content = path.read_text(encoding='utf-8')
        existing_hash = hashlib.sha256(existing_content.encode('utf-8')).hexdigest()
        
        if new_hash == existing_hash:
            return False  # No changes needed
    
    if dry_run:
        return True  # Would write
    
    # Write atomically via temporary file
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        temp_path.write_text(content, encoding='utf-8')
        temp_path.replace(path)
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise
    
    return True


def main() -> int:
    """
    Main entry point for the test runner generator.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Generate Unity test runners for embedded ML model tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("tests"),
        help="Root directory to scan for test directories (default: tests/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write files; print planned actions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing files (default: True, idempotent based on content)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print discovered headers and generated file paths",
    )
    parser.add_argument(
        "--runner-name",
        type=str,
        default="test_runner.c",
        help="Name of the generated runner file (default: test_runner.c)",
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    root_path = args.root.resolve()
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}", file=sys.stderr)
        return 1
    
    if not root_path.is_dir():
        print(f"Error: Root path is not a directory: {root_path}", file=sys.stderr)
        return 1
    
    # Discover test directories
    test_dirs = discover_test_dirs(root_path)
    
    if not test_dirs:
        print(f"Error: No test directories found in {root_path}", file=sys.stderr)
        print("(A test directory must contain pi/*_model.h)", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Discovered {len(test_dirs)} test director{'y' if len(test_dirs) == 1 else 'ies'}:")
        for test_dir in test_dirs:
            print(f"  {test_dir.relative_to(root_path)}")
        print()
    
    # Process each test directory
    written_count = 0
    skipped_count = 0
    
    for test_dir in test_dirs:
        headers = find_model_headers(test_dir)
        
        if not headers:
            # Should not happen given discovery logic, but be defensive
            continue
        
        if args.verbose:
            print(f"Processing: {test_dir.relative_to(root_path)}")
            for header in headers:
                prefix = extract_model_prefix(header)
                print(f"  - {header.name} (prefix: {prefix})")
        
        # Generate runner content
        content = render_runner(test_dir, headers, root_path, args.runner_name)
        runner_path = test_dir / args.runner_name
        
        # Write if changed
        was_written = write_if_changed(runner_path, content, dry_run=args.dry_run)
        
        if was_written:
            action = "Would write" if args.dry_run else "Wrote"
            print(f"{action}: {runner_path.relative_to(root_path)}")
            written_count += 1
        else:
            if args.verbose:
                print(f"Skipped (unchanged): {runner_path.relative_to(root_path)}")
            skipped_count += 1
        
        if args.verbose:
            print()
    
    # Print summary
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  Directories processed: {len(test_dirs)}")
    print(f"  Files written:         {written_count}")
    print(f"  Files skipped:         {skipped_count} (no changes)")
    if args.dry_run:
        print("  (Dry-run mode: no files were actually written)")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

