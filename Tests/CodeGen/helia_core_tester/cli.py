"""
Command-line interface for CMSIS-NN Tools.

This module provides a subcommand-based CLI using Typer.
"""

import sys
import shutil
from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.core.config import Config
from helia_core_tester.core.logging import setup_logger
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps import (
    GenerateStep,
    BuildStep,
    RunStep,
    CleanStep,
)

app = typer.Typer(
    name="helia_core_tester",
    help="CMSIS-NN testing toolkit - Generate, build, and run tests for CMSIS-NN kernels",
    add_completion=False,
)


def _print_plan_item(plan_item) -> None:
    typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
    for cmd in plan_item.commands:
        typer.echo(f"   cmd: {' '.join(cmd)}")
    if plan_item.outputs:
        outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
        if outputs:
            typer.echo(f"   outputs: {outputs}")
    if plan_item.details:
        details = ", ".join(f"{k}={v}" for k, v in plan_item.details.items() if v is not None)
        if details:
            typer.echo(f"   details: {details}")

def get_config(
    cpu: str = "cortex-m55",
    verbosity: int = 0,
    dry_run: bool = False,
    project_root: Optional[Path] = None,
    **kwargs
) -> Config:
    """Create and configure Config object."""
    config = Config()
    if project_root:
        config.project_root = project_root
    config.cpu = cpu
    config.verbosity = verbosity
    config.dry_run = dry_run
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config


def run_step_exit(step, config: Config, success_msg: str, failure_prefix: Optional[str] = None) -> None:
    """Run a step, echo result, and exit with appropriate code."""
    setup_logger(verbosity=config.verbosity)
    result = step.execute()
    if result.success:
        typer.echo(success_msg if success_msg else result.message)
        sys.exit(0)
    if result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    msg = f"{failure_prefix}: {result.message}" if failure_prefix else result.message
    typer.echo(f"✗ {msg}", err=True)
    sys.exit(1)


@app.command()
def generate(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(500, help="Random seed for test generation"),
    no_clean_generated: bool = typer.Option(False, "--no-clean-generated", help="Keep existing generated tests"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Generate TFLite models and template C/H files."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        op_filter=op, dtype_filter=dtype, name_filter=name, limit=limit, seed=seed,
        clean_generated_tests=not no_clean_generated,
    )
    if config.plan:
        plan_item = GenerateStep(config).plan()
        _print_plan_item(plan_item)
        sys.exit(0)
    run_step_exit(
        GenerateStep(config), config,
        "✓ Generation completed successfully",
        failure_prefix="Generation failed",
    )




@app.command()
def build(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    no_clean_build: bool = typer.Option(False, "--no-clean-build", help="Keep existing build directory"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Build test executables using CMake."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        optimization=opt, jobs=jobs, clean_build=not no_clean_build,
    )
    if config.plan:
        plan_item = BuildStep(config).plan()
        _print_plan_item(plan_item)
        sys.exit(0)
    run_step_exit(
        BuildStep(config), config,
        f"✓ Build completed successfully for {cpu}",
        failure_prefix="Build failed",
    )


@app.command()
def run(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run tests on FVP simulator."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        timeout=timeout, fail_fast=not no_fail_fast, enable_reporting=not no_report,
        report_formats=report_formats, report_dir=report_dir,
    )
    if config.plan:
        plan_item = RunStep(config).plan()
        _print_plan_item(plan_item)
        sys.exit(0)
    run_step_exit(
        RunStep(config), config,
        "✓ All tests completed successfully",
        failure_prefix="Test execution failed",
    )


@app.command()
def full(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(500, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    no_clean_generated: bool = typer.Option(False, "--no-clean-generated", help="Keep existing generated tests"),
    no_clean_build: bool = typer.Option(False, "--no-clean-build", help="Keep existing build directory"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip TFLite generation"),
    skip_conversion: bool = typer.Option(False, "--skip-conversion", help="Skip TFLite to C conversion"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip FVP build"),
    skip_run: bool = typer.Option(False, "--skip-run", help="Skip FVP test execution"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run the complete pipeline (generate → build → run)."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        plan=plan,
        project_root=project_root,
        op_filter=op,
        dtype_filter=dtype,
        name_filter=name,
        limit=limit,
        seed=seed,
        optimization=opt,
        jobs=jobs,
        clean_generated_tests=not no_clean_generated,
        clean_build=not no_clean_build,
        timeout=timeout,
        fail_fast=not no_fail_fast,
        skip_generation=skip_generation,
        skip_conversion=skip_conversion,
        skip_build=skip_build,
        skip_run=skip_run,
        enable_reporting=not no_report,
        report_formats=report_formats,
        report_dir=report_dir,
    )
    
    pipeline = FullTestPipeline(config)
    if config.plan:
        pipeline.print_plan()
        sys.exit(0)
    success = pipeline.run()
    
    if success:
        typer.echo("✓ Pipeline completed successfully")
        sys.exit(0)
    else:
        typer.echo("✗ Pipeline failed", err=True)
        sys.exit(1)


@app.command()
def clean(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Clean build artifacts and reports."""
    config = get_config(cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root)
    if config.plan:
        plan_item = CleanStep(config).plan()
        _print_plan_item(plan_item)
        sys.exit(0)
    run_step_exit(CleanStep(config), config, "", failure_prefix="Clean failed")


@app.command(name="clean-all")
def clean_all(
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Remove all artifacts (generated tests, downloads, reports, builds)."""
    config = get_config(verbosity=verbosity, dry_run=dry_run, project_root=project_root)
    artifacts_root = config.project_root / "artifacts"
    if not artifacts_root.exists():
        typer.echo("No artifacts directory to clean.")
        sys.exit(0)

    if dry_run:
        typer.echo(f"DRY RUN: Would remove {artifacts_root}")
        sys.exit(0)

    if verbosity >= 1:
        typer.echo(f"Removing artifacts directory: {artifacts_root}")
    shutil.rmtree(artifacts_root, ignore_errors=True)
    typer.echo("✓ Clean-all completed")


@app.command()
def doctor(
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run preflight checks (verify tools, paths, permissions)."""
    typer.echo("Running preflight checks...")
    
    # Check repository root
    try:
        from .core.discovery import find_repo_root
        if project_root:
            repo_root = Path(project_root).resolve()
        else:
            repo_root = find_repo_root()
        typer.echo(f"✓ Repository root: {repo_root}")
    except Exception as e:
        typer.echo(f"✗ Repository root not found: {e}", err=True)
        sys.exit(1)
    
    # Check required tools
    import shutil
    
    tools = {
        "python3": "Python interpreter",
        "pytest": "pytest (for test generation)",
        "cmake": "CMake (for building)",
    }
    
    all_ok = True
    for tool, description in tools.items():
        if shutil.which(tool):
            typer.echo(f"✓ {tool} found ({description})")
        else:
            typer.echo(f"✗ {tool} not found ({description})", err=True)
            all_ok = False
    
    # Check key directories
    key_dirs = {
        "assets/descriptors": "Test descriptors",
        "artifacts/generated_tests": "Generated tests (will be created)",
        "artifacts/downloads": "Downloads (will be created)",
    }
    
    for dir_name, description in key_dirs.items():
        dir_path = repo_root / dir_name
        if dir_path.exists() or dir_name in ["artifacts/generated_tests", "artifacts/downloads"]:
            typer.echo(f"✓ {dir_name}/ exists or will be created ({description})")
        else:
            typer.echo(f"⚠ {dir_name}/ not found ({description})", err=True)
    
    if all_ok:
        typer.echo("\n✓ All preflight checks passed")
        sys.exit(0)
    else:
        typer.echo("\n✗ Some preflight checks failed", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
