"""
Command-line interface for CMSIS-NN Tools.

This module provides a subcommand-based CLI using Typer.
"""

import sys
from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.core.config import Config
from helia_core_tester.core.logging import setup_logger
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps import (
    GenerateStep,
    RunnersStep,
    BuildStep,
    RunStep,
    CleanStep,
)

app = typer.Typer(
    name="cmsis-nn-tools",
    help="CMSIS-NN testing toolkit - Generate, build, and run tests for CMSIS-NN kernels",
    add_completion=False,
)

# Common options that can be shared
def common_options(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
) -> tuple:
    """Common options for subcommands."""
    return cpu, verbosity, dry_run, project_root


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
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    return config


@app.command()
def generate(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Generate TFLite models and template C/H files."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
        op_filter=op,
        dtype_filter=dtype,
        name_filter=name,
        limit=limit,
        seed=seed,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    step = GenerateStep(config)
    result = step.execute()
    
    if result.success:
        typer.echo("✓ Generation completed successfully")
        sys.exit(0)
    elif result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    else:
        typer.echo(f"✗ Generation failed: {result.message}", err=True)
        sys.exit(1)


@app.command()
def runners(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Generate Unity test runners."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    step = RunnersStep(config)
    result = step.execute()
    
    if result.success:
        typer.echo("✓ Test runners generated successfully")
        sys.exit(0)
    elif result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    else:
        typer.echo(f"✗ Runner generation failed: {result.message}", err=True)
        sys.exit(1)


@app.command()
def build(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Build test executables using CMake."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
        optimization=opt,
        jobs=jobs,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    step = BuildStep(config)
    result = step.execute()
    
    if result.success:
        typer.echo(f"✓ Build completed successfully for {cpu}")
        sys.exit(0)
    elif result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    else:
        typer.echo(f"✗ Build failed: {result.message}", err=True)
        sys.exit(1)


@app.command()
def run(
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run tests on FVP simulator."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
        timeout=timeout,
        fail_fast=not no_fail_fast,
        enable_reporting=not no_report,
        report_formats=report_formats,
        report_dir=report_dir,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    step = RunStep(config)
    result = step.execute()
    
    if result.success:
        typer.echo("✓ All tests completed successfully")
        sys.exit(0)
    elif result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    else:
        typer.echo(f"✗ Test execution failed: {result.message}", err=True)
        sys.exit(1)


@app.command()
def full(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip TFLite generation"),
    skip_conversion: bool = typer.Option(False, "--skip-conversion", help="Skip TFLite to C conversion"),
    skip_runners: bool = typer.Option(False, "--skip-runners", help="Skip test runner generation"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip FVP build"),
    skip_run: bool = typer.Option(False, "--skip-run", help="Skip FVP test execution"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: int = typer.Option(0, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run the complete pipeline (generate → runners → build → run)."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
        op_filter=op,
        dtype_filter=dtype,
        name_filter=name,
        limit=limit,
        seed=seed,
        optimization=opt,
        jobs=jobs,
        timeout=timeout,
        fail_fast=not no_fail_fast,
        skip_generation=skip_generation,
        skip_conversion=skip_conversion,
        skip_runners=skip_runners,
        skip_build=skip_build,
        skip_run=skip_run,
        enable_reporting=not no_report,
        report_formats=report_formats,
        report_dir=report_dir,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    pipeline = FullTestPipeline(config)
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
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Clean build artifacts and reports."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        project_root=project_root,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    step = CleanStep(config)
    result = step.execute()
    
    if result.success:
        typer.echo(f"✓ {result.message}")
        sys.exit(0)
    elif result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    else:
        typer.echo(f"✗ Clean failed: {result.message}", err=True)
        sys.exit(1)


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
        "descriptors": "Test descriptors",
        "GeneratedTests": "Generated tests (will be created)",
        "downloads": "Downloads (will be created)",
    }
    
    for dir_name, description in key_dirs.items():
        dir_path = repo_root / dir_name
        if dir_path.exists() or dir_name in ["GeneratedTests", "downloads"]:
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
