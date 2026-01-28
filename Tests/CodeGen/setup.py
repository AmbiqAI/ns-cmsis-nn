#!/usr/bin/env python3
"""
Setup script for CMSIS-AOT-Tester.

This script sets up the environment for CI builds and local development using uv
for Python dependencies.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_command(cmd: str) -> str | None:
    """Check if a command exists in PATH."""
    return shutil.which(cmd)


def install_uv() -> bool:
    """Install uv if not present."""
    print("uv not found. Installing uv...")
    try:
        # Use the official uv installer via curl | sh
        # This is the recommended way to install uv
        install_result = subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            check=True
        )
        
        # Add ~/.cargo/bin to PATH if uv was installed there
        cargo_bin = Path.home() / ".cargo" / "bin"
        if (cargo_bin / "uv").exists():
            os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ.get("PATH", "")
        
        # Verify installation
        if find_command("uv"):
            print("✓ uv installed successfully")
            return True
        else:
            print("✗ Failed to install uv", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install uv: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("✗ curl or sh not found. Please install uv manually:", file=sys.stderr)
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh", file=sys.stderr)
        return False


def ensure_uv() -> bool:
    """Ensure uv is available, install if needed."""
    if find_command("uv"):
        print("✓ uv found")
        return True
    
    return install_uv()


def setup_python_env(repo_root: Path, downloads_dir: Path) -> bool:
    """Set up Python virtual environment using uv sync."""
    print("\n=== Setting up Python Environment ===")
    
    pyproject_file = repo_root / "pyproject.toml"
    uv_lock_file = repo_root / "uv.lock"
    
    # Use uv sync if pyproject.toml exists
    if pyproject_file.exists():
        print("Using uv sync to install dependencies...")
        try:
            # uv sync automatically:
            # - Creates .venv if it doesn't exist
            # - Generates/updates uv.lock if needed
            # - Installs all dependencies
            subprocess.run(
                ["uv", "sync"],
                cwd=repo_root,
                check=True
            )
            print("✓ Python dependencies installed with uv sync")
            
            # Note: uv sync creates .venv in the repo root, not in downloads_dir
            # This is the standard uv behavior
            venv_path = repo_root / ".venv"
            if venv_path.exists():
                print(f"✓ Virtual environment created at {venv_path}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to sync dependencies: {e}", file=sys.stderr)
            # Fallback to manual installation
            print("Falling back to manual installation...", file=sys.stderr)
    
    # Fallback: Manual venv creation and installation
    venv_dir = downloads_dir / "cmsis_nn_venv"
    requirements_file = repo_root / "requirements.txt"
    
    if not venv_dir.exists():
        print("Creating virtual environment with uv...")
        try:
            subprocess.run(
                ["uv", "venv", str(venv_dir)],
                check=True
            )
            print("✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create virtual environment: {e}", file=sys.stderr)
            return False
    else:
        print(f"Virtual environment already exists at {venv_dir}")
    
    # Determine Python executable
    if sys.platform.startswith('win'):
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
    
    if not python_exe.exists():
        print(f"✗ Could not find Python in virtual environment", file=sys.stderr)
        return False
    
    # Install dependencies using uv pip install
    print("Installing Python dependencies with uv...")
    try:
        if pyproject_file.exists():
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python_exe), "-e", str(repo_root)],
                cwd=repo_root,
                check=True
            )
        elif requirements_file.exists():
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python_exe), "-r", str(requirements_file)],
                cwd=repo_root,
                check=True
            )
        else:
            print("✗ Neither pyproject.toml nor requirements.txt found", file=sys.stderr)
            return False
        
        print("✓ Python dependencies installed with uv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}", file=sys.stderr)
        return False


def setup_build_deps(repo_root: Path, downloads_dir: Path, skip: bool = False) -> bool:
    """Set up build dependencies (ARM GCC, CMSIS-5, etc.) using uv run."""
    if skip:
        print("Skipping build dependencies setup")
        return True
    
    print("\n=== Setting up Build Dependencies ===")
    
    setup_script = repo_root / "cmsis_nn_tools" / "scripts" / "setup_dependencies.py"
    
    if not setup_script.exists():
        print(f"Setup script not found at {setup_script}. Skipping build dependencies.")
        return True
    
    # Use uv run to execute the script in the project environment
    # This ensures we use the correct Python and dependencies from .venv
    try:
        subprocess.run(
            ["uv", "run", "python", str(setup_script),
             "--downloads-dir", str(downloads_dir),
             "--skip-python"],
            cwd=repo_root,
            check=False  # Don't fail if build deps have issues
        )
        print("✓ Build dependencies setup complete")
        return True
    except FileNotFoundError:
        # Fallback to direct Python execution if uv run fails
        # Check for .venv first (uv sync standard), then fallback to downloads_dir
        venv_dir = repo_root / ".venv"
        if not venv_dir.exists():
            venv_dir = downloads_dir / "cmsis_nn_venv"
        
        if sys.platform.startswith('win'):
            python_exe = venv_dir / "Scripts" / "python.exe" if venv_dir.exists() else None
        else:
            python_exe = venv_dir / "bin" / "python" if venv_dir.exists() else None
        
        if not python_exe or not python_exe.exists():
            python_exe = sys.executable
        
        try:
            subprocess.run(
                [str(python_exe), str(setup_script),
                 "--downloads-dir", str(downloads_dir),
                 "--skip-python"],
                check=False
            )
            print("✓ Build dependencies setup complete")
            return True
        except Exception as e:
            print(f"Warning: Build dependencies setup had issues: {e}", file=sys.stderr)
            return True  # Continue anyway
    except Exception as e:
        print(f"Warning: Build dependencies setup had issues: {e}", file=sys.stderr)
        return True  # Continue anyway


def setup_environment(repo_root: Path, downloads_dir: Path) -> None:
    """Set up environment variables and create environment file."""
    print("\n=== Environment Configuration ===")
    
    # Check for .venv (uv sync standard location) or fallback to downloads_dir
    venv_dir = repo_root / ".venv"
    if not venv_dir.exists():
        venv_dir = downloads_dir / "cmsis_nn_venv"
    
    arm_gcc_path = downloads_dir / "arm_gcc_download" / "bin"
    
    # Create environment file
    env_file = repo_root / ".ci_env"
    with open(env_file, "w") as f:
        f.write("# CI Environment Configuration\n")
        f.write("# Source this file to activate the environment: source .ci_env\n\n")
        
        if sys.platform.startswith('win'):
            f.write(f'set PATH={venv_dir}\\Scripts;%PATH%\n')
            if arm_gcc_path.exists():
                f.write(f'set PATH={arm_gcc_path};%PATH%\n')
        else:
            f.write(f'export PATH="{venv_dir}/bin:$PATH"\n')
            if arm_gcc_path.exists():
                f.write(f'export PATH="{arm_gcc_path}:$PATH"\n')
    
    print("✓ Environment configured")
    print(f"\nEnvironment file created at: {env_file}")
    print("To activate the environment, run:")
    if sys.platform.startswith('win'):
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")
    print(f"\nOr use uv run :")
    print(f"  uv run python cmsis_nn_tools/cli.py --help")
    print(f"\nOr source the CI environment file:")
    print(f"  source {env_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup CMSIS-AOT-Tester environment using uv",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=None,
        help="Directory for downloads (default: ./downloads)"
    )
    parser.add_argument(
        "--skip-build-deps",
        action="store_true",
        help="Skip build dependencies setup"
    )
    
    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).resolve().parent
    downloads_dir = args.downloads_dir or (repo_root / "downloads")
    
    print("=== CMSIS-AOT-Tester Setup ===")
    print(f"Repository root: {repo_root}")
    print(f"Downloads directory: {downloads_dir}")
    print()
    
    # Ensure uv is available
    if not ensure_uv():
        print("\n✗ Setup failed: uv is required", file=sys.stderr)
        print("Please install uv manually:", file=sys.stderr)
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh", file=sys.stderr)
        return 1
    
    # Create downloads directory
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Python environment
    if not setup_python_env(repo_root, downloads_dir):
        print("\n✗ Setup failed: Could not set up Python environment", file=sys.stderr)
        return 1
    
    # Setup build dependencies
    if not setup_build_deps(repo_root, downloads_dir, args.skip_build_deps):
        print("\n⚠ Warning: Build dependencies setup had issues", file=sys.stderr)
        # Continue anyway
    
    # Setup environment
    setup_environment(repo_root, downloads_dir)
    
    print("\n=== Setup Complete ===")
    # Check which venv was created
    venv_dir = repo_root / ".venv"
    if venv_dir.exists():
        print(f"Virtual environment: {venv_dir} (created by uv sync)")
    else:
        print(f"Virtual environment: {downloads_dir / 'cmsis_nn_venv'}")
    print(f"Downloads directory: {downloads_dir}")
    print("\nNext steps:")
    print("  # Run commands using uv run (recommended)")
    print("  uv run python cmsis_nn_tools/cli.py --help")
    print("\n  # Or activate the virtual environment")
    if (repo_root / ".venv").exists():
        print("  source .venv/bin/activate")
    else:
        print(f"  source {downloads_dir / 'cmsis_nn_venv'}/bin/activate")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

