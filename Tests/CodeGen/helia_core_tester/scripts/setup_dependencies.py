#!/usr/bin/env python3
"""
setup_dependencies.py — Download and setup build dependencies.

This script downloads and sets up the dependencies needed for building and running
CMSIS-NN unit tests, similar to the Setup_Environment() function in build_and_run_tests.sh.

Dependencies downloaded:
- Corstone300 FVP (Fixed Virtual Platform)
- ARM GCC toolchain
- CMSIS-5 library
- Ethos-U core platform
- Python virtual environment with requirements

Usage:
    python3 scripts/setup_dependencies.py [--downloads-dir DOWNLOADS_DIR] [--force]
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from helia_core_tester.core.discovery import find_repo_root


def get_architecture() -> str:
    """Get the system architecture (x86_64 or aarch64)."""
    machine = platform.machine().lower()
    if machine in ['x86_64', 'amd64']:
        return 'x86_64'
    elif machine in ['aarch64', 'arm64']:
        return 'aarch64'
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")


def get_os() -> str:
    """Get the operating system."""
    system = platform.system().lower()
    if system == 'linux':
        return 'linux'
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")




def download_file(url: str, dest_path: Path, description: str) -> None:
    """Download a file from URL to destination path."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    
    try:
        with urllib.request.urlopen(url) as response:
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        print("Downloaded successfully")
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def extract_tar_gz(archive_path: Path, extract_to: Path, strip_components: int = 0) -> None:
    """Extract a .tar.gz or .tar.xz file."""
    print(f"Extracting {archive_path.name} to {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(archive_path, 'r:*') as tar:
        # Get all members
        members = tar.getmembers()
        
        if strip_components > 0:
            # Remove the specified number of path components
            for member in members:
                path_parts = member.name.split('/')
                if len(path_parts) > strip_components:
                    member.name = '/'.join(path_parts[strip_components:])
                else:
                    member.name = '.'
        
        # Extract all members at once (more robust for complex archives)
        try:
            tar.extractall(extract_to, members=members)
        except (OSError, IOError) as e:
            print(f"Warning: Some files could not be extracted: {e}")
            # Try extracting members one by one for better error handling
            for member in members:
                try:
                    tar.extract(member, extract_to)
                except (OSError, IOError) as member_error:
                    if member.issym() or member.islnk():
                        print(f"Warning: Skipping symlink {member.name}: {member_error}")
                    else:
                        print(f"Warning: Skipping {member.name}: {member_error}")
                    continue
    
    print("Extracted successfully")


def run_command(cmd: list[str], cwd: Optional[Path] = None, description: str = "") -> None:
    """Run a command and handle errors."""
    if description:
        print(f"Running: {description}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        raise


def setup_corstone300(downloads_dir: Path, force: bool = False) -> None:
    """Download and setup Corstone300 FVP (mirrors the bash flow)."""
    corstone_dir = downloads_dir / "corstone300_download"

    # Existing install?
    if corstone_dir.exists() and not force:
        print("Corstone300 already installed. If you wish to install a new version, please delete the old folder.")
        return

    # Force re-install
    if force and corstone_dir.exists():
        print("Removing existing Corstone300 installation...")
        shutil.rmtree(corstone_dir)

    arch = get_architecture()
    if arch == 'x86_64':
        corstone_url = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz"
    elif arch == 'aarch64':
        corstone_url = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz"
    else:
        raise RuntimeError(f"Unsupported architecture for Corstone300: {arch}")

    # Work in temp dirs like the bash script
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "corstone300.tgz"

        # Download (equivalent to: wget -q "${CORSTONE_URL}" -O "${TEMPFILE}")
        try:
            download_file(corstone_url, archive_file, "Corstone300")
        except Exception:
            # Match the bash error message
            raise RuntimeError("Download Corstone300 failed!")

        # Extract (equivalent to: tar -C ${TEMPDIR} -xzf ${TEMPFILE})
        # Use system 'tar' to mirror bash behavior closely.
        run_command(
            ["tar", "-C", str(temp_path), "-xzf", str(archive_file)],
            description=f"Extracting {archive_file.name}"
        )

        # Find the installer script (bash expects ${TEMPDIR}/FVP_Corstone_SSE-300.sh)
        installer_script = None
        for item in temp_path.iterdir():
            if item.name.startswith("FVP_Corstone_SSE-300") and item.suffix == ".sh":
                installer_script = item
                break
        if not installer_script:
            raise RuntimeError("Could not find Corstone300 installer script after extraction")

        # Ensure destination exists (equivalent to: mkdir ${WORKING_DIR}/corstone300_download)
        corstone_dir.mkdir(parents=True, exist_ok=True)

        # Make sure script is executable (some environments lose +x)
        try:
            current_mode = installer_script.stat().st_mode
            installer_script.chmod(current_mode | 0o111)
        except Exception:
            pass  # best-effort

        # Run installer (equiv: ${TEMPDIR}/FVP_Corstone_SSE-300.sh --i-agree... -d <dir>)
        print("Installing Corstone300...")
        run_command(
            [
                str(installer_script),
                "--i-agree-to-the-contained-eula",
                "--no-interactive",
                "-q",
                "-d", str(corstone_dir),
            ],
            description="Installing Corstone300 FVP"
        )

    print("Corstone300 setup complete")



def setup_arm_gcc(downloads_dir: Path, force: bool = False) -> None:
    """Download and setup ARM GCC toolchain."""
    gcc_dir = downloads_dir / "arm_gcc_download"
    
    if gcc_dir.exists() and not force:
        print("Arm GCC already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and gcc_dir.exists():
        print("Removing existing ARM GCC installation...")
        shutil.rmtree(gcc_dir)
    
    arch = get_architecture()
    if arch == 'x86_64':
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz"
    elif arch == 'aarch64':
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz"
    else:
        raise RuntimeError(f"Unsupported architecture for ARM GCC: {arch}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "arm_gcc.tar.xz"
        
        download_file(gcc_url, archive_file, "ARM GCC toolchain")
        
        # Extract to temporary directory first
        temp_extract = temp_path / "extracted"
        extract_tar_gz(archive_file, temp_extract, strip_components=0)
        
        # Find the toolchain directory (should be the only subdirectory)
        toolchain_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
        if not toolchain_dirs:
            raise RuntimeError("Could not find toolchain directory in archive")
        
        toolchain_dir = toolchain_dirs[0]
        
        # Move contents to final destination
        print(f"Moving toolchain from {toolchain_dir.name} to {gcc_dir}")
        shutil.move(str(toolchain_dir), str(gcc_dir))
    
    print("ARM GCC setup complete")


def setup_cmsis5(downloads_dir: Path, force: bool = False) -> None:
    """Clone CMSIS-5 repository."""
    cmsis5_dir = downloads_dir / "CMSIS_5"
    
    # Check if it's a valid installation (has .git or CMSIS subdirectory)
    is_valid_install = cmsis5_dir.exists() and (
        (cmsis5_dir / ".git").exists() or 
        (cmsis5_dir / "CMSIS").exists()
    )
    
    if is_valid_install and not force:
        print("CMSIS-5 already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and cmsis5_dir.exists():
        print("Removing existing CMSIS-5 installation...")
        shutil.rmtree(cmsis5_dir)
    elif cmsis5_dir.exists() and not is_valid_install:
        # Directory exists but is empty/invalid, remove it
        print("Removing invalid/empty CMSIS-5 directory...")
        shutil.rmtree(cmsis5_dir)
    
    print("Cloning CMSIS-5...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://github.com/ARM-software/CMSIS_5.git"],
        cwd=downloads_dir,
        description="Cloning CMSIS-5"
    )
    
    print("CMSIS-5 setup complete")


def setup_ethos_u_platform(downloads_dir: Path, force: bool = False) -> None:
    """Clone Ethos-U core platform repository."""
    ethos_dir = downloads_dir / "ethos-u-core-platform"
    
    # Check if it's a valid installation (has .git or core_platform subdirectory)
    is_valid_install = ethos_dir.exists() and (
        (ethos_dir / ".git").exists() or 
        (ethos_dir / "core_platform").exists()
    )
    
    if is_valid_install and not force:
        print("Ethos-U core platform already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and ethos_dir.exists():
        print("Removing existing Ethos-U core platform installation...")
        shutil.rmtree(ethos_dir)
    elif ethos_dir.exists() and not is_valid_install:
        # Directory exists but is empty/invalid, remove it
        print("Removing invalid/empty Ethos-U core platform directory...")
        shutil.rmtree(ethos_dir)
    
    print("Cloning Ethos-U core platform...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git"],
        cwd=downloads_dir,
        description="Cloning Ethos-U core platform"
    )
    
    print("Ethos-U core platform setup complete")


def setup_python_venv(downloads_dir: Path, force: bool = False) -> None:
    """Setup Python virtual environment using uv (required)."""
    venv_dir = downloads_dir / "cmsis_nn_venv"
    
    if venv_dir.exists() and not force:
        print("Python venv already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and venv_dir.exists():
        print("Removing existing Python venv...")
        shutil.rmtree(venv_dir)
    
    print("Setting up Python virtual environment with uv...")
    
    # Check if uv is available (required)
    uv_available = shutil.which("uv") is not None
    if not uv_available:
        raise RuntimeError(
            "uv is required but not found. Please install uv:\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    
    # Create virtual environment with uv
    run_command(
        ["uv", "venv", str(venv_dir)],
        description="Creating Python virtual environment with uv"
    )
    
    # Determine Python executable in venv
    if sys.platform.startswith('win'):
        python_cmd = str(venv_dir / "Scripts" / "python.exe")
    else:
        python_cmd = str(venv_dir / "bin" / "python")
    
    # Install dependencies using uv pip
    repo_root = find_repo_root()
    requirements_file = repo_root / "requirements.txt"
    pyproject_file = repo_root / "pyproject.toml"
    
    if pyproject_file.exists():
        print("Installing Python requirements from pyproject.toml with uv...")
        run_command(
            ["uv", "pip", "install", "--python", python_cmd, "-e", str(repo_root)],
            description="Installing Python requirements with uv"
        )
    elif requirements_file.exists():
        print("Installing Python requirements from requirements.txt with uv...")
        run_command(
            ["uv", "pip", "install", "--python", python_cmd, "-r", str(requirements_file)],
            description="Installing Python requirements with uv"
        )
    else:
        raise RuntimeError("Neither pyproject.toml nor requirements.txt found")
    
    print("✓ Python virtual environment setup complete")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and setup build dependencies for CMSIS-NN unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/setup_dependencies.py
    python3 scripts/setup_dependencies.py --downloads-dir ./my_downloads
    python3 scripts/setup_dependencies.py --force
        """
    )
    
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("artifacts/downloads"),
        help="Directory to store downloaded dependencies (default: artifacts/downloads)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reinstall all dependencies"
    )
    parser.add_argument(
        "--skip-corstone",
        action="store_true",
        help="Skip Corstone300 FVP download"
    )
    parser.add_argument(
        "--skip-gcc",
        action="store_true",
        help="Skip ARM GCC toolchain download"
    )
    parser.add_argument(
        "--skip-cmsis5",
        action="store_true",
        help="Skip CMSIS-5 download"
    )
    parser.add_argument(
        "--skip-ethos",
        action="store_true",
        help="Skip Ethos-U core platform download"
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python virtual environment setup"
    )
    
    args = parser.parse_args()
    
    # Validate system
    try:
        get_os()
        get_architecture()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    
    # Create downloads directory
    args.downloads_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Setting up CMSIS-NN build dependencies")
    print(f"Downloads directory: {args.downloads_dir}")
    print(f"Force mode: {args.force}")
    print("=" * 80)
    
    try:
        if not args.skip_corstone:
            setup_corstone300(args.downloads_dir, args.force)
            print()
        
        if not args.skip_gcc:
            setup_arm_gcc(args.downloads_dir, args.force)
            print()
        
        if not args.skip_cmsis5:
            setup_cmsis5(args.downloads_dir, args.force)
            print()
        
        if not args.skip_ethos:
            setup_ethos_u_platform(args.downloads_dir, args.force)
            print()
        
        if not args.skip_python:
            setup_python_venv(args.downloads_dir, args.force)
            print()
        
        
        print("=" * 80)
        print("All dependencies setup complete!")
        print("=" * 80)
        
        # Print summary
        print("\nInstalled dependencies:")
        for item in args.downloads_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
