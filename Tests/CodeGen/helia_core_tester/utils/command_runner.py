"""
Command execution utilities.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from helia_core_tester.core.logging import get_logger


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = False,
    verbosity: int = 0,
    env: Optional[dict] = None,
    stream_output: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a command with proper error handling and output.
    
    Args:
        cmd: Command to run as list of strings
        cwd: Working directory (default: current directory)
        check: Whether to raise exception on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        verbosity: Verbosity level (0=minimal, 1=progress, 2=commands, 3=debug)
        env: Optional environment variables dict
        stream_output: If True, stream subprocess output directly to the console
        
    Returns:
        CompletedProcess object
        
    Raises:
        subprocess.CalledProcessError: If command fails and check=True
        FileNotFoundError: If command not found
    """
    logger = get_logger(__name__)
    
    # Control output based on verbosity
    # Level 0-1: Suppress subprocess output (capture but don't print)
    # Level 2: Print subprocess output for commands
    # Level 3: Print all subprocess output with timestamps
    show_output = verbosity >= 2
    show_debug = verbosity >= 3
    
    if show_debug:
        logger.debug(f"Running: {' '.join(cmd)}")
        if cwd:
            logger.debug(f"Directory: {cwd}")
    elif verbosity >= 2:
        logger.info(f"Running: {' '.join(cmd)}")
        if cwd:
            logger.info(f"Directory: {cwd}")
    
    try:
        if stream_output:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=check,
                env=env,
                text=True
            )
        else:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=check,
                env=env,
                capture_output=capture_output or not show_output,  # Capture output if not showing or explicitly requested
                text=True
            )
        
        # Print output at verbosity 2+
        if show_output and result.stdout and not capture_output and not stream_output:
            print(result.stdout, end='')
        if show_output and result.stderr and not capture_output and not stream_output:
            print(result.stderr, end='', file=sys.stderr)
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if (capture_output or not show_output) and e.stdout:
            if show_debug:
                logger.debug(f"stdout: {e.stdout}")
            elif verbosity >= 1:
                logger.info(f"stdout: {e.stdout}")
        if (capture_output or not show_output) and e.stderr:
            if show_debug:
                logger.debug(f"stderr: {e.stderr}")
            elif verbosity >= 1:
                logger.info(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        raise
