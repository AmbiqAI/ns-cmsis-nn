#!/usr/bin/env python3
"""
build_deploy_view.py — Build → Deploy → View helper.

Examples:
  python3 python_scripts/build_deploy_view.py --env TEST_DIR=tests/add_add_int8
  python3 python_scripts/build_deploy_view.py -j 8 --clean --env BOARD=apollo5b --env TOOLCHAIN=arm-none-eabi
  python3 python_scripts/build_deploy_view.py --timeout-build 600 --verbose
"""

from __future__ import annotations
import argparse, os, sys, time, shlex, subprocess
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager

def find_make_root(start: Path) -> Path:
    """Ascend from start to filesystem root to find a directory containing Makefile."""
    p = start.resolve()
    for ancestor in [p] + list(p.parents):
        if (ancestor / "Makefile").exists():
            return ancestor
    raise FileNotFoundError("Makefile not found. Run in repo or provide --root.")

def merge_env(base: Dict[str, str], kv: List[str]) -> Dict[str, str]:
    """Merge KEY=VAL tokens into a copy of base; later overrides earlier."""
    env = dict(base)
    for item in kv:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VAL, got: {item}")
        k, v = item.split("=", 1)
        env[k] = v
    return env

def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)

def run_stream(cmd: List[str], cwd: Path, env: Dict[str, str], timeout: Optional[float] = None) -> int:
    """Run a command, streaming stdout/stderr to the console. Optional timeout (seconds)."""
    preexec = None
    try:
        # POSIX: start a new process group to make termination more robust
        if os.name == "posix":
            import signal
            preexec = os.setsid
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=preexec,
        )
    except FileNotFoundError:
        print(f"ERROR: failed to start: {_format_cmd(cmd)} (cwd={cwd})", file=sys.stderr)
        return 127

    start = time.time()
    try:
        # Stream output line-by-line
        assert proc.stdout and proc.stderr
        from threading import Thread

        def pump(stream, writer):
            for line in iter(stream.readline, ""):
                writer.write(line)
                writer.flush()

        t_out = Thread(target=pump, args=(proc.stdout, sys.stdout), daemon=True)
        t_err = Thread(target=pump, args=(proc.stderr, sys.stderr), daemon=True)
        t_out.start(); t_err.start()

        proc.wait(timeout=timeout)
        t_out.join(); t_err.join()
    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT after {timeout:.0f}s: {_format_cmd(cmd)}", file=sys.stderr)
        try:
            if os.name == "posix":
                import signal, os as _os
                _os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        return 124
    return proc.returncode

@contextmanager
def step(title: str):
    t0 = time.time()
    print(title, flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"{title} ... done in {dt:.2f}s", flush=True)

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="Repo root containing Makefile; auto-detected if omitted.")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Parallel build jobs.")
    ap.add_argument("--clean", action="store_true", help="Run `make clean` before build.")
    ap.add_argument("--make-cmd", default="make", help="Make command (make/gmake/mingw32-make...).")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL to add to environment (repeatable).")
    ap.add_argument("--timeout-build", type=float, default=0.0, help="Build timeout seconds (0=none).")
    ap.add_argument("--verbose", action="store_true", help="Echo commands before execution.")
    args = ap.parse_args(argv)

    try:
        root = args.root.resolve() if args.root else find_make_root(Path(__file__).resolve())
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 2

    if not (root / "Makefile").exists():
        print(f"ERROR: Makefile not found at {root}", file=sys.stderr); return 2

    env = merge_env(os.environ.copy(), args.env)
    make = args.make_cmd

    total_start = time.time()

    if args.clean:
        cmd = [make, "-C", str(root), "clean"]
        if args.verbose: print(f"[cmd] {_format_cmd(cmd)} (cwd={root})")
        rc = run_stream(cmd, root, env)
        if rc != 0: print(f"[clean] failed (rc={rc})", file=sys.stderr); return rc

    # [1/3] Build
    build_cmd = [make, "-C", str(root), f"-j{args.jobs}", "all"]
    if args.verbose: print(f"[cmd] {_format_cmd(build_cmd)} (cwd={root})")
    with step("[1/3] Building..."):
        rc = run_stream(build_cmd, root, env, timeout=(args.timeout_build or None))
        if rc != 0:
            print(f"[build] failed (rc={rc})", file=sys.stderr)
            return rc

    # [2/3] Deploy
    deploy_cmd = [make, "-C", str(root), "deploy"]
    if args.verbose: print(f"[cmd] {_format_cmd(deploy_cmd)} (cwd={root})")
    with step("[2/3] Deploying..."):
        rc = run_stream(deploy_cmd, root, env)
        if rc != 0:
            print(f"[deploy] failed (rc={rc})", file=sys.stderr)
            return rc

    total_dt = time.time() - total_start
    print(f"[info] Build+Deploy completed in {total_dt:.2f}s")

    # [3/3] View
    view_cmd = [make, "-C", str(root), "view"]
    if args.verbose: print(f"[cmd] {_format_cmd(view_cmd)} (cwd={root})")
    print("[3/3] Viewing logs (Ctrl-C to exit)...", flush=True)
    try:
        rc = run_stream(view_cmd, root, env)
    except KeyboardInterrupt:
        print("\n[view] interrupted by user; exiting cleanly.")
        return 0
    return 0 if rc == 0 else rc

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
