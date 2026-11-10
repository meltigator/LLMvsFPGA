#!/usr/bin/env python3
"""LLMvsFPGA - simulation front end.

Builds rtl/matvec.sv with Icarus Verilog, runs the cocotb testbench
(tests/test_matvec.py) for one or more LANES values and prints a summary.
The testbench compares the RTL bit-for-bit against software/reference.py.

Examples:
    python sim/run_sim.py                      # LANES=16 and LANES=4
    python sim/run_sim.py --lanes 16           # only the M0 target configuration
    python sim/run_sim.py --random 200 --big   # heavier regression
    python sim/run_sim.py --waves --view       # dump waveforms and open GTKWave
    python sim/run_sim.py --test tb_grouped    # run a single testbench function
    python sim/run_sim.py --clean              # remove build products and exit

Exit code: 0 = everything passed, 1 = a test failed, 2 = environment problem.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "sim" / "sim_build"


def check_environment() -> list[str]:
    """Return a list of human-readable problems (empty list = all good)."""
    problems = []
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        problems.append(
            "Icarus Verilog not found (iverilog/vvp).\n"
            "      Linux:   sudo apt install iverilog\n"
            "      macOS:   brew install icarus-verilog\n"
            "      Windows: use WSL, or the OSS CAD Suite (https://github.com/YosysHQ/oss-cad-suite-build)")
    for module in ("numpy", "cocotb", "pytest"):
        if importlib.util.find_spec(module) is None:
            problems.append(f"Python module '{module}' missing.  pip install {module}")
    for path in (ROOT / "rtl" / "matvec.sv", ROOT / "software" / "reference.py",
                 ROOT / "tests" / "test_matvec.py"):
        if not path.exists():
            problems.append(f"missing file: {path.relative_to(ROOT)}")
    return problems


def read_results(lanes: int) -> list[tuple[str, str, float]]:
    """(name, status, seconds) for each cocotb test of one LANES build."""
    xml = BUILD_ROOT / f"lanes{lanes}" / "results.xml"
    if not xml.exists():
        return []
    rows = []
    for tc in ET.parse(xml).iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "FAIL"
        elif tc.find("skipped") is not None:
            status = "skip"
        else:
            status = "ok"
        rows.append((tc.get("name", "?"), status, float(tc.get("time", 0))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", type=int, nargs="+", default=[16, 4],
                    help="LANES values to simulate (default: 16 4)")
    ap.add_argument("--random", type=int, default=30, metavar="N",
                    help="number of random 16x16 matrices (default 30)")
    ap.add_argument("--big", action="store_true",
                    help="also run the full-size M_MAX x N_MAX test (slower)")
    ap.add_argument("--waves", action="store_true",
                    help="dump waveforms (FST) in sim/sim_build/lanes<N>/")
    ap.add_argument("--view", action="store_true",
                    help="open the first waveform with GTKWave/Surfer (implies --waves)")
    ap.add_argument("--test", metavar="REGEX",
                    help="run only the cocotb tests whose name matches REGEX (cocotb >= 2.0)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream the full simulator log")
    ap.add_argument("--clean", action="store_true",
                    help="delete sim/sim_build and exit")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(BUILD_ROOT, ignore_errors=True)
        print(f"removed {BUILD_ROOT.relative_to(ROOT)}")
        return 0

    problems = check_environment()
    if problems:
        print("Environment problems:")
        for p in problems:
            print(f"  - {p}")
        return 2

    waves = args.waves or args.view
    env = dict(os.environ,
               LLMVSFPGA_LANES=",".join(str(n) for n in args.lanes),
               LLMVSFPGA_RANDOM=str(args.random),
               LLMVSFPGA_BIG="1" if args.big else "0",
               LLMVSFPGA_WAVES="1" if waves else "0")
    if args.test:
        env["COCOTB_TEST_FILTER"] = args.test

    for n in args.lanes:                       # stale results must not be mistaken for new ones
        shutil.rmtree(BUILD_ROOT / f"lanes{n}", ignore_errors=True)

    cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"]
    if args.verbose:
        cmd.append("-s")

    print(f"LLMvsFPGA simulation: LANES={args.lanes}  random={args.random}  "
          f"big={'yes' if args.big else 'no'}  waves={'yes' if waves else 'no'}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=not args.verbose, text=True)
    elapsed = time.time() - t0

    all_ok = proc.returncode == 0
    for n in args.lanes:
        rows = read_results(n)
        print(f"\n  LANES={n}")
        if not rows:
            print("    no results (build or simulator error)")
            all_ok = False
            continue
        for name, status, secs in rows:
            print(f"    [{status:4s}] {name:36s} {secs:6.2f} s")
            if status == "FAIL":
                all_ok = False

    if not all_ok and not args.verbose:
        print("\n--- simulator / pytest output (tail) ---")
        print("\n".join((proc.stdout + proc.stderr).splitlines()[-40:]))
        print("Re-run with -v for the full log.")

    fsts = sorted(BUILD_ROOT.glob("lanes*/*.fst")) if waves else []
    for f in fsts:
        print(f"\n  waveform: {f.relative_to(ROOT)}")
    if args.view and fsts:
        viewer = shutil.which("gtkwave") or shutil.which("surfer")
        if viewer:
            subprocess.Popen([viewer, str(fsts[0])])
        else:
            print("  (no gtkwave/surfer found: install one to view the waveform)")

    print(f"\n{'ALL PASS' if all_ok else 'FAILURES'}  ({elapsed:.1f} s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

