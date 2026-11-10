"""LLMvsFPGA - automatic hardware/software comparison for the matvec accelerator.

The RTL (rtl/matvec.sv) is driven through its Avalon-MM register map exactly as
the ARM driver will drive it on the board, and every result is compared
bit-for-bit with the golden model (software/reference.py).

Run everything (builds with Icarus Verilog, then simulates):

    pytest tests/ -v

Useful environment variables:

    LLMVSFPGA_RANDOM=100   number of random 16x16 matrices (default 30)
    LLMVSFPGA_BIG=1        also run the full-size M_MAX x N_MAX test (slow)
    LLMVSFPGA_LANES=16,4   LANES values to simulate (default "16,4")
    LLMVSFPGA_WAVES=1      dump waveforms (FST) in sim/sim_build/lanes<N>/
    SIM=icarus             simulator passed to the cocotb runner (default icarus)

The convenient front end is sim/run_sim.py (see its --help).

Structure of this file:
  * cocotb testbench (functions named tb_*): executed inside the simulator.
  * test_matvec_sim (pytest): builds the RTL and launches the simulator.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import pytest

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))
import reference as ref  # noqa: E402  (golden model)

# ---------------------------------------------------------------------------
# Register map (byte offsets) - keep in sync with docs/architecture.md, sec. 8
# ---------------------------------------------------------------------------
REG_ID, REG_VERSION, REG_CTRL, REG_STATUS = 0x00, 0x04, 0x08, 0x0C
REG_DIM_M, REG_DIM_N, REG_GROUP, REG_CYCLES = 0x10, 0x14, 0x18, 0x1C
REG_CAPS, REG_LIMITS = 0x20, 0x24
X_BASE, Y_BASE, W_BASE = 0x1000, 0x2000, 0x10000

CTRL_START, CTRL_RESET = 0x1, 0x2
ST_BUSY, ST_DONE, ST_ERR = 0x1, 0x2, 0x4
ID_VALUE = 0x4C4C4650  # "LLFP"

N_RANDOM = int(os.environ.get("LLMVSFPGA_RANDOM", "30"))
RUN_BIG = os.environ.get("LLMVSFPGA_BIG", "0") == "1"
WAVES = os.environ.get("LLMVSFPGA_WAVES", "0") == "1"
LANES_LIST = [int(v) for v in os.environ.get("LLMVSFPGA_LANES", "16,4").split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Bus driver (Avalon-MM slave: word addressing, read latency 1, no waitrequest)
# All activity is driven on falling edges, so the DUT samples stable values on
# the rising edge in between.
# ---------------------------------------------------------------------------
class Bus:
    def __init__(self, dut):
        self.dut = dut

    def _idle(self):
        self.dut.avs_read.value = 0
        self.dut.avs_write.value = 0

    async def write(self, offset: int, value: int, byteenable: int = 0xF) -> None:
        d = self.dut
        d.avs_address.value = offset >> 2
        d.avs_writedata.value = value & 0xFFFFFFFF
        d.avs_byteenable.value = byteenable
        d.avs_read.value = 0
        d.avs_write.value = 1
        await FallingEdge(d.clk)
        self._idle()

    async def read(self, offset: int) -> int:
        d = self.dut
        d.avs_address.value = offset >> 2
        d.avs_write.value = 0
        d.avs_read.value = 1
        await FallingEdge(d.clk)          # DUT latched readdata on the rising edge
        self._idle()
        return int(d.avs_readdata.value)

    async def wait_cycles(self, n: int) -> None:
        for _ in range(n):
            await FallingEdge(self.dut.clk)

    async def write_bytes(self, base: int, data: np.ndarray) -> None:
        """Write an int8 array little-endian, 4 bytes per word (tail uses byteenable)."""
        raw = np.ascontiguousarray(data, dtype=np.int8).reshape(-1).view(np.uint8)
        for i in range(0, len(raw), 4):
            chunk = raw[i:i + 4]
            word = sum(int(b) << (8 * k) for k, b in enumerate(chunk))
            await self.write(base + i, word, (1 << len(chunk)) - 1)

    async def read_words(self, base: int, count: int) -> list[int]:
        return [await self.read(base + 4 * i) for i in range(count)]

    async def read_y(self, count: int) -> np.ndarray:
        words = await self.read_words(Y_BASE, count)
        return np.array(words, dtype=np.uint32).view(np.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def setup(dut) -> tuple[Bus, int, int, int]:
    """Start the clock, apply reset, return (bus, lanes, m_max, n_max)."""
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.reset_n.value = 0
    dut.avs_address.value = 0
    dut.avs_writedata.value = 0
    dut.avs_byteenable.value = 0xF
    dut.avs_read.value = 0
    dut.avs_write.value = 0
    for _ in range(5):
        await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    for _ in range(3):
        await FallingEdge(dut.clk)
    bus = Bus(dut)
    lanes = await bus.read(REG_CAPS)
    limits = await bus.read(REG_LIMITS)
    return bus, lanes, limits >> 16, limits & 0xFFFF


def expected_cycles(m: int, n: int, gs: int, lanes: int) -> int:
    """Cycle model of the M0 engine: each group takes ceil(group/lanes) cycles."""
    gs_eff = gs or n
    return m * (n // gs_eff) * math.ceil(gs_eff / lanes)


async def wait_done(bus: Bus, max_cycles: int, fail_on_err: bool = True) -> int:
    for _ in range(max_cycles):
        st = await bus.read(REG_STATUS)
        if fail_on_err:
            assert not (st & ST_ERR), f"unexpected ERR, STATUS=0x{st:x}"
        if st & ST_DONE:
            return st
        assert st & ST_BUSY, f"engine neither busy nor done, STATUS=0x{st:x}"
    raise AssertionError(f"timeout: DONE not raised within {max_cycles} polls")


def describe_mismatch(got: np.ndarray, exp: np.ndarray, label: str) -> str:
    bad = np.flatnonzero(got != exp)
    lines = [f"{label}: {len(bad)} mismatching value(s) out of {exp.size}"]
    for i in bad[:5]:
        lines.append(f"    index {int(i)}: hardware={int(got[i])} expected={int(exp[i])}")
    return "\n".join(lines)


async def run_job(bus: Bus, lanes: int, w: np.ndarray, x: np.ndarray, gs: int = 0,
                  label: str = "", check_cycles: bool = True) -> np.ndarray:
    """Program dims, load data, START, wait, read Y, compare bit-exact."""
    m, n = w.shape
    groups = n // gs if gs else 1
    await bus.write(REG_DIM_M, m)
    await bus.write(REG_DIM_N, n)
    await bus.write(REG_GROUP, gs)
    await bus.write_bytes(X_BASE, x)
    await bus.write_bytes(W_BASE, w)
    await bus.write(REG_CTRL, CTRL_START)
    await wait_done(bus, expected_cycles(m, n, gs, lanes) + 200)

    y = await bus.read_y(m * groups)
    exp = ref.matvec(w, x, gs).reshape(-1)
    assert np.array_equal(y, exp), describe_mismatch(
        y, exp, f"{label or 'job'} (M={m} N={n} GS={gs})")

    if check_cycles:
        cyc = await bus.read(REG_CYCLES)
        want = expected_cycles(m, n, gs, lanes)
        assert cyc == want, f"{label}: CYCLES={cyc}, expected {want} (M={m} N={n} GS={gs})"
    return y


# ---------------------------------------------------------------------------
# Testbench
# ---------------------------------------------------------------------------
@cocotb.test()
async def tb_identity_and_reset_state(dut):
    """ID/version/capabilities are readable; reset leaves the engine idle."""
    bus, lanes, m_max, n_max = await setup(dut)
    assert await bus.read(REG_ID) == ID_VALUE
    assert await bus.read(REG_VERSION) != 0
    assert lanes >= 1 and m_max >= 1 and n_max >= 1
    assert await bus.read(REG_STATUS) == 0
    assert await bus.read(REG_DIM_M) == 16, "reset default DIM_M must be 16 (M0 target)"
    assert await bus.read(REG_DIM_N) == 16, "reset default DIM_N must be 16 (M0 target)"
    assert await bus.read(REG_GROUP) == 0
    dut._log.info(f"LANES={lanes} M_MAX={m_max} N_MAX={n_max}")


@cocotb.test()
async def tb_corner_cases_16x16(dut):
    """M0 target: 16x16 corner cases, whole-row and grouped."""
    bus, lanes, _, _ = await setup(dut)
    for gs in (0, 4):
        for name, (w, x) in ref.corner_cases(16, 16).items():
            await run_job(bus, lanes, w, x, gs, label=f"corner '{name}'")
    dut._log.info("corner cases passed")


@cocotb.test()
async def tb_random_16x16(dut):
    """M0 target: random 16x16 matrices, fixed seeds, bit-exact."""
    bus, lanes, _, _ = await setup(dut)
    for seed in range(N_RANDOM):
        w, x = ref.random_case(16, 16, 1000 + seed)
        await run_job(bus, lanes, w, x, 0, label=f"random seed={1000 + seed}")
    dut._log.info(f"{N_RANDOM} random matrices passed")


@cocotb.test()
async def tb_shapes(dut):
    """Sizes that are not multiples of LANES, tiny and larger shapes."""
    bus, lanes, m_max, n_max = await setup(dut)
    shapes = [(1, 1), (1, 16), (3, 17), (5, 33), (16, 20), (7, 64), (2, 255),
              (32, 128), (4, lanes - 1 if lanes > 1 else 1), (4, lanes + 1)]
    for m, n in shapes:
        if m > m_max or n > n_max:
            continue
        w, x = ref.random_case(m, n, 2000 + m * 1000 + n)
        await run_job(bus, lanes, w, x, 0, label="shape")
    dut._log.info("shapes passed")


@cocotb.test()
async def tb_grouped(dut):
    """Per-group integer partial sums (llama2.c style), incl. group not multiple of LANES."""
    bus, lanes, _, _ = await setup(dut)
    cases = [(4, 64, 16), (4, 64, 32), (8, 128, 64), (3, 48, 12), (5, 30, 10),
             (2, 16, 16), (2, 16, 1)]
    for m, n, gs in cases:
        w, x = ref.random_case(m, n, 3000 + m * 100 + gs)
        y = await run_job(bus, lanes, w, x, gs, label="grouped")
        # consistency: group sums add up to the whole-row result
        whole = ref.matvec(w, x, 0)
        assert np.array_equal(y.reshape(m, n // gs).sum(axis=1, dtype=np.int32), whole)
    for name, (w, x) in ref.corner_cases(16, 32).items():
        await run_job(bus, lanes, w, x, 8, label=f"grouped corner '{name}'")
    dut._log.info("grouped passed")


@cocotb.test()
async def tb_back_to_back_no_stale_state(dut):
    """Consecutive jobs without reset: no accumulator/data leakage, DONE semantics."""
    bus, lanes, _, _ = await setup(dut)
    w1 = np.full((16, 64), 127, dtype=np.int8)
    x1 = np.full(64, 127, dtype=np.int8)
    await run_job(bus, lanes, w1, x1, 0, label="job A (saturating)")

    # DONE is sticky until the next START
    for _ in range(3):
        assert await bus.read(REG_STATUS) == ST_DONE

    w2, x2 = ref.random_case(5, 20, 4242)       # smaller N: stale data beyond must be ignored
    await run_job(bus, lanes, w2, x2, 0, label="job B (smaller)")
    await run_job(bus, lanes, w2, x2, 0, label="job B repeated")

    # START clears DONE immediately
    await bus.write(REG_CTRL, CTRL_START)
    st = await bus.read(REG_STATUS)
    assert not (st & ST_DONE), f"DONE must clear on START, STATUS=0x{st:x}"
    await wait_done(bus, 500)


@cocotb.test()
async def tb_error_handling(dut):
    """Invalid configurations raise ERR and never start the engine."""
    bus, lanes, m_max, n_max = await setup(dut)

    async def expect_err(m, n, gs, why):
        await bus.write(REG_CTRL, CTRL_RESET)      # clear a leftover ERR from earlier cases
        await bus.write(REG_DIM_M, m)
        await bus.write(REG_DIM_N, n)
        await bus.write(REG_GROUP, gs)
        await bus.write(REG_CTRL, CTRL_START)
        await bus.wait_cycles(2)
        st = await bus.read(REG_STATUS)
        assert st == ST_ERR, f"{why}: expected only ERR, STATUS=0x{st:x}"

    await expect_err(0, 16, 0, "M=0")
    await expect_err(16, 0, 0, "N=0")
    await expect_err(m_max + 1, 16, 0, "M > M_MAX")
    await expect_err(16, n_max + 1, 0, "N > N_MAX")
    await expect_err(16, 16, 32, "GROUP_SIZE > N")
    await expect_err(16, 30, 8, "N not a multiple of GROUP_SIZE")
    if m_max * n_max > 2048:
        await expect_err(m_max, n_max, 1, "Y results exceed the 2048-word window")

    # ERR is cleared by the next valid START, and the engine works afterwards
    w, x = ref.random_case(16, 16, 77)
    await run_job(bus, lanes, w, x, 0, label="after errors")
    assert await bus.read(REG_STATUS) == ST_DONE


@cocotb.test()
async def tb_busy_semantics(dut):
    """START and register/data writes while BUSY are ignored; SOFT_RESET aborts."""
    bus, lanes, m_max, n_max = await setup(dut)
    m, n = min(16, m_max), min(256, n_max)
    w, x = ref.random_case(m, n, 99)

    # --- ignored writes while busy
    await bus.write(REG_DIM_M, m)
    await bus.write(REG_DIM_N, n)
    await bus.write(REG_GROUP, 0)
    await bus.write_bytes(X_BASE, x)
    await bus.write_bytes(W_BASE, w)
    await bus.write(REG_CTRL, CTRL_START)
    st = await bus.read(REG_STATUS)
    assert st & ST_BUSY, "job should still be running"
    await bus.write(REG_CTRL, CTRL_START)          # second START: ignored, raises ERR
    await bus.write(REG_DIM_N, 1)                  # ignored
    await bus.write(X_BASE, 0x7F7F7F7F)            # ignored
    await bus.write(W_BASE, 0x80808080)            # ignored
    st = await wait_done(bus, expected_cycles(m, n, 0, lanes) + 200, fail_on_err=False)
    assert st & ST_ERR, "START while BUSY must raise ERR"
    assert await bus.read(REG_DIM_N) == n, "DIM_N write while BUSY must be ignored"
    y = await bus.read_y(m)
    exp = ref.matvec(w, x, 0)
    assert np.array_equal(y, exp), describe_mismatch(y, exp, "job disturbed while BUSY")
    cyc = await bus.read(REG_CYCLES)
    assert cyc == expected_cycles(m, n, 0, lanes), "ignored START must not restart the job"

    # --- SOFT_RESET aborts a running job
    await bus.write(REG_CTRL, CTRL_START)
    await bus.wait_cycles(3)
    assert (await bus.read(REG_STATUS)) & ST_BUSY
    await bus.write(REG_CTRL, CTRL_RESET)
    assert await bus.read(REG_STATUS) == 0, "SOFT_RESET must clear BUSY/DONE/ERR"
    w2, x2 = ref.random_case(8, 24, 5)
    await run_job(bus, lanes, w2, x2, 0, label="after SOFT_RESET")


@cocotb.test()
async def tb_memory_windows(dut):
    """X readback, byte enables, write-only W window, read-only Y window."""
    bus, lanes, _, _ = await setup(dut)

    await bus.write(X_BASE, 0x11223344)
    assert await bus.read(X_BASE) == 0x11223344
    await bus.write(X_BASE, 0xAABBCCDD, byteenable=0b0101)   # only bytes 0 and 2
    assert await bus.read(X_BASE) == 0x11BB33DD, "byte enables not honored"

    await bus.write(W_BASE, 0xDEADBEEF)
    assert await bus.read(W_BASE) == 0, "W window must read as zero (write-only)"

    w, x = ref.random_case(16, 16, 31337)
    y = await run_job(bus, lanes, w, x, 0, label="before Y write")
    await bus.write(Y_BASE, 0xDEADBEEF)                       # read-only: ignored
    assert await bus.read(Y_BASE) == int(y[0]) & 0xFFFFFFFF


@cocotb.test(skip=not RUN_BIG)
async def tb_full_size(dut):
    """Optional (LLMVSFPGA_BIG=1): largest supported matrix."""
    bus, lanes, m_max, n_max = await setup(dut)
    w, x = ref.random_case(m_max, n_max, 123456)
    await run_job(bus, lanes, w, x, 0, label="full size")


# ---------------------------------------------------------------------------
# pytest entry point: build the RTL with the simulator and run the testbench
# ---------------------------------------------------------------------------
def _failed_testcases(results_xml: Path) -> list[str]:
    tree = ET.parse(results_xml)
    return [tc.get("name", "?") for tc in tree.iter("testcase")
            if tc.find("failure") is not None or tc.find("error") is not None]


def _run_simulation(lanes: int) -> None:
    try:                                    # cocotb >= 2.0
        from cocotb_tools.runner import get_runner
    except ImportError:                     # cocotb 1.8 / 1.9
        from cocotb.runner import get_runner

    sim = os.environ.get("SIM", "icarus")
    build_dir = ROOT / "sim" / "sim_build" / f"lanes{lanes}"
    runner = get_runner(sim)
    sources = [ROOT / "rtl" / "matvec.sv"]
    build_kwargs = dict(
        hdl_toplevel="matvec",
        parameters={"LANES": lanes},
        build_args=["-g2012"] if sim == "icarus" else [],
        build_dir=build_dir,
        always=True,
        timescale=("1ns", "1ps"),   # the RTL carries no `timescale directive
        waves=WAVES,
    )
    try:
        runner.build(sources=sources, **build_kwargs)
    except TypeError:
        runner.build(verilog_sources=sources, **build_kwargs)

    try:
        results = runner.test(
            hdl_toplevel="matvec",
            test_module="test_matvec",
            test_dir=str(Path(__file__).resolve().parent),
            build_dir=build_dir,
            results_xml=str(build_dir / "results.xml"),
            waves=WAVES,
        )
    except SystemExit as exc:               # cocotb 2.x exits on failing tests
        pytest.fail(f"simulation reported failures (exit code {exc.code}); "
                    f"see output above")
    failed = _failed_testcases(Path(results))
    assert not failed, f"failing cocotb tests: {failed}"


@pytest.mark.parametrize("lanes", LANES_LIST, ids=lambda n: f"lanes{n}")
def test_matvec_sim(lanes):
    """Whole suite for one LANES value (16 = the M0 target; 4 stresses multi-cycle rows)."""
    _run_simulation(lanes)
