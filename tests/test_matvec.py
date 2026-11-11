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

    async def write_bytes_individually(self, base: int, data: np.ndarray) -> None:
        """One bus write per byte, using only a single byte enable (stresses write lanes)."""
        raw = np.ascontiguousarray(data, dtype=np.int8).reshape(-1).view(np.uint8)
        for i, b in enumerate(raw):
            await self.write(base + (i & ~3), int(b) << (8 * (i & 3)), 1 << (i & 3))

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


def chunks_per_row(n: int, lanes: int) -> int:
    return math.ceil(n / lanes)


def expected_cycles(m: int, n: int, gs: int, lanes: int) -> int:
    """Cycle model of the M1 engine (docs/architecture.md, section 7).

    One LANES-wide slice enters the pipeline every cycle (groups do not cost
    extra cycles because they are LANES-aligned), plus the pipeline latency:
    RAM read, multiply, log2(LANES) adder-tree levels, accumulate/Y write.
    """
    return m * chunks_per_row(n, lanes) + int(math.log2(lanes)) + 3


PAD_VALUE = -128   # deliberately nasty: (-128) x (-128) is the largest possible product


def pad_w(w: np.ndarray, lanes: int) -> np.ndarray:
    """W window layout: every row padded to a multiple of LANES bytes.

    The padding content is architecturally "don't care" (the engine masks it),
    so the tests fill it with PAD_VALUE rather than zeros: a missing or wrong
    mask would then change the result.
    """
    m, n = w.shape
    padded = np.full((m, chunks_per_row(n, lanes) * lanes), PAD_VALUE, dtype=np.int8)
    padded[:, :n] = w
    return padded


def pad_x(x: np.ndarray, lanes: int) -> np.ndarray:
    """X bytes beyond N are also don't-care; fill them with PAD_VALUE."""
    padded = np.full(chunks_per_row(len(x), lanes) * lanes, PAD_VALUE, dtype=np.int8)
    padded[:len(x)] = x
    return padded


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
                  label: str = "", check_cycles: bool = True,
                  bytewise: bool = False) -> np.ndarray:
    """Program dims, load data, START, wait, read Y, compare bit-exact."""
    m, n = w.shape
    groups = n // gs if gs else 1
    await bus.write(REG_DIM_M, m)
    await bus.write(REG_DIM_N, n)
    await bus.write(REG_GROUP, gs)
    writer = bus.write_bytes_individually if bytewise else bus.write_bytes
    await writer(X_BASE, pad_x(x, lanes))
    await writer(W_BASE, pad_w(w, lanes))
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
    for gs in ((0, lanes) if lanes <= 16 else (0,)):
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
    """Per-group integer partial sums (llama2.c style). GROUP_SIZE: power of two >= LANES."""
    bus, lanes, m_max, n_max = await setup(dut)
    cases = []
    for gs in (lanes, 2 * lanes, 4 * lanes):
        n = 4 * gs
        if n <= n_max:
            cases.append((4, n, gs))
    cases.append((3, 2 * lanes, 2 * lanes))        # one group per row
    cases.append((2, 5 * lanes, lanes))            # N = 5 groups, odd count
    for m, n, gs in cases:
        w, x = ref.random_case(m, n, 3000 + m * 100 + gs)
        y = await run_job(bus, lanes, w, x, gs, label="grouped")
        whole = ref.matvec(w, x, 0)                # group sums add up to the whole-row result
        assert np.array_equal(y.reshape(m, n // gs).sum(axis=1, dtype=np.int32), whole)
    for name, (w, x) in ref.corner_cases(16, 4 * lanes).items():
        await run_job(bus, lanes, w, x, lanes, label=f"grouped corner '{name}'")
    dut._log.info("grouped passed")


@cocotb.test()
async def tb_write_lanes_bytewise(dut):
    """W/X loaded one byte at a time with single byte enables (RAM byte lanes)."""
    bus, lanes, _, _ = await setup(dut)
    for m, n in ((3, 2 * lanes + 1), (2, lanes), (4, lanes - 1 if lanes > 1 else 1)):
        w, x = ref.random_case(m, n, 6000 + m * 100 + n)
        await run_job(bus, lanes, w, x, 0, label="bytewise load", bytewise=True)


@cocotb.test()
async def tb_pipeline_throughput(dut):
    """One slice per cycle: CYCLES = M*ceil(N/LANES) + log2(LANES) + 3, for several shapes."""
    bus, lanes, m_max, n_max = await setup(dut)
    for m, n in ((1, 1), (1, lanes), (8, 4 * lanes), (16, 16), (5, 3 * lanes + 1)):
        if m > m_max or n > n_max:
            continue
        w, x = ref.random_case(m, n, 7000 + m + n)
        await run_job(bus, lanes, w, x, 0, label="throughput", check_cycles=True)
        cyc = await bus.read(REG_CYCLES)
        dut._log.info(f"M={m} N={n}: {cyc} cycles "
                      f"(issue {m * chunks_per_row(n, lanes)} + latency {cyc - m * chunks_per_row(n, lanes)})")


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

    # START right after DONE (no data reload): nothing may still be in flight from the last job
    for rep_i in range(3):
        await bus.write(REG_CTRL, CTRL_START)
        await wait_done(bus, 500)
        await bus.write(REG_CTRL, CTRL_START)      # immediately again
        await wait_done(bus, 500)
        y = await bus.read_y(5)
        exp2 = ref.matvec(w2, x2, 0)
        assert np.array_equal(y, exp2), describe_mismatch(y, exp2, f"START right after DONE (round {rep_i})")
        assert await bus.read(REG_CYCLES) == expected_cycles(5, 20, 0, lanes)

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
    await expect_err(16, 3 * lanes, 2 * lanes, "N not a multiple of GROUP_SIZE")
    await expect_err(16, 2 * lanes, lanes // 2, "GROUP_SIZE < LANES")
    await expect_err(16, 6 * lanes, 3 * lanes, "GROUP_SIZE not a power of two")
    # N & (GS-1) == 0 here, so only the explicit power-of-two check rejects it
    await expect_err(16, 4 * lanes, 3 * lanes, "GROUP_SIZE not a power of two (bit-mask test passes)")
    if m_max * (n_max // lanes) > 2048:
        await expect_err(m_max, n_max, lanes, "Y results exceed the 2048-word window")

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
    await bus.write_bytes(X_BASE, pad_x(x, lanes))
    await bus.write_bytes(W_BASE, pad_w(w, lanes))
    await bus.write(REG_CTRL, CTRL_START)
    st = await bus.read(REG_STATUS)
    assert st & ST_BUSY, "job should still be running"
    assert await bus.read(X_BASE) == 0, "X window must read 0 while BUSY"
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
    # --- START right after SOFT_RESET: in-flight pipeline data must not leak into the new job
    await bus.write(REG_CTRL, CTRL_START)
    await bus.wait_cycles(lanes // 4 + 4)          # pipeline partially filled
    await bus.write(REG_CTRL, CTRL_RESET)
    await bus.write(REG_CTRL, CTRL_START)          # same dims, X and W still loaded
    await wait_done(bus, expected_cycles(m, n, 0, lanes) + 200)
    y = await bus.read_y(m)
    assert np.array_equal(y, exp), describe_mismatch(y, exp, "job started right after SOFT_RESET")
    assert await bus.read(REG_CYCLES) == expected_cycles(m, n, 0, lanes)

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
