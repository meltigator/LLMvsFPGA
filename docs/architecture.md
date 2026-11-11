# LLMvsFPGA — Architecture

Design document for the INT8 matvec accelerator on the Terasic DE10-Nano.
Status: **draft, pre-M0**. Everything marked **[verify]** must be checked against the official documentation (Cyclone V Device Handbook, Cyclone V HPS Technical Reference Manual, DE10-Nano User Manual) before being relied upon.

---

## 1. Goal and scope

Accelerate one operation:

    y = W · x        W: int8 [M×N],  x: int8 [N],  y: int32 [M]   (or per-group, see §6)

Everything else in a transformer (RMSNorm, softmax, SiLU, RoPE, sampling, tokenizer) stays on the ARM. The accelerator is a **coprocessor** driven by software running on the HPS.

**Out of scope (for now):** GGUF, full transformer in logic, batch > 1, training, INT4, floating point in fabric.

## 2. Platform

| Item | Value |
|------|-------|
| Board | Terasic DE10-Nano |
| Device | Intel Cyclone V SE 5CSEBA6U23I7 **[verify]** |
| HPS | Dual-core ARM Cortex-A9, Linux |
| Memory | 1 GB DDR3, attached to the HPS (not to the fabric) |
| Fabric budget (approx.) | ~41.9K ALMs, 112 DSP blocks, ~5.5 Mbit M10K **[verify]** |
| Toolchain | Quartus Prime Lite + Platform Designer, simulator TBD (Verilator / Icarus + cocotb) |

The fabric has **no direct DDR3 controller**: it reaches DDR3 through the HPS bridges. This is the central constraint of the project (see §5).

## 3. System overview

    ┌─────────────────────────┐          ┌───────────────────────────────┐
    │  HPS (ARM Cortex-A9)    │          │  FPGA fabric                  │
    │                         │          │                               │
    │  llama2.c host          │  LW H2F  │  ┌─────────────────────────┐  │
    │  driver (hps/)  ────────┼─────────►│  │ Avalon-MM slave:        │  │
    │                         │  bridge  │  │ control + X/Y/W windows │  │
    │                         │◄─────────┼──│                         │  │
    │                         │          │  └──────────┬──────────────┘  │
    └───────────┬─────────────┘          │             │                 │
                │                        │  ┌──────────▼──────────────┐  │
                │                        │  │ matvec core (rtl/)      │  │
                │                        │  │ LANES × (INT8 MAC)      │  │
                │                        │  └──────────▲──────────────┘  │
                │                        │             │ weights (M3+)   │
                │                        │  ┌──────────┴──────────────┐  │
                │                        │  │ Avalon-MM master        │  │
                │                        │  └──────────┬──────────────┘  │
                │                        └─────────────┼─────────────────┘
                │      FPGA-to-SDRAM port              │
                └────────── DDR3 ◄─────────────────────┘

Bridges **[verify base addresses and widths in the HPS TRM]**:

- **Lightweight HPS-to-FPGA (LW H2F)**: 32-bit, base ≈ `0xFF200000`. Used for control registers and small data windows.
- **HPS-to-FPGA (H2F)**: base ≈ `0xC0000000`. Optional, for bulk writes from ARM.
- **FPGA-to-SDRAM**: used from M3 on, for the fabric to stream weights from DDR3.

## 4. Milestone-driven interface evolution

| Milestone | Weights come from | Data path |
|-----------|-------------------|-----------|
| M0–M1 | On-chip RAM, loaded by ARM over LW H2F | ARM → registers/RAM → core |
| M2 | Same, with a proper C driver | `/dev/mem` or UIO mmap |
| M3 | DDR3, streamed by an Avalon-MM master | DDR3 → master → core |
| M4+ | DDR3, multiple layers back to back | Descriptor-based (TBD) |

The control interface is designed in M0 so that M3 only **adds** registers (a weight base address) and does not change existing ones.

## 5. The real bottleneck: weight bandwidth

In batch-1 matvec, **every weight is used exactly once**. There is no weight reuse, so:

    bytes/s needed = LANES × f_clk        (1 INT8 weight per MAC per cycle)

Example: 16 lanes at 100 MHz need 1.6 GB/s of weight bandwidth. If the bandwidth the fabric can actually obtain from DDR3 is lower, extra lanes sit idle.

Sizing rule:

    LANES ≤ B_eff / f_clk        where B_eff = measured DDR3 bandwidth available to the fabric

Upper bound for generation speed (model weights only, ignoring attention cache and other overhead):

    tokens/s ≤ B_eff / model_bytes

Order of magnitude: a ~15M-parameter INT8 model is ~15 MB; at 1 GB/s effective that is at most ~65 tokens/s. The theoretical DDR3 peak on the HPS side (32-bit bus) is a few GB/s, but the share available to the fabric is lower: **measure it in M3, do not assume it**.

Consequence: the project is a roofline exercise. The ARM baseline faces the same memory wall, so the honest result of "ARM vs FPGA" may be a modest gain, and that is a valid outcome.

## 6. Numerical design

### 6.1 Types

- Weights and activations: signed INT8, range [-128, 127].
- Product: fits in signed 16 bits (max magnitude 128 × 128 = 16384 = 2^14).
- Accumulator width:

      W_acc = 16 + ceil(log2(N_acc))       N_acc = number of products summed

  | N_acc | W_acc (minimum) |
  |-------|-----------------|
  | 16 | 20 |
  | 64 | 22 |
  | 768 | 26 |
  | 4096 | 28 |

  A 32-bit accumulator is safe up to N_acc = 65536 and is used for simplicity.

### 6.2 Group-wise scaling (design decision)

llama2.c's INT8 path (`runq.c`) quantizes with **per-group float scales** (group size GS, default 64 **[verify]**). The dot product is computed as integer sums per group, then multiplied by the two scales and accumulated in float.

Decision: the accelerator computes **integer partial sums per group** and returns them; scaling and the float accumulation stay on the ARM.

- Register `GROUP_SIZE` selects GS; `0` means "whole row" (plain y = W·x).
- **Constraint (M1):** GS must be 0 or a **power of two ≥ `LANES`** that divides `DIM_N`. Groups are therefore `LANES`-aligned (no extra cycles) and the hardware needs no divider. This matches llama2.c, whose exporter halves GS until it divides the model dimension, so GS is always a power of two.
- Output layout: `Y[row][group]`, int32, row-major.
- This keeps the fabric integer-only and bit-exact testable.

## 7. Microarchitecture (M1)

Parameters: `LANES` (power of two ≥ 4, default 16), `M_MAX`, `N_MAX`, `Y_WORDS`.

    X RAM ─┐                                                         ┌─► Y RAM
           ├─► [mask] ─► [LANES × int8×int8] ─► [adder tree] ─► [acc]┘
    W RAM ─┘   stage 1        stage 2          stages 3..2+log2 L   stage 3+log2 L

One `LANES`-wide slice of the current row enters the pipeline **every cycle**; results leave `PIPE_LATENCY` cycles later.

### 7.1 Memories (block RAM)

- **W RAM** and **X RAM** are `LANES` bytes wide, so a whole row-slice is fetched per cycle. Each is built from `LANES/4` banks of four byte-wide simple-dual-port RAMs (one write port from the bus, one synchronous read port), which lets the 32-bit bus write any byte with a byte enable while the core reads the full width. Written this way the RAMs are inferable as M10K.
- **Y RAM** is one 32-bit simple-dual-port RAM (write: core, read: bus).
- Bank mapping: bus word index `i` goes to bank `i mod (LANES/4)`, address `i / (LANES/4)`; bank `b` supplies lanes `4b … 4b+3`.
- The read-during-write behavior is unspecified (`no_rw_check`): the architecture never reads a RAM while the bus writes it (bus writes are ignored while BUSY).
- X has one read port shared by core and bus: **X reads from the bus return 0 while BUSY**.

### 7.2 Layout contract

- **W window:** row `r` starts at byte `r × CPR × LANES`, with `CPR = ceil(DIM_N / LANES)`. When `DIM_N` is a multiple of `LANES` this is plain packed row-major. Otherwise each row is padded up to a multiple of `LANES`.
- Padding bytes in W, and X bytes beyond `DIM_N`, are **don't-care**: the last chunk of a row is masked. The tests fill them with −128 to prove it.
- llama2.c matrices (dimensions multiples of the group size) need no padding in practice.

### 7.3 Pipeline

| Stage | What happens |
|-------|--------------|
| issue | address generator: row, chunk, group counter; emits `valid`, `last` (end of group), lane mask |
| 1 | RAM output (registered inside the RAM) |
| 2 | `LANES` signed 8×8 products, masked lanes forced to 0 (DSP 9×9) |
| 3 … 2+log2 L | registered adder tree, one level per stage |
| 3+log2 L | accumulate; on `last`, result goes to a register |
| 4+log2 L | registered write into Y RAM; `DONE` set after the final write |

`valid` and `last` travel alongside the data, so a new group can start the cycle after the previous one ends with no bubble.

### 7.4 Timing model

    CYCLES = DIM_M × ceil(DIM_N / LANES) + PIPE_LATENCY        PIPE_LATENCY = log2(LANES) + 3

Groups cost nothing extra. `CYCLES` counts every cycle with `BUSY=1`. `tests/test_matvec.py` checks this formula on every job, so any change to the engine's timing is a deliberate, visible change.

Peak throughput is `LANES` MAC/cycle. As §5 explains, that rate is only useful if the weights arrive that fast: with on-chip weights (M1/M2) it is, with DDR3 streaming (M3) it is bounded by the measured bandwidth.

FSM: `IDLE → BUSY → DONE`. `START` from IDLE begins a run; `DONE` is sticky until the next `START` or `SOFT_RESET`.

### 7.5 Synthesis notes

- No variable divider: `GROUP_SIZE` is restricted to powers of two, `DIM_N / GROUP_SIZE` becomes a shift.
- One combinational multiply remains (`DIM_M × groups`, 16×16 bit), evaluated only at START for the Y-capacity check. It is a candidate multi-cycle path if timing is tight.
- Open items for Quartus: confirm M10K inference, confirm the 8×8 products land in DSP blocks (`multstyle`), `f_max`, and the resource report.

## 8. Register map (LW H2F slave)

Offsets from the slave base address. 32-bit registers, little-endian.

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x0000` | `ID` | RO | `0x4C4C4650` ("LLFP") |
| `0x0004` | `VERSION` | RO | Design version (`2` = M1) |
| `0x0008` | `CTRL` | RW | bit0 `START` (self-clearing), bit1 `SOFT_RESET` |
| `0x000C` | `STATUS` | RO | bit0 `BUSY`, bit1 `DONE`, bit2 `ERR` |
| `0x0010` | `DIM_M` | RW | Number of rows |
| `0x0014` | `DIM_N` | RW | Row length |
| `0x0018` | `GROUP_SIZE` | RW | 0 = whole row, else a power of two ≥ `LANES` dividing `DIM_N` |
| `0x001C` | `CYCLES` | RO | Engine cycles of the last run |
| `0x0020` | `CAPS` | RO | bits [7:0] = `LANES` |
| `0x0024` | `LIMITS` | RO | bits [31:16] = `M_MAX`, bits [15:0] = `N_MAX` |
| `0x1000` | `X` window | RW | int8, 4 per word, little-endian, up to `N_MAX` bytes (4 KB). Reads return 0 while `BUSY` |
| `0x2000` | `Y` window | RO | int32 results, up to 2048 words (8 KB) |
| `0x10000` | `W` window | WO | int8, row-major, each row padded to a multiple of `LANES` (§7.2), up to 64 KB (M0–M2) |

Software discovers `LANES`, `M_MAX` and `N_MAX` from `CAPS`/`LIMITS` instead of hard-coding them, so the same driver and the same tests run against different builds.

Reset defaults: `DIM_M = 16`, `DIM_N = 16`, `GROUP_SIZE = 0` (the M0 target configuration), `STATUS = 0`.

The W/X layout and the `GROUP_SIZE` constraint changed between M0 and M1 (§6.2, §7.2). `VERSION` lets a driver tell the builds apart.

### 8.1 Behavior rules

- **START** (`CTRL` bit0) is evaluated when written. If the configuration is valid and the engine is idle: `BUSY=1`, `DONE=0`, `ERR=0`, `CYCLES=0`.
- **Invalid configuration** on START (nothing starts; `STATUS` reads exactly `ERR`):
  `DIM_M = 0` or `> M_MAX`; `DIM_N = 0` or `> N_MAX`; `GROUP_SIZE` nonzero and not a power of two, `< LANES`, `> DIM_N`, or not dividing `DIM_N`; or `DIM_M × groups_per_row` larger than the Y window (2048 words).
- **START while BUSY** is ignored (the running job is not restarted) and raises `ERR`.
- **While BUSY**, writes to `DIM_M`, `DIM_N`, `GROUP_SIZE` and to the X/W windows are ignored, so a running job cannot be corrupted. Only `CTRL` is accepted.
- **SOFT_RESET** (`CTRL` bit1) aborts a running job, flushes the pipeline, and clears `BUSY`, `DONE` and `ERR`. It takes priority over START if both bits are set. A START may follow in the very next cycle.
- A START may also be issued in the cycle right after `DONE` is seen: nothing is left in flight.
- **DONE** is sticky until the next START or SOFT_RESET. **ERR** is cleared by SOFT_RESET or by the next accepted START.
- Reads of the W window return 0; writes to the Y window are ignored. Y is valid once `DONE` is set.
- The X and W windows honor the Avalon byte enables.
- Avalon-MM slave: word addressing (byte offset >> 2), fixed read latency of 1 cycle, no `waitrequest`.

M3 adds (no existing register changes): `W_BASE_ADDR`, `W_STRIDE`, and a `SRC` bit selecting on-chip vs DDR3 weights.

## 9. Verification strategy

Three levels, same test vectors at each level:

1. **Golden model** (`software/reference.py`): pure Python/NumPy, `int8` inputs, `int32` accumulation. Includes the group-wise variant.
2. **RTL simulation** (`sim/run_sim.py`, `tests/test_matvec.py`): cocotb with Icarus Verilog, driving the Avalon register map and comparing RTL output with the golden model, for `LANES` = 4, 8, 16, 32.
3. **On board** (`hps/`): the same vectors driven from the ARM; mismatches are reported with row/column and expected vs obtained values.

Test vector classes:

- Random matrices with fixed seeds (reproducible).
- Corner cases: all `-128`, all `127`, `-128 × -128` (maximum magnitude), zeros, identity, alternating signs.
- Non-multiples of `LANES`, with the padding filled with −128 (the mask must hide it).
- Group sizes: `0` and powers of two ≥ `LANES`; invalid sizes must raise `ERR`.
- Byte-at-a-time loading (single byte enables) to exercise every RAM byte lane.
- Control corner cases: START while BUSY, writes while BUSY, SOFT_RESET mid-job followed by an immediate START, START right after DONE.
- Timing: `CYCLES` must equal the formula of §7.4 on every job.

Pass criterion: **bit-exact** equality. No tolerance anywhere in the integer path.

**Testing the tests.** The suite was checked by mutation: 18 deliberate RTL bugs (unsigned weights, missing lane mask, wrong tree wiring, bank mapping, byte enables, pipeline flush, validation rules, extra row issued, ...) were injected one at a time and every one is caught. Two rounds of this found real gaps in the first version of the tests (zero padding hiding a missing mask; no START immediately after DONE), which is why those cases exist.

Static check: Yosys sees all memories as memories (W and X: `4 × LANES/4` byte RAMs each, Y: one 32-bit RAM), each with one write port and one synchronous read port, and emits no "replaced by registers" warning.

## 10. Benchmark methodology

- Baseline: llama2.c `runq` (INT8) compiled with optimizations and NEON enabled, on the same board, same model, same prompt, same token count.
- Metrics: time per matvec, tokens/s, measured DDR3 bandwidth, resource usage (ALMs, DSP, M10K), `f_max` from timing closure, power if measurable.
- Method: warm-up runs, ≥ 30 repetitions, report median and spread, fixed CPU frequency/governor.
- Report where time goes (matvec vs everything else) to quantify the Amdahl limit of the hybrid design.

## 11. Risks and open questions

| Risk | Mitigation |
|------|------------|
| Effective DDR3 bandwidth to the fabric is low | Measure early (start of M3), size `LANES` from data |
| Cortex-A9 + NEON is already near the memory wall | Report honestly; the ratio is the result |
| Non-matvec operations dominate after offload | Profile in M4 before optimizing |
| M10K capacity limits on-chip weights | Tiling in M1, DDR3 streaming in M3 |
| Quartus Lite limited SystemVerilog support | The RTL uses Verilog-2001 constructs plus `$clog2`; verify in Quartus |
| M1 RTL has been simulated and checked with Yosys, never with Quartus | First on-board step: synthesize with Quartus, read the resource and timing reports before anything else |
| Yosys/ABC crashed once during a full Cyclone V mapping, so logic-cell counts are unreliable | Use Quartus reports; rely on Yosys only for memory inference and DSP counts |
| Timing closure below target `f_clk` | Pipeline the adder tree, reduce `LANES` |

## 12. References

- Cyclone V Device Handbook, Intel/Altera
- Cyclone V Hard Processor System Technical Reference Manual
- DE10-Nano User Manual, Terasic
- A. Karpathy, llama2.c (`run.c`, `runq.c`)
- RCoeurjoly, LLM2FPGA (related work, compiler-based approach)
- S. Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009)
  
