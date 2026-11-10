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

- Register `GROUP_SIZE` selects GS; `0` means "whole row" (plain y = W·x, used in M0/M1).
- Output layout: `Y[row][group]`, int32, row-major.
- This keeps the fabric integer-only and bit-exact testable.

## 7. Microarchitecture (M0/M1)

Parameters: `LANES` (default 16), `M_MAX`, `N_MAX`.

Datapath per cycle: read `LANES` weights of the current row plus the matching `LANES` activations, multiply, reduce, accumulate.

    W RAM ──┐
            ├─► [LANES × int8×int8] ─► [pipelined adder tree] ─► [accumulator] ─► Y RAM
    X RAM ──┘

- Multipliers are mapped to DSP blocks (INT8 fits the 9×9 mode).
- W RAM is organized with word width `LANES × 8` bits so one row-slice is read per cycle.
- X is read with the same width and the same address sequence.
- The accumulator is cleared at the start of each row (or group) and written out at its end.

Timing model:

    cycles ≈ M × ceil(N / LANES) + PIPE_LATENCY        PIPE_LATENCY ≈ ceil(log2 LANES) + 3

For 16×16 with 16 lanes: 16 rows × 1 cycle + latency. The `CYCLES` register exposes the real count for benchmarking.

FSM: `IDLE → RUN → DONE`. `START` from IDLE begins a run; `DONE` is sticky until the next `START`.

## 8. Register map (LW H2F slave)

Offsets from the slave base address. 32-bit registers, little-endian.

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x0000` | `ID` | RO | `0x4C4C4650` ("LLFP") |
| `0x0004` | `VERSION` | RO | Design version |
| `0x0008` | `CTRL` | RW | bit0 `START` (self-clearing), bit1 `SOFT_RESET` |
| `0x000C` | `STATUS` | RO | bit0 `BUSY`, bit1 `DONE`, bit2 `ERR` |
| `0x0010` | `DIM_M` | RW | Number of rows |
| `0x0014` | `DIM_N` | RW | Row length |
| `0x0018` | `GROUP_SIZE` | RW | 0 = whole row |
| `0x001C` | `CYCLES` | RO | Cycles of the last run |
| `0x1000` | `X` window | RW | int8, 4 per word, up to `N_MAX` bytes |
| `0x2000` | `Y` window | RO | int32 results |
| `0x10000` | `W` window | WO | int8, row-major, up to 64 KB (M0–M2) |

`ERR` is raised if dimensions exceed `M_MAX`/`N_MAX` or `START` arrives while `BUSY`.

M3 adds (no existing register changes): `W_BASE_ADDR`, `W_STRIDE`, and a `SRC` bit selecting on-chip vs DDR3 weights.

## 9. Verification strategy

Three levels, same test vectors at each level:

1. **Golden model** (`software/reference.py`): pure Python/NumPy, `int8` inputs, `int32` accumulation. Includes the group-wise variant.
2. **RTL simulation** (`sim/`, `tests/test_matvec.py`): cocotb with Verilator or Icarus, comparing RTL output with the golden model.
3. **On board** (`hps/`): the same vectors driven from the ARM; mismatches are reported with row/column and expected vs obtained values.

Test vector classes:

- Random matrices with fixed seeds (reproducible).
- Corner cases: all `-128`, all `127`, `-128 × -128` (maximum magnitude), zeros, identity, alternating signs.
- Non-multiples of `LANES` (padding behavior).
- Group sizes: `0`, 16, 64.

Pass criterion: **bit-exact** equality. No tolerance anywhere in the integer path.

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
| Quartus Lite limited SystemVerilog support | Restrict to a conservative subset or use Verilog-2001 |
| Timing closure below target `f_clk` | Pipeline the adder tree, reduce `LANES` |

## 12. References

- Cyclone V Device Handbook, Intel/Altera
- Cyclone V Hard Processor System Technical Reference Manual
- DE10-Nano User Manual, Terasic
- A. Karpathy, llama2.c (`run.c`, `runq.c`)
- RCoeurjoly, LLM2FPGA (related work, compiler-based approach)
- S. Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009)
