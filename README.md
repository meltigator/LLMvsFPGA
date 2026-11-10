# LLMvsFPGA

A hardware/software experiment: can a small FPGA accelerator beat the ARM core sitting next to it at the core operation of LLM inference?

This project does **not** claim to run "an LLM on an FPGA". It isolates one operation, INT8 matrix-vector multiplication (matvec), which dominates transformer inference, and measures it honestly: ARM Cortex-A9 (software) vs FPGA fabric (Verilog), on the same board, with the same model.

Target board: Terasic DE10-Nano (Intel Cyclone V SoC: dual-core ARM Cortex-A9 + FPGA, 1 GB DDR3).

## Inspiration and Related Work

Running LLMs on FPGAs is an active area. This project takes the opposite route from compiler-based flows:

- **LLM2FPGA** (RCoeurjoly): lowers a PyTorch model to RTL through torch-mlir/CIRCT and an open-source EDA flow. Its own report shows that even TinyStories-1M is far too large for the target device if mapped entirely into logic.
- **llama2.c** (A. Karpathy): minimal C inference for Llama-2-style models, used here as the software baseline and reference.

LLMvsFPGA goes bottom-up: hand-written RTL for one kernel, an ARM host for everything else.

## The Experiment

### Fundamental Concept

Rather than mapping a whole model into logic, we accelerate only the kernel that dominates the cost:

    y = W · x        W: int8 [M×N],  x: int8 [N],  y: int32 [M]

Everything else (RMSNorm, softmax, SiLU, RoPE, sampling, tokenizer) stays on the ARM.

### Hypothesis

LLM inference is memory-bound. Therefore:

    tokens/s ≈ effective DDR3 bandwidth / model size in bytes

If this holds, the FPGA advantage depends less on raw multiply-accumulate throughput than on how well the fabric can stream weights from DDR3. The experiment tests this.

### Numerical Design

- Signed INT8 inputs and weights, signed accumulator.
- Accumulator width ≥ 16 + ceil(log2 N) bits; 32 bits is safe for any realistic N.
- First version: no GGUF, no transformer, no DDR access. Data loaded from software.

## Architecture

    ┌────────────────────┐        ┌──────────────────────────┐
    │  ARM Cortex-A9     │        │  FPGA fabric             │
    │  (HPS, Linux)      │        │                          │
    │                    │ regs   │  ┌────────────────────┐  │
    │  llama2.c host ────┼───────►│  │  matvec (INT8 MAC) │  │
    │  driver (C)        │◄───────┼──│  + local buffers   │  │
    │                    │ result │  └─────────▲──────────┘  │
    └─────────┬──────────┘        │            │ weights     │
              │                   └────────────┼─────────────┘
              └──────────── DDR3 (1 GB) ───────┘

## Roadmap

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | 16×16 matvec, bit-exact vs software reference | planned |
| M1 | Larger matrices, tiling, weights in on-chip RAM | planned |
| M2 | HPS integration: control registers + C driver | planned |
| M3 | Weight streaming from DDR3, bandwidth measurement | planned |
| M4 | One full transformer layer (matvec on FPGA, rest on ARM) | planned |
| M5 | End-to-end tiny model (stories260K, then stories15M) | planned |
| M6 | ARM vs FPGA benchmark report | planned |

**M0 success criterion:** the DE10-Nano computes a 16×16 matvec and passes an automatic test against the software reference.

## Analysis Metrics

- **Correctness**: bit-exact match with the reference (random matrices, automatic diff).
- **Latency**: time per matvec, ARM vs FPGA.
- **Throughput**: tokens/s on the same model and prompt.
- **Bandwidth utilization**: measured DDR3 throughput vs theoretical.
- **Resources**: ALMs, DSP blocks, M10K usage.
- **Efficiency**: performance per unit of resources (and power, if measurable).

## Results

No results yet. This section will only contain measured numbers.

## Repository Structure

    LLMvsFPGA/
    ├── README.md
    ├── docs/architecture.md
    ├── rtl/matvec.sv            # accelerator
    ├── software/reference.py    # golden model
    ├── hps/                     # ARM-side driver (C)
    ├── quartus/                 # project, pin assignments
    ├── sim/                     # simulation scripts
    ├── bench/                   # ARM vs FPGA benchmarks
    └── tests/test_matvec.py     # random-matrix hardware/software comparison

## How to Run

    # Golden reference
    python software/reference.py

    # Automatic test: random matrices, hardware vs software
    pytest tests/

## Theoretical References

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009)
- Eldan, Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (2023)

## Scientific Disclaimer

This is an engineering experiment on a hobbyist-class board. It makes no claim of competing with GPUs or production inference hardware; the goal is to measure where the bottlenecks really are.

## License

MIT License - see LICENSE for details.
