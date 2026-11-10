#!/usr/bin/env python3
"""LLMvsFPGA - golden reference model for the INT8 matvec accelerator.

Computes y = W . x with signed INT8 inputs and exact INT32 accumulation,
optionally split into per-group integer partial sums (see docs/architecture.md,
section 6.2). This is the bit-exact reference used by the RTL simulation
(tests/test_matvec.py) and by the on-board test (hps/).

Usage:
    python software/reference.py                      # self-check on all test vectors
    python software/reference.py --m 16 --n 16 --seed 1
    python software/reference.py --m 16 --n 64 --group 16 --dump vectors/

No tolerance is used anywhere in the integer path: results must be bit-exact.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


# --------------------------------------------------------------------------
# Core reference
# --------------------------------------------------------------------------

def required_acc_bits(n_acc: int) -> int:
    """Minimum signed accumulator width to sum n_acc INT8 x INT8 products.

    Rule from docs/architecture.md: 16 + ceil(log2(n_acc)).
    """
    if n_acc < 1:
        raise ValueError("n_acc must be >= 1")
    return 16 + math.ceil(math.log2(n_acc))


def _check_inputs(w: np.ndarray, x: np.ndarray, group_size: int) -> None:
    if w.dtype != np.int8 or x.dtype != np.int8:
        raise TypeError(f"W and x must be int8, got {w.dtype} and {x.dtype}")
    if w.ndim != 2 or x.ndim != 1:
        raise ValueError("W must be 2-D [M x N] and x must be 1-D [N]")
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"shape mismatch: W is {w.shape}, x is {x.shape}")
    if group_size < 0:
        raise ValueError("group_size must be >= 0")
    if group_size and w.shape[1] % group_size:
        raise ValueError(f"N={w.shape[1]} is not a multiple of group_size={group_size}")


def matvec(w: np.ndarray, x: np.ndarray, group_size: int = 0) -> np.ndarray:
    """Bit-exact INT8 matvec with INT32 accumulation.

    group_size == 0 : returns y[M]            (plain y = W.x)
    group_size  > 0 : returns y[M, N/group]   (integer partial sum per group)
    """
    _check_inputs(w, x, group_size)
    m, n = w.shape
    prod = w.astype(np.int32) * x.astype(np.int32)[None, :]  # each fits in 16 bit
    if group_size == 0:
        return prod.sum(axis=1, dtype=np.int32)
    return prod.reshape(m, n // group_size, group_size).sum(axis=2, dtype=np.int32)


def matvec_naive(w: np.ndarray, x: np.ndarray, group_size: int = 0) -> np.ndarray:
    """Independent implementation with Python integers (arbitrary precision).

    Slow, deliberately simple. Used to cross-check `matvec` and to verify that
    no result overflows INT32.
    """
    _check_inputs(w, x, group_size)
    m, n = w.shape
    gs = group_size or n
    out = []
    for i in range(m):
        row = []
        for g in range(0, n, gs):
            acc = 0
            for k in range(g, g + gs):
                acc += int(w[i, k]) * int(x[k])
            if not INT32_MIN <= acc <= INT32_MAX:
                raise OverflowError(f"INT32 overflow at row {i}, group starting {g}: {acc}")
            row.append(acc)
        out.append(row)
    y = np.array(out, dtype=np.int32)
    return y[:, 0] if group_size == 0 else y


# --------------------------------------------------------------------------
# Group-wise quantization (llama2.c / runq.c style), used from M4 on
# --------------------------------------------------------------------------

def quantize_groupwise(v: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a float array to INT8 with one float scale per group.

    scale = max(|v|) / 127 per group, q = round(v / scale).
    Returns (q int8 with v's shape, scales float32 with shape [..., n_groups]).
    """
    if v.shape[-1] % group_size:
        raise ValueError("last dimension must be a multiple of group_size")
    g = v.astype(np.float32).reshape(*v.shape[:-1], -1, group_size)
    wmax = np.abs(g).max(axis=-1)
    scale = np.where(wmax > 0, wmax / 127.0, 1.0).astype(np.float32)
    q = np.rint(g / scale[..., None]).clip(INT8_MIN, INT8_MAX).astype(np.int8)
    return q.reshape(v.shape), scale


def scaled_matvec(wq: np.ndarray, ws: np.ndarray, xq: np.ndarray, xs: np.ndarray,
                  group_size: int) -> np.ndarray:
    """Float result from integer group sums: what the ARM does after the FPGA.

    y[i] = sum_g  isum[i, g] * ws[i, g] * xs[g]
    """
    isum = matvec(wq, xq, group_size).astype(np.float32)
    return (isum * ws * xs[None, :]).sum(axis=1, dtype=np.float32)


# --------------------------------------------------------------------------
# Test vectors
# --------------------------------------------------------------------------

def random_case(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.integers(INT8_MIN, INT8_MAX + 1, size=(m, n), dtype=np.int8)
    x = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int8)
    return w, x


def corner_cases(m: int, n: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Deterministic corner cases that stress the datapath."""
    def full(v):  # constant array helpers
        return np.full((m, n), v, dtype=np.int8), np.full(n, v, dtype=np.int8)

    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cases["zeros"] = full(0)
    cases["all_min (-128 x -128, max magnitude)"] = full(INT8_MIN)
    cases["all_max (127 x 127)"] = full(INT8_MAX)
    w_min, _ = full(INT8_MIN)
    _, x_max = full(INT8_MAX)
    cases["-128 x 127 (most negative product)"] = (w_min, x_max)
    sign = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
    cases["alternating signs"] = (
        np.tile(sign * np.int8(127), (m, 1)).astype(np.int8),
        (sign * np.int8(127)).astype(np.int8),
    )
    if m == n:
        cases["identity"] = (np.eye(n, dtype=np.int8),
                             np.arange(n, dtype=np.int64).astype(np.int8) - np.int8(n // 2))
    return cases


# --------------------------------------------------------------------------
# Export for RTL simulation ($readmemh) and on-board tests
# --------------------------------------------------------------------------

def dump_vectors(out_dir: Path, w: np.ndarray, x: np.ndarray, group_size: int) -> None:
    """Write W.hex, x.hex, y.hex (one value per line, two's complement hex)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y = matvec(w, x, group_size).reshape(-1)
    (out_dir / "W.hex").write_text(
        "\n".join(f"{int(v) & 0xFF:02x}" for v in w.reshape(-1)) + "\n")
    (out_dir / "x.hex").write_text(
        "\n".join(f"{int(v) & 0xFF:02x}" for v in x) + "\n")
    (out_dir / "y.hex").write_text(
        "\n".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in y) + "\n")
    (out_dir / "meta.txt").write_text(
        f"M={w.shape[0]}\nN={w.shape[1]}\nGROUP_SIZE={group_size}\n"
        f"ACC_BITS_MIN={required_acc_bits(group_size or w.shape[1])}\n")


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

def _check(name: str, w: np.ndarray, x: np.ndarray, group_size: int) -> bool:
    fast = matvec(w, x, group_size)
    slow = matvec_naive(w, x, group_size)
    ok = fast.dtype == np.int32 and np.array_equal(fast, slow)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        bad = np.argwhere(fast != slow)
        for idx in bad[:5]:
            idx = tuple(int(i) for i in idx)
            print(f"         at {idx}: fast={int(fast[idx])} naive={int(slow[idx])}")
    return ok


def self_check(m: int, n: int, seed: int, group_size: int) -> bool:
    print(f"Reference self-check: M={m} N={n} group_size={group_size} seed={seed}")
    print(f"  minimum accumulator width: {required_acc_bits(group_size or n)} bit "
          f"(design uses 32)")
    ok = True
    for name, (w, x) in corner_cases(m, n).items():
        ok &= _check(name, w, x, group_size)
    for s in range(seed, seed + 5):
        w, x = random_case(m, n, s)
        ok &= _check(f"random seed={s}", w, x, group_size)

    # Consistency: group sums must add up to the plain (whole-row) result.
    if group_size:
        w, x = random_case(m, n, seed)
        consistent = np.array_equal(matvec(w, x, group_size).sum(axis=1, dtype=np.int32),
                                    matvec(w, x, 0))
        print(f"  [{'PASS' if consistent else 'FAIL'}] group sums == whole-row result")
        ok &= consistent

        # Informational (not bit-exact): quantized pipeline vs float matvec.
        rng = np.random.default_rng(seed)
        wf = rng.standard_normal((m, n)).astype(np.float32)
        xf = rng.standard_normal(n).astype(np.float32)
        wq, ws = quantize_groupwise(wf, group_size)
        xq, xs = quantize_groupwise(xf, group_size)
        yq = scaled_matvec(wq, ws, xq, xs, group_size)
        yf = wf @ xf
        err = np.abs(yq - yf).max() / max(np.abs(yf).max(), 1e-9)
        print(f"  [info] quantization error vs float (max, relative): {err:.3%}")

    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=16, help="rows (default 16)")
    p.add_argument("--n", type=int, default=16, help="columns (default 16)")
    p.add_argument("--group", type=int, default=0, help="group size, 0 = whole row")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--dump", type=Path, default=None,
                   help="directory where W.hex, x.hex, y.hex are written")
    args = p.parse_args()

    ok = self_check(args.m, args.n, args.seed, args.group)
    if args.dump is not None:
        w, x = random_case(args.m, args.n, args.seed)
        dump_vectors(args.dump, w, x, args.group)
        print(f"Vectors written to {args.dump}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

