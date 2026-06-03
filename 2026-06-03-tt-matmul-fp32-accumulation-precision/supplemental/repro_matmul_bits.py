"""Measure the effective mantissa-bit precision of TT matmul INPUTS to settle
tf32 (~10 bits) vs bf16 (~7-8 bits). The compiled IR shows the matmul carries
all-f32 operands + fp32_dest_acc_en, which per tt-metal genfiles.cpp should select
the tf32 unpack path. This probe checks the HW reality.

For each contraction dim K, build f32 a=(M,K), b=(K,N), compute a high-precision
reference in float64 on CPU and the TT result, and report relative error and the
implied effective mantissa bits = -log2(rel_err). Small K isolates INPUT rounding
(little accumulation). Reference signatures: bf16 ~7-8 bits (rel ~4e-3), tf32 ~10
bits (rel ~5e-4), fp32 ~23 bits (rel ~1e-7).

Run (chip 1):
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
    ARCH_NAME=blackhole python repro_matmul_bits.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import math
import jax
import jax.numpy as jnp
import numpy as np


def _bits(rel):
    return float("inf") if rel <= 0 else -math.log2(rel)


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    M, N = 64, 64

    print("matmul input precision: effective mantissa bits = -log2(rel err)", flush=True)
    print(f"  reference: bf16~7-8 bits, tf32~10 bits, fp32~23 bits", flush=True)
    print(f"  {'K':>6}{'rel_err(median)':>18}{'eff_bits(median)':>18}{'rel_err(p99)':>16}", flush=True)
    for K in (1, 2, 4, 8, 16, 64):
        a = rng.standard_normal((M, K), dtype=np.float32)
        b = rng.standard_normal((K, N), dtype=np.float32)
        ref = (a.astype(np.float64) @ b.astype(np.float64))
        a_tt = jax.device_put(jnp.asarray(a), tt)
        b_tt = jax.device_put(jnp.asarray(b), tt)
        y_tt = np.array(jax.jit(lambda x, y: x @ y, device=tt)(a_tt, b_tt),
                        dtype=np.float64)
        denom = np.maximum(np.abs(ref), 1e-6)
        rel = np.abs(y_tt - ref) / denom
        med = float(np.median(rel))
        p99 = float(np.percentile(rel, 99))
        print(f"  {K:>6}{med:>18.3e}{_bits(med):>18.2f}{p99:>16.3e}", flush=True)


if __name__ == "__main__":
    main()
