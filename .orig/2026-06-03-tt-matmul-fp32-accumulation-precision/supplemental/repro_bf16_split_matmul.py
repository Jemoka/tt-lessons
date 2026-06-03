"""Validate 3xTF32 fp32-emulation for matmul on the TT matrix engine.

The Blackhole matmul FPU runs fp32 inputs at TF32 (~10 mantissa bits). NVIDIA-
Ampere-style "3xTF32" recovers ~fp32 by splitting each operand into hi+lo TF32
and summing 3 TF32 matmuls in fp32 (the TT SFPU/eltwise path is fp32-exact):

    a = a_hi + a_lo,  b = b_hi + b_lo      (a_hi,b_hi are TF32; lo are remainders)
    a@b ~= a_hi@b_hi + a_hi@b_lo + a_lo@b_hi   (drop lo@lo)

Each matmul runs at TF32 on TT; the + are fp32 on the vector unit. Compares plain
TT matmul vs 3xTF32 TT matmul against CPU fp32. If 3xTF32 error << plain, the
approach recovers fp32 and is worth implementing as a compiler decomposition.

Usage: TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
       ARCH_NAME=blackhole python tt_3xtf32_validate.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np

tt = jax.devices("tt")[0]
cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(0)


def tf32_hi(x):
    """Round-to-nearest fp32 -> BF16 (7 mantissa bits). The TT matmul UNPACKER
    truncates fp32 SrcA/SrcB to Float16_b (bf16), so the split must be bf16-based:
    a_hi = bf16(a) is then a fixed point under the device matmul, and a_lo's top
    bf16 carries the next 8 mantissa bits. 3 bf16 matmuls recover ~16 bits."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    # round-to-nearest: add half-ulp (2^15) then truncate low 16 bits -> bf16
    u_r = ((u + np.uint64(0x8000)) & np.uint64(0xFFFFFFFFFFFF0000)).astype(np.uint32)
    return u_r.view(np.float32)


def split(x):
    hi = tf32_hi(x)
    lo = (x.astype(np.float32) - hi).astype(np.float32)  # remainder, ~13 low bits
    return hi, lo


def main():
    M, K, N = 64, 512, 256
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    y_cpu = np.asarray(a, np.float64) @ np.asarray(b, np.float64)

    # plain TT matmul (runs at TF32)
    plain = jax.jit(lambda x, y: x @ y, device=tt)
    y_plain = np.asarray(plain(jax.device_put(jnp.asarray(a), tt),
                               jax.device_put(jnp.asarray(b), tt)), np.float64)

    # 3xTF32 on TT: split on host, 3 matmuls + fp32 adds on device
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)

    def mm3(ah, al, bh, bl):
        return (ah @ bh) + (ah @ bl) + (al @ bh)

    mm3_tt = jax.jit(mm3, device=tt)
    y3 = np.asarray(mm3_tt(*[jax.device_put(jnp.asarray(z), tt)
                             for z in (a_hi, a_lo, b_hi, b_lo)]), np.float64)

    def rep(name, y):
        d = np.abs(y_cpu - y)
        denom = np.abs(y_cpu) + 1e-30
        print(f"[{name:18s}] max={d.max():.3e} mean={d.mean():.3e} "
              f"relmedian={np.median(d/denom):.3e}", flush=True)

    rep("plain TT (tf32)", y_plain)
    rep("3xTF32 TT", y3)
    print("CPU is fp64 reference; fp32 round-trip noise floor ~1e-6 to 1e-7.")


if __name__ == "__main__":
    main()
