"""Pure-JAX repro: the gradient of a RoPE-style rotate_half gather (jnp.take on
axis=-1) over a tensor with a NON-32-aligned dim (n_head) triggers tt-metal's
reshape FATAL (reshape_common.cpp: new_volume == old_volume) — a tile-padding
volume mismatch in the take/gather-VJP reshape lowering.

x shape (B, T, N, H) = (8,128,4,64); N=4 is not a multiple of 32. Forward gather
is fine; its grad reshapes-and-flattens B*T*N (incl. the tile-padded N=4) -> FATAL.
Control: make N tile-aligned (32) -> expected clean.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np


def rotate_half_sum(x, perm, signs):
    # multiply by x so the cotangent into the gather is DATA-dependent (not a
    # constant all-ones), preventing XLA from constant-folding the scatter-add
    # off-device. This matches the real trainer, where the gather grad runs on TT.
    rotated = jnp.take(x, perm, axis=-1) * signs
    return (rotated * x).sum()


def try_case(name, B, T, N, H):
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    half = H // 2
    perm = jnp.asarray(np.concatenate([np.arange(half, H), np.arange(half)]).astype(np.int32))
    signs = jnp.asarray(np.concatenate([-np.ones(H - half), np.ones(half)]).astype(np.float32))
    x = jnp.asarray(np.random.default_rng(0).standard_normal((B, T, N, H), dtype=np.float32))
    g = jax.grad(lambda z: rotate_half_sum(z, perm, signs))
    gc = np.asarray(jax.jit(g, device=cpu)(jax.device_put(x, cpu)))
    try:
        gt = np.asarray(jax.jit(g, device=tt)(jax.device_put(x, tt)))
        print(f"[{name}] (B{B},T{T},N{N},H{H}) TT grad OK  max|cpu-tt|={np.max(np.abs(gc-gt)):.3e}", flush=True)
    except Exception as e:
        print(f"[{name}] (B{B},T{T},N{N},H{H}) TT grad FAILED: {str(e).splitlines()[0][:100]}", flush=True)


if __name__ == "__main__":
    try_case("N=4 non-aligned", 8, 128, 4, 64)    # expect FATAL (matches trainer)
    try_case("N=32 aligned   ", 8, 128, 32, 64)   # control: expect OK
