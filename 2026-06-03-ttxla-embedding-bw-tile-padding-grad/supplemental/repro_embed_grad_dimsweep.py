"""Characterize the TT embedding-backward spurious-gradient pattern across dim.
loss=sum(W[idx]); correct grad row = count(t) in every column. TT adds a
structured spurious offset. Print per-column mean of (tt-cpu) to reveal whether
it's odd-columns (stride-2) or tile-padding related, for several dims.
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


def run(vocab, dim, idx_np):
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((vocab, dim), dtype=np.float32))
    idx = jnp.asarray(idx_np.astype(np.int32))

    def loss(Wm):
        return jnp.sum(Wm[idx])

    g = jax.grad(loss)
    g_cpu = np.array(jax.jit(g, device=cpu)(jax.device_put(W, cpu)))
    g_tt = np.array(jax.jit(g, device=tt)(jax.device_put(W, tt)))
    diff = g_tt - g_cpu                      # spurious offset
    per_col = diff.mean(axis=0)              # mean over rows, per column
    return float(np.max(np.abs(diff))), per_col


def main():
    idx = np.array([2, 2, 2, 5, 5, 9, 0, 0])
    for dim in (1, 2, 4, 8, 16, 32):
        maxd, per_col = run(16, dim, idx)
        cols = np.round(per_col, 3).tolist()
        print(f"vocab=16 dim={dim:2d}  max|tt-cpu|={maxd:.3f}  per-col mean(tt-cpu)={cols}",
              flush=True)


if __name__ == "__main__":
    main()
