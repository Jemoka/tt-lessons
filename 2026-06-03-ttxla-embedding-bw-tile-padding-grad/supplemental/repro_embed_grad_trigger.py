"""Isolate the TRIGGER of the TT embedding-backward tile-padding grad leak.

At a non-32-aligned shape (vocab=16, dim=4), the gather VJP leaks a spurious +1.0
into odd columns of even rows of the weight gradient — but only for some index
forms. This probe varies two axes to pin the trigger:

  - index source: COMPILE-TIME CONSTANT (closed over) vs DYNAMIC (jit argument)
  - indexing form: W[idx] (advanced indexing) vs jnp.take(W, idx, axis=0)

Four variants at vocab=16, dim=4, same idx [2,2,2,5,5,9,0,0]:
  A: const idx + W[idx]
  B: const idx + jnp.take
  C: dynamic idx + W[idx]
  D: dynamic idx + jnp.take

Per-column mean(tt-cpu) is printed; the leak shows as ~+1.0 (or +0.5) in odd
columns. Expected: constant-index variants LEAK, dynamic-index variants are CLEAN
— so normal training (dynamic input_ids) is unaffected even at non-aligned shapes.

Run on a box with the TT PJRT plugin:
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
      ARCH_NAME=blackhole python repro_embed_grad_trigger.py
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

VOCAB, DIM = 16, 4
IDX = np.array([2, 2, 2, 5, 5, 9, 0, 0], dtype=np.int32)


def report(name, g_cpu, g_tt):
    g_cpu = np.asarray(g_cpu, np.float64)
    g_tt = np.asarray(g_tt, np.float64)
    col_mean = (g_tt - g_cpu).mean(axis=0)
    print(f"{name:32s} max|cpu-tt|={np.abs(g_tt-g_cpu).max():.3f}  "
          f"per-col mean(tt-cpu)={np.round(col_mean,3).tolist()}", flush=True)


def main():
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}\n", flush=True)
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((VOCAB, DIM), dtype=np.float32))
    idx_arr = jnp.asarray(IDX)

    # A: constant idx, advanced indexing W[idx]
    gA = jax.grad(lambda Wm: jnp.sum(Wm[idx_arr]))
    gA_cpu = np.array(jax.jit(gA, device=cpu)(jax.device_put(W, cpu)))
    gA_tt = np.array(jax.jit(gA, device=tt)(jax.device_put(W, tt)))
    report("A const-idx W[idx]", gA_cpu, gA_tt)

    # B: constant idx, jnp.take
    gB = jax.grad(lambda Wm: jnp.sum(jnp.take(Wm, idx_arr, axis=0)))
    gB_cpu = np.array(jax.jit(gB, device=cpu)(jax.device_put(W, cpu)))
    gB_tt = np.array(jax.jit(gB, device=tt)(jax.device_put(W, tt)))
    report("B const-idx jnp.take", gB_cpu, gB_tt)

    # C: dynamic idx (jit arg), advanced indexing
    def fnC(Wm, ix):
        return jax.grad(lambda w: jnp.sum(w[ix]))(Wm)
    gC_cpu = np.array(jax.jit(fnC, device=cpu)(jax.device_put(W, cpu), jax.device_put(idx_arr, cpu)))
    gC_tt = np.array(jax.jit(fnC, device=tt)(jax.device_put(W, tt), jax.device_put(idx_arr, tt)))
    report("C dyn-idx W[idx]", gC_cpu, gC_tt)

    # D: dynamic idx (jit arg), jnp.take
    def fnD(Wm, ix):
        return jax.grad(lambda w: jnp.sum(jnp.take(w, ix, axis=0)))(Wm)
    gD_cpu = np.array(jax.jit(fnD, device=cpu)(jax.device_put(W, cpu), jax.device_put(idx_arr, cpu)))
    gD_tt = np.array(jax.jit(fnD, device=tt)(jax.device_put(W, tt), jax.device_put(idx_arr, tt)))
    report("D dyn-idx jnp.take", gD_cpu, gD_tt)


if __name__ == "__main__":
    main()
