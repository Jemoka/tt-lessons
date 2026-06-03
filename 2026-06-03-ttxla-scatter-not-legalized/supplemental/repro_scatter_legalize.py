"""Pure-JAX reproducer: tt-xla cannot legalize stablehlo.scatter (blocks training).

Standalone JAX. A forward gather (jnp.take / advanced indexing) is
fine on TT, but its GRADIENT is a stablehlo.scatter (scatter-add), which tt-xla's
SHLO->TTIR conversion fails to legalize. This breaks on-device training of any
model that gathers in the forward pass (RoPE rotate_half, integer-label
cross-entropy, embedding grads, etc.). A slicing/concatenate formulation has a
scatter-free VJP and compiles fine.

Run on a box with the TT PJRT plugin:
    TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
        ARCH_NAME=blackhole python repro_scatter_legalize.py

Expected on Blackhole (observed 2026-06-03):
  - grad of gather-based fn  -> "failed to legalize operation 'stablehlo.scatter'"
  - grad of slice-based fn   -> OK, matches CPU.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np

tt = jax.devices("tt")[0]
cpu = jax.devices("cpu")[0]

D = 64
x = np.random.default_rng(0).standard_normal((8, D)).astype(np.float32)

# rotate_half via gather: take a permutation of the last-dim indices.
# Forward is a gather; VJP is scatter-add.
half = D // 2
perm = np.concatenate([np.arange(half, D), np.arange(half)]).astype(np.int32)
signs = np.concatenate([-np.ones(D - half), np.ones(half)]).astype(np.float32)


def f_gather(x):
    rotated = jnp.take(x, jnp.asarray(perm), axis=-1)
    return (rotated * jnp.asarray(signs)).sum()


# rotate_half via slicing/concat (scatter-free VJP).
def f_slice(x):
    x1, x2 = x[..., :half], x[..., half:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return rotated.sum()


# embedding gradient (the unavoidable scatter): gather rows by index (embedding
# lookup); VJP scatter-adds gradients back into the rows of W.
W = np.random.default_rng(1).standard_normal((16, 8)).astype(np.float32)


def f_embed(W):
    idx = jnp.asarray(np.array([0, 3, 1, 3, 2, 0, 1, 2], dtype=np.int32))
    return jnp.take(W, idx, axis=0).sum()


def try_grad(name, fn, inp):
    g_cpu = np.asarray(jax.jit(jax.grad(fn), device=cpu)(jax.device_put(jnp.asarray(inp), cpu)))
    try:
        g_tt = np.asarray(jax.jit(jax.grad(fn), device=tt)(jax.device_put(jnp.asarray(inp), tt)))
        diff = float(np.max(np.abs(g_cpu - g_tt)))
        print(f"[{name:16s}] TT grad OK   max|cpu-tt|={diff:.3e}", flush=True)
    except Exception as e:
        msg = str(e).splitlines()[0][:120]
        print(f"[{name:16s}] TT grad FAILED: {msg}", flush=True)


if __name__ == "__main__":
    try_grad("gather (take)", f_gather, x)   # expect: failed to legalize stablehlo.scatter
    try_grad("slice/concat", f_slice, x)     # expect: OK
    try_grad("embed grad", f_embed, W)       # expect: failed (the unavoidable scatter)
