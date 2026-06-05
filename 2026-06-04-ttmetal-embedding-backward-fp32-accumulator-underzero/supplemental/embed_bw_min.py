"""Minimal: ONE TT compile — grad of an embedding lookup table (embedding_backward).
Tests the generate_zeros_cb fp32 fix directly. Prints embed_bw_norm CPU vs TT.
Before fix: TT = inf. After fix: TT ~= CPU (~3.12e2)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
import numpy as np
import jax
import jax.numpy as jnp

V, D, B, T = 1000, 128, 8, 64
rng = np.random.default_rng(0)
wte0 = jnp.asarray(rng.standard_normal((V, D)).astype(np.float32) * 0.02)
idx0 = jnp.asarray(rng.integers(0, V, size=(B, T)).astype(np.int32))


def embed_loss(wte, idx):
    return jnp.sum(jnp.take(wte, idx, axis=0))


def gn(g):
    return float(jnp.linalg.norm(g.astype(jnp.float32).ravel()))


for name in (["cpu", "tt"] if "tt" in {d.platform for d in jax.devices()} else ["cpu"]):
    dev = jax.devices(name)[0]
    g = jax.jit(jax.grad(embed_loss), device=dev)(
        jax.device_put(wte0, dev), jax.device_put(idx0, dev)
    )
    print(f"[{name}] embed_bw_norm={gn(g):.4e}", flush=True)
print("EMBED_MIN_DONE", flush=True)
