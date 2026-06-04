"""Minimal theseus-free reproducer for the TT FATAL pinned to optax.clip_by_global_norm.

Primary's runtime trace (talk.md): the wte-sized clipped grad [100288,256] (global_id
1260, vol 25,673,728) is read by a reshape->[1] produced by
optax.clip_by_global_norm.<locals>.clip_fn (_clipping.py:105), inside apply_gradients.
The standalone optax.global_norm repro PASSED, so the bug is in the CLIP-AND-APPLY
chain (global_norm -> div -> rescale of each leaf), not global_norm alone.

This isolates `optax.clip_by_global_norm(1.0).update(grads, state)` on a grad pytree
whose largest leaf is the wte table [100288,256], CPU vs TT. Variants:
  A: full clip_by_global_norm.update  (== the optimizer step; expect the TT FATAL)
  B: the clip MATH inline (norm -> scale=clip/max(norm,clip) -> tree_map(*scale))
     to see whether the FATAL is in optax's exact formulation or any clip-and-apply
  C: a TT-safe clip candidate (compute scale as a 0-d scalar, multiply per leaf,
     avoiding any reshape of a big leaf to [1])

A TT_FATAL is a SIGABRT Python can't catch; each variant prints `>> trying` first so
the last line before a core dump names the culprit.

Run (one healthy chip):
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 \
    python /home/houjun/.agents/repro_clip_by_global_norm.py
"""

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
import optax

VOCAB, NEMB = 100288, 256
CLIP = 1.0


def make_grads(seed=0):
    rng = np.random.default_rng(seed)
    # pytree mimicking real grads: big wte leaf + a couple small leaves
    return {
        "wte": jnp.asarray(rng.standard_normal((VOCAB, NEMB)).astype(np.float32) * 1e-3),
        "ln": jnp.asarray(rng.standard_normal((NEMB,)).astype(np.float32)),
        "head": jnp.asarray(rng.standard_normal((NEMB, 128)).astype(np.float32) * 1e-3),
    }


def variant_A(grads):
    tx = optax.clip_by_global_norm(CLIP)
    state = tx.init(grads)
    updates, _ = tx.update(grads, state)
    return updates["wte"].sum()


def variant_B(grads):
    g = optax.global_norm(grads)
    scale = CLIP / jnp.maximum(g, CLIP)
    updates = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return updates["wte"].sum()


def variant_C(grads):
    # TT-safe candidate: staged norm (no full 2D->scalar single reduce), scalar scale
    def leaf_sq(x):
        return jnp.sum(jnp.sum(jnp.square(x), axis=-1))
    g = jnp.sqrt(sum(leaf_sq(x) for x in jax.tree_util.tree_leaves(grads)))
    scale = CLIP / jnp.maximum(g, CLIP)
    updates = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return updates["wte"].sum()


VARIANTS = [
    ("A optax.clip_by_global_norm.update", variant_A),
    ("B inline clip math", variant_B),
    ("C TT-safe staged clip", variant_C),
]


def main():
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}  wte=[{VOCAB},{NEMB}] vol={VOCAB*NEMB}", flush=True)
    grads = make_grads()
    for name, fn in VARIANTS:
        ref = float(jax.jit(fn, device=cpu)(jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu), grads)))
        print(f"  >> trying on TT: {name}", flush=True)
        try:
            got = float(jax.jit(fn, device=tt)(jax.tree_util.tree_map(lambda x: jax.device_put(x, tt), grads)))
            rel = abs(got - ref) / (abs(ref) + 1e-30)
            print(f"  {name:36s} TT OK   cpu={ref:.6e} tt={got:.6e} rel={rel:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:36s} TT FAIL  cpu={ref:.6e}  err={repr(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
