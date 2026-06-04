"""Standalone smoke: train theseus GPT a few steps on the synthetic train.bin and
print the loss directly (bypasses the trainer's loguru/wandb reporting).

Confirms the gpt pretraining numerics actually go through: real GPT model, real
PMD tokens, finite + decreasing loss.

Usage: JAX_PLATFORMS=cpu uv run python scripts/synthetic_train_smoke.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from theseus.config import patch, configure
from theseus.model.models import GPT

DATA = os.path.expanduser("~/theseus/data/synthetic/train.bin")
VOCAB = 100288
N_EMBD = 256
N_LAYERS = 2
N_HEAD = 4
BLOCK = 128
BATCH = 8
STEPS = 30


def main():
    tokens = np.memmap(DATA, dtype=np.uint32, mode="r")
    print(f"DTYPE param={os.environ.get('SMOKE_PARAM','float32')} act={os.environ.get('SMOKE_ACT','float32')}", flush=True)
    print(f"loaded {len(tokens):,} synthetic tokens from {DATA}", flush=True)

    with patch() as cfg:
        cfg.architecture = OmegaConf.create(
            {
                "n_layers": N_LAYERS,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "block_size": BLOCK,
                "vocab_size": VOCAB,
                "dropout": 0.0,
                "bias": True,
                "rope": True,
                "layer_norm_eps": 1e-5,
                "intermediate_size": -1,
                "dtype": {"param": os.environ.get("SMOKE_PARAM","float32"), "activation": os.environ.get("SMOKE_ACT","float32")},
            }
        )
        model = configure(GPT)
        key = jax.random.PRNGKey(0)
        dummy = jnp.zeros((1, BLOCK), dtype=jnp.int32)
        params = model.init(key, dummy)["params"]

        opt = optax.adamw(3e-4)
        opt_state = opt.init(params)

        def get_batch(rng):
            ix = rng.integers(0, len(tokens) - BLOCK - 1, size=(BATCH,))
            x = np.stack([tokens[i : i + BLOCK].astype(np.int32) for i in ix])
            y = np.stack([tokens[i + 1 : i + 1 + BLOCK].astype(np.int32) for i in ix])
            return jnp.asarray(x), jnp.asarray(y)

        def loss_fn(params, x, y):
            logits, _ = model.apply({"params": params}, x, deterministic=True)
            # Scatter-free cross-entropy (one-hot + log_softmax), matching the
            # theseus model's own loss. The integer-label/gather form
            # (optax.softmax_cross_entropy_with_integer_labels) has a VJP that
            # emits stablehlo.scatter, which tt-xla's SHLO->TTIR cannot legalize.
            logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            onehot = jax.nn.one_hot(y, VOCAB, dtype=jnp.float32)
            return -(onehot * logp).sum(-1).mean()

        @jax.jit
        def step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        rng = np.random.default_rng(0)
        losses = []
        for s in range(STEPS):
            x, y = get_batch(rng)
            params, opt_state, loss = step(params, opt_state, x, y)
            lv = float(loss)
            losses.append(lv)
            if s % 5 == 0 or s == STEPS - 1:
                print(f"step {s:3d} | loss {lv:.4f}", flush=True)

        print(
            f"\nfirst loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}  "
            f"(finite={np.isfinite(losses).all()}, decreased={losses[-1] < losses[0]})",
            flush=True,
        )
        print(f"ln(vocab) reference = {np.log(VOCAB):.4f}", flush=True)


if __name__ == "__main__":
    main()
