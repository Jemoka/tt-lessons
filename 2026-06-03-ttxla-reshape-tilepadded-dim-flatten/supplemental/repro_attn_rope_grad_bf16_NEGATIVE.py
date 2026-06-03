"""Standalone JAX reproducer: training-step grad of a minimal multi-head attention
with RoPE rotate_half crashes in ttnn.reshape when n_head is not 32-aligned.

This builds a tiny GPT-style causal-attention block (no model harness) at the
shapes that trigger the bug: n_embd=256, n_head=4 (head_dim=64), seq=128, batch=8.
rotate_half is a `jnp.take(..., axis=-1)` gather over a constant index permutation,
exactly as in the real RoPE. The loss grad emits the attention-jvp reshape that
flattens a tile-padded n_head dim:

    ttnn.reshape  (64x8x128x4, n_head=4 padded to a 32-wide tile) -> (64x4096)
    => TT_FATAL reshape_common.cpp:50  new_volume == old_volume

The bug is layout/context dependent: a bare gather-grad (see
repro_rope_reshape_NEGATIVE.py) does NOT reproduce because tt-mlir picks an
unpadded layout. The attention preprocessing transpose is what places n_head in
the tiled position, so the reproducer needs the attention block, not just the take.

Set n_head to a multiple of 32 (control) and the reshape is tile-aligned -> no FATAL.

Run on a box with the TT PJRT plugin:
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
      ARCH_NAME=blackhole python repro_attn_rope_grad.py

Expected (Blackhole, patched plugin): n_head=4 -> reshape FATAL during the TT grad;
n_head=32 control -> grad runs. (Verify on hardware.)
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


def make_params(rng, n_embd, vocab, seq):
    return {
        "wte": rng.standard_normal((vocab, n_embd), dtype=np.float32) * 0.02,
        "wpe": rng.standard_normal((seq, n_embd), dtype=np.float32) * 0.02,
        "qkv": rng.standard_normal((n_embd, 3 * n_embd), dtype=np.float32) * 0.02,
        "proj": rng.standard_normal((n_embd, n_embd), dtype=np.float32) * 0.02,
        "head": rng.standard_normal((n_embd, vocab), dtype=np.float32) * 0.02,
    }


def rotate_half(x, perm, signs):
    # RoPE rotate_half as a gather over a constant index permutation of head_dim.
    return jnp.take(x, perm, axis=-1) * signs


def loss_fn(params, idx, tgt, n_head, perm, signs):
    B, T = idx.shape
    C = params["wte"].shape[1]
    hd = C // n_head
    x = (params["wte"][idx] + params["wpe"][:T]).astype(jnp.bfloat16)   # bf16 activations (match trainer)
    qkv = x @ params["qkv"].astype(jnp.bfloat16)                                 # (B,T,3C)
    qkv = qkv.reshape(B, T, 3, n_head, hd)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]      # (B,T,nh,hd)
    # apply RoPE rotate_half to q,k (the gather whose VJP emits the reshape)
    q = q + rotate_half(q, perm, signs)
    k = k + rotate_half(k, perm, signs)
    q = q.transpose(0, 2, 1, 3)                             # (B,nh,T,hd)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)       # (B,nh,T,T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(mask, att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v                                             # (B,nh,T,hd)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    y = y.astype(jnp.bfloat16) @ params["proj"].astype(jnp.bfloat16)
    logits = (y.astype(jnp.bfloat16) @ params["head"].astype(jnp.bfloat16)).astype(jnp.float32)                             # (B,T,vocab)
    logp = jax.nn.log_softmax(logits, axis=-1)
    oh = jax.nn.one_hot(tgt, logits.shape[-1])
    return -(oh * logp).sum() / (B * T)


def run(name, n_head, dev, params, idx, tgt):
    C = params["wte"].shape[1]
    hd = C // n_head
    half = hd // 2
    perm = jnp.asarray(np.concatenate([np.arange(half, hd), np.arange(half)]).astype(np.int32))
    signs = jnp.asarray(np.concatenate([-np.ones(hd - half), np.ones(half)]).astype(np.float32))
    g = jax.grad(lambda p: loss_fn(p, idx, tgt, n_head, perm, signs))
    p = jax.device_put(params, dev)
    out = jax.jit(g, device=dev)(p)
    jax.block_until_ready(out)
    return out


def main():
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    B, T, vocab, n_embd = 8, 128, 100288, 256
    idx = jnp.asarray(rng.integers(0, vocab, size=(B, T), dtype=np.int32))
    tgt = jnp.asarray(rng.integers(0, vocab, size=(B, T), dtype=np.int32))

    for name, n_head in (("n_head=4 (non-aligned, trigger)", 4),
                         ("n_head=32 (aligned, control)", 32)):
        params = make_params(rng, n_embd, vocab, T)
        run(name, n_head, cpu, params, idx, tgt)  # CPU reference (always fine)
        try:
            run(name, n_head, tt, params, idx, tgt)
            print(f"[{name}] TT grad OK", flush=True)
        except Exception as e:
            print(f"[{name}] TT grad FAILED: {str(e).splitlines()[0][:120]}", flush=True)


if __name__ == "__main__":
    main()
