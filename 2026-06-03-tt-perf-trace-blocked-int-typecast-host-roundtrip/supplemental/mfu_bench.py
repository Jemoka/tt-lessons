# Standalone MFU bench: a minimal GPT training step (no model harness) timed on TT.
# For a compute-dense model a step >> the per-op dispatch floor, so plain
# block_until_ready step timing is valid. MFU = achieved / peak.
# Model FLOPs/step (fwd+bwd) ~= 6 * N_params * tokens (+ attention term).
#
# The loss uses one-hot cross-entropy over the big vocab (the workaround for
# integer-label CE, whose gather-VJP emits a batched stablehlo.scatter that fails
# to legalize). That one_hot[B,T,V] + its grad is the dominant per-step overhead
# behind the low training MFU — see the lesson README.
#
# Run on a box with the TT PJRT plugin:
#   TT_VISIBLE_DEVICES=0 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole python mfu_bench.py
import os, time
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
import numpy as np, jax, jax.numpy as jnp, optax

# compute-dense config
N_EMBD, N_LAYERS, N_HEAD, BLOCK, BATCH, VOCAB, STEPS = 1536, 12, 12, 512, 8, 32000, 20
HD = N_EMBD // N_HEAD
bf16 = jnp.bfloat16


def init(rng):
    k = iter(jax.random.split(rng, 3 + 4 * N_LAYERS))
    p = {"wte": jax.random.normal(next(k), (VOCAB, N_EMBD), bf16) * 0.02,
         "wpe": jax.random.normal(next(k), (BLOCK, N_EMBD), bf16) * 0.02,
         "head": jax.random.normal(next(k), (N_EMBD, VOCAB), bf16) * 0.02,
         "blocks": []}
    for _ in range(N_LAYERS):
        p["blocks"].append({
            "qkv": jax.random.normal(next(k), (N_EMBD, 3 * N_EMBD), bf16) * 0.02,
            "proj": jax.random.normal(next(k), (N_EMBD, N_EMBD), bf16) * 0.02,
            "fc": jax.random.normal(next(k), (N_EMBD, 4 * N_EMBD), bf16) * 0.02,
            "fcp": jax.random.normal(next(k), (4 * N_EMBD, N_EMBD), bf16) * 0.02,
        })
    return p


def block(x, w):
    B, T, C = x.shape
    qkv = (x @ w["qkv"]).reshape(B, T, 3, N_HEAD, HD)
    q, kk, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    att = jax.nn.softmax((q @ kk.transpose(0, 1, 3, 2)) / np.sqrt(HD), axis=-1)
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    x = x + y @ w["proj"]
    x = x + jax.nn.gelu(x @ w["fc"]) @ w["fcp"]
    return x


def loss(p, idx, tgt):
    x = p["wte"][idx] + p["wpe"][:idx.shape[1]]
    for w in p["blocks"]:
        x = block(x, w)
    logits = x @ p["head"]
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    oh = jax.nn.one_hot(tgt, VOCAB)            # big-vocab one_hot CE (the overhead)
    return -(oh * lp).sum() / (idx.shape[0] * idx.shape[1])


def main():
    tt = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    p = jax.device_put(init(jax.random.PRNGKey(0)), tt)
    idx = jax.device_put(jnp.asarray(rng.integers(0, VOCAB, (BATCH, BLOCK), np.int32)), tt)
    tgt = jax.device_put(jnp.asarray(rng.integers(0, VOCAB, (BATCH, BLOCK), np.int32)), tt)
    opt = optax.adamw(1e-4)
    st = opt.init(p)

    @jax.jit
    def step(p, st, idx, tgt):
        g = jax.grad(loss)(p, idx, tgt)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st

    p, st = step(p, st, idx, tgt)        # warm up / compile
    jax.block_until_ready(p)
    t0 = time.time()
    for _ in range(STEPS):
        p, st = step(p, st, idx, tgt)
    jax.block_until_ready(p)
    dt = (time.time() - t0) / STEPS

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(p))
    tokens = BATCH * BLOCK
    flops = 6 * n_params * tokens
    print(f"params={n_params/1e6:.1f}M step={dt*1e3:.1f}ms "
          f"achieved={flops/dt/1e12:.2f} TFLOP/s  MFU(vs175)={flops/dt/175e12*100:.2f}%",
          flush=True)


if __name__ == "__main__":
    main()
