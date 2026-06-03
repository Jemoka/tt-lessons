"""Pure-JAX softmax / reduce precision probe: CPU vs Tenstorrent.

Theseus-free reproducer for the suspected tt-xla softmax precision bug.
Hypothesis (from tt-mlir lowering dive, talk.md ~12:10 PDT Jun 3): ttnn.softmax
and generic reduces are lowered with NO compute_config, so they execute at the
TTNN default (LoFi math fidelity, fp32_dest_acc_en=false -> bf16 dest
accumulation), unlike ttnn.rms_norm which is forced to HiFi4+fp32. An f32
softmax should therefore diverge from CPU far more than plain bf16 matmul does.

Shapes are Qwen2.5-0.5B attention: 14 query heads, head_dim 64, and the
vocab=151936 log_softmax used in the loss. Everything stays f32 on the host so
any downcast is the backend's doing.

Run (one chip; pin to a free one):
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 \
    python /home/houjun/.agents/repro_softmax_precision.py
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


def _report(name, a_cpu, a_tt):
    diff = np.abs(np.array(a_cpu, dtype=np.float64) - np.array(a_tt, dtype=np.float64))
    print(
        f"  {name:<28} max diff {float(diff.max()):.3e}   "
        f"mean diff {float(diff.mean()):.3e}",
        flush=True,
    )


def _run_pair(fn, x_host, cpu, tt):
    x_cpu = jax.device_put(jnp.asarray(x_host), cpu)
    x_tt = jax.device_put(jnp.asarray(x_host), tt)
    y_cpu = np.array(jax.jit(fn, device=cpu)(x_cpu))
    y_tt = np.array(jax.jit(fn, device=tt)(x_tt))
    return y_cpu, y_tt


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}", flush=True)
    rng = np.random.default_rng(0)

    # --- 1. attention-shaped f32 softmax over the last (key) axis ---------
    # [batch, n_head, q_seq, k_seq]; logits scaled like real attention scores.
    n_head = 14
    print("f32 softmax over attention scores [1, 14, S, S]:", flush=True)
    for S in (32, 128, 512):
        scores = (rng.standard_normal((1, n_head, S, S), dtype=np.float32)
                  / np.sqrt(64.0)).astype(np.float32)
        y_cpu, y_tt = _run_pair(lambda x: jax.nn.softmax(x, axis=-1), scores, cpu, tt)
        _report(f"S={S}", y_cpu, y_tt)

    # --- 2. control: bare exp (elementwise, no reduce) -------------------
    print("control elementwise exp [1, 14, 128, 128]:", flush=True)
    x = (rng.standard_normal((1, n_head, 128, 128), dtype=np.float32) * 0.5)
    y_cpu, y_tt = _run_pair(lambda z: jnp.exp(z), x, cpu, tt)
    _report("exp", y_cpu, y_tt)

    # --- 3. control: bare reduce_sum over last axis ----------------------
    print("control reduce_sum axis=-1 [1, 14, 512, 512]:", flush=True)
    x = rng.standard_normal((1, n_head, 512, 512), dtype=np.float32)
    y_cpu, y_tt = _run_pair(lambda z: jnp.sum(z, axis=-1), x, cpu, tt)
    _report("reduce_sum", y_cpu, y_tt)

    # --- 4. loss path: f32 log_softmax over vocab ------------------------
    # qwen log_softmax over vocab=151936; seq rows = a short prompt.
    vocab = 151936
    print(f"f32 log_softmax over vocab={vocab} [seq, vocab]:", flush=True)
    for seq in (1, 8):
        logits = rng.standard_normal((seq, vocab), dtype=np.float32)
        y_cpu, y_tt = _run_pair(lambda z: jax.nn.log_softmax(z, axis=-1),
                                logits, cpu, tt)
        _report(f"seq={seq}", y_cpu, y_tt)


if __name__ == "__main__":
    main()
