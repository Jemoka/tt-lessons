"""Standalone JAX reproducer: TT casts fp32 embedding weights to bf16.

Two parts, run CPU vs TT:

  1. CONTROL — pure host<->device roundtrip, NO compute. device_put an fp32 array
     to TT and read it back. Exact (~0) means DRAM storage and the upload path are
     fp32-faithful, so any error in part 2 is an op-level cast, not a storage bug.

  2. BUG — an fp32 embedding gather (table[ids], no arithmetic). On TT this routes
     through tt-mlir's embedding workaround, which casts the fp32 weight (and
     output) to bf16 to satisfy tt-metal's BFLOAT16-only embedding kernel. The
     gather therefore returns bf16-rounded values: ~1.5e-2 max error despite doing
     no math. The same lookup on CPU is exact.

bf16 has 8 mantissa bits (~2^-8 = 4e-3 relative), which is the signature printed
for comparison.

Run on a box with the TT PJRT plugin:
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
      ARCH_NAME=blackhole python repro_embedding_bf16_cast.py

Expected (Blackhole p150b, observed 2026-06-03):
  roundtrip max = 0.000e+00 at every shape (storage is exact fp32)
  embedding gather max ~= 1.5e-2 on TT, 0 on CPU (operand cast to bf16)
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
import ml_dtypes


def _bf16_truncate(x_f32):
    # what x would become if forced through bf16 and back
    return np.asarray(x_f32, dtype=ml_dtypes.bfloat16).astype(np.float32)


def roundtrip_control(cpu, tt, rng):
    print("CONTROL: pure roundtrip (device_put fp32 -> read back, NO compute)", flush=True)
    print(f"  {'shape':<22}{'roundtrip max':>16}{'vs-bf16-trunc max':>20}", flush=True)
    for shape in [(32,), (1, 896), (32, 896), (151936, 896), (4864, 896)]:
        x = rng.standard_normal(shape, dtype=np.float32)
        x_tt = jax.device_put(jnp.asarray(x), tt)
        x_back = np.array(jax.device_put(x_tt, cpu), dtype=np.float32)
        rt = float(np.max(np.abs(x_back - x)))
        bf16_gap = float(np.max(np.abs(x_back - _bf16_truncate(x))))
        tag = "  <- looks bf16!" if rt > 1e-3 and bf16_gap < 1e-6 else ""
        print(f"  {str(shape):<22}{rt:>16.3e}{bf16_gap:>20.3e}{tag}", flush=True)


def embedding_bug(cpu, tt, rng):
    print("\nBUG: fp32 embedding gather table[ids] (NO arithmetic)", flush=True)
    vocab, dim, seq = 151936, 896, 32
    table = rng.standard_normal((vocab, dim), dtype=np.float32)
    ids = rng.integers(0, vocab, size=(1, seq)).astype(np.int32)

    gather = lambda w, idx: jnp.take(w, idx, axis=0)
    w_cpu, i_cpu = jax.device_put(jnp.asarray(table), cpu), jax.device_put(jnp.asarray(ids), cpu)
    w_tt, i_tt = jax.device_put(jnp.asarray(table), tt), jax.device_put(jnp.asarray(ids), tt)

    y_cpu = np.array(jax.jit(gather, device=cpu)(w_cpu, i_cpu), dtype=np.float32)
    y_tt = np.array(jax.jit(gather, device=tt)(w_tt, i_tt), dtype=np.float32)

    # the host reference, exact
    ref = table[ids[0]]
    print(f"  gather TT  vs CPU : max={np.max(np.abs(y_tt - y_cpu)):.3e}", flush=True)
    print(f"  gather TT  vs host: max={np.max(np.abs(y_tt - ref)):.3e}", flush=True)
    print(f"  gather TT  vs bf16(host): max={np.max(np.abs(y_tt - _bf16_truncate(ref))):.3e}"
          "  (near 0 == TT returned bf16-rounded values)", flush=True)


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}\n", flush=True)
    rng = np.random.default_rng(0)
    roundtrip_control(cpu, tt, rng)
    embedding_bug(cpu, tt, rng)


if __name__ == "__main__":
    main()
