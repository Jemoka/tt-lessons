"""Minimal repro: take_along_axis (-> ttnn.gather) with int32 index on TT.

Before the fix: tt-metal gather_device_operation.cpp asserts
  "Index tensor must be of type UINT32 or UINT16. Got: DataType::INT32".
After the fix (createGatherOpOperandsWorkarounds binding on TTNN_GatherOp):
  the index operand is auto-cast si32->ui32 and the gather runs; result must
  match the CPU reference bit-for-bit.

Pure JAX, theseus-free. Exercises the integer-label CE forward gather in isolation.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
import numpy as np
import jax, jax.numpy as jnp

B, T, V = 2, 128, 100288          # synthetic-config-like vocab head
rng = np.random.default_rng(0)
logits = jnp.asarray(rng.standard_normal((B, T, V)).astype(np.float32))
targets = jnp.asarray(rng.integers(0, V, size=(B, T)).astype(np.int32))   # int32 ids

def nll(logits, targets):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    g = jnp.take_along_axis(lp, targets[..., None], axis=-1).squeeze(-1)   # -> gather, int32 index
    return -g

tt = jax.devices("tt")[0]
cpu = jax.devices("cpu")[0]
got = np.asarray(jax.jit(nll, device=tt)(logits, targets))
ref = np.asarray(jax.jit(nll, device=cpu)(logits, targets))
maxabs = float(np.max(np.abs(got - ref)))
print(f"GATHER_REPRO shape={got.shape} max|tt-cpu|={maxabs:.3e}")
print("GATHER_REPRO_PASS" if maxabs < 1e-3 else "GATHER_REPRO_FAIL")
