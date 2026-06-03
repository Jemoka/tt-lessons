"""Standalone JAX reproducer for the Shardy reshard_to_collectives `isDone()`
assertion that aborts compilation on TT.

Annotating weight kernels for tensor parallelism on a <"batch"=1,"shard"=1> mesh
— e.g. a kernel sharded [{}, {"shard"}] feeding a matmul whose other operand wants
[{"shard"}, {}] — forces an sdy.ReshardOp even with shard=1 (one device). tt-xla
lowers it through Shardy's reshard->collectives pass, which asserts:
reshard_to_collectives.cc:394 `isDone()`.

Run on a box with the TT PJRT plugin:
  CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
      python repro_sdy_reshard.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

devs = jax.devices("tt")
print("tt devices:", devs, flush=True)
mesh = Mesh(np.array(devs[:1]).reshape(1, 1), ("batch", "shard"))

col_sh = NamedSharding(mesh, P(None, "shard"))   # kernel [{}, {"shard"}]
row_sh = NamedSharding(mesh, P("shard", None))   # kernel [{"shard"}, {}]

@jax.jit
def f(w1, w2):
    w1 = jax.lax.with_sharding_constraint(w1, col_sh)
    w2 = jax.lax.with_sharding_constraint(w2, row_sh)
    # matmul forces resharding of one operand on the size-1 "shard" axis
    y = w1 @ w2
    y = jax.lax.with_sharding_constraint(y, row_sh)   # reshard result
    return y.sum()

w1 = jnp.ones((16, 32), dtype=jnp.float32)
w2 = jnp.ones((32, 16), dtype=jnp.float32)
try:
    print("result:", float(f(w1, w2)), flush=True)
    print("RESHARD_OK", flush=True)
except Exception as e:
    print("RESHARD_FAILED:", repr(e)[:200], flush=True)
