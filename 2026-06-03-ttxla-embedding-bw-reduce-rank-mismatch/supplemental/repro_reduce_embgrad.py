# Theseus-free pure-JAX reproducer for the embedding-backward rank-4 / reduce
# rank-mismatch bug on Tenstorrent.
#
# Structure mirrors optax.clip_by_global_norm over the wte gradient: gather
# (embedding fwd) -> grad (scatter / embedding-bw, producing a [V,E] tensor) ->
# jnp.sum(grad*grad) (all-dims reduce to a rank-0 scalar = global_norm sum-of-sq).
#
# BEFORE fix: ttnn::sum over the embedding-bw gradient returns the tensor
#   UNREDUCED (the runtime tensor is rank-4 [1,1,V,E] while the IR reduce uses
#   dim_arg=[0,1], so it reduces only the leading unit dims). The "scalar" then
#   carries the full [V,E] buffer; downstream a binary broadcast reshapes it to
#   [1] -> tt-metal reshape_common.cpp:50 `new_volume == old_volume` abort /
#   segfault.
# AFTER fix: reduces to a true scalar (shape ()); the norm is correct (~181.0).
import os
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import jax, jax.numpy as jnp, numpy as np
print("devices:", jax.devices())
V, E = 100288, 256
emb = jnp.ones((V, E), jnp.float32)
idx = jnp.arange(32, dtype=jnp.int32) % V


def fwd(e):
    g = e[idx]             # [32,E] gather (embedding fwd)
    return jnp.sum(g * g)  # scalar loss


def global_norm_of_grad(e):
    grad = jax.grad(fwd)(e)                 # [V,E] via scatter (embedding-bw)
    return jnp.sqrt(jnp.sum(grad * grad))   # <-- the reduce-to-scalar over the bw grad


try:
    out = jax.jit(global_norm_of_grad)(emb)
    out.block_until_ready()
    ok = (out.shape == ())
    print(f"[{'PASS' if ok else 'WRONG'}] norm over emb-bw grad: "
          f"shape={out.shape} val={float(np.asarray(out)):.4f}")
except Exception as ex:
    print(f"[FAIL] {type(ex).__name__}: {str(ex)[:200]}")
