import os
os.environ.setdefault("ARCH_NAME","blackhole"); os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("TT_VISIBLE_DEVICES","0")
import jax, jax.numpy as jnp, numpy as np
print("devices:", jax.devices())
a = jax.device_put(jnp.arange(8, dtype=jnp.int32))
# SINGLE eager (non-jit) integer index -> apply_primitive dynamic_slice path
try:
    v = int(a[0])
    print(f"[PASS] eager int(a[0]) = {v}")
except Exception as e:
    print(f"[FAIL] eager int(a[0]): {type(e).__name__} {str(e)[:120]}")
