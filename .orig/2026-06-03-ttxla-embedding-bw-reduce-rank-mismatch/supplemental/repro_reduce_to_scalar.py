# Pure-JAX: all-dims reduction of a large 2D tensor to a rank-0 scalar on TT.
# Hypothesis: jnp.sum(big) (reduce over ALL dims -> scalar) does not actually
# reduce on TT; runtime output keeps full volume -> downstream scalar ops get a
# big tensor (mirrors optax global_norm sum-of-squares over wte [100288,256]).
import jax, jax.numpy as jnp
print("devices:", jax.devices())
V, E = 100288, 256
t = jnp.ones((V, E), jnp.float32) * 1.001
exp_sum = float(V * E) * 1.001
exp_sq  = float(V * E) * (1.001 ** 2)

def check(name, fn, expect, *a):
    try:
        o = jax.jit(fn)(*a); o.block_until_ready()
        import numpy as np
        val = float(np.asarray(o).sum())
        print(f"[{ 'PASS' if abs(val-expect) < expect*0.02 else 'WRONG'}] {name}: out_shape={o.shape} sum={val:.1f} expect~{expect:.1f}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")

check("sum(t) all-dims -> scalar", lambda x: jnp.sum(x), exp_sum, t)
check("sum(t*t) all-dims -> scalar", lambda x: jnp.sum(x*x), exp_sq, t)
check("sqrt(sum(t*t)) (global_norm)", lambda x: jnp.sqrt(jnp.sum(x*x)), exp_sq**0.5, t)
check("sum keepdims=False reshape", lambda x: jnp.sum(x, axis=(0,1)), exp_sum, t)
