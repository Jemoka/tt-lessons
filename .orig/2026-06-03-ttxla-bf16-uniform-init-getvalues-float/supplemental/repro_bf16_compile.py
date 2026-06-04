# Bisect the fully-bf16 compiler abort (DenseElementsAttr::getValues<float> on a
# bf16 dense attr). Escalating cases — find the minimal one that aborts at compile.
import os
os.environ.setdefault("ARCH_NAME","blackhole"); os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("TT_VISIBLE_DEVICES","0")
import jax, jax.numpy as jnp, numpy as np
print("devices:", jax.devices(), flush=True)
bf16 = jnp.bfloat16

def ck(name, fn, *a):
    try:
        r = fn(*a)
        if hasattr(r,"block_until_ready"): r.block_until_ready()
        print(f"[PASS] {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__} {str(e)[:100]}", flush=True)

# 1. bf16 matmul, bf16 inputs (args)
x = jnp.ones((32,64), bf16); w = jnp.ones((64,128), bf16)
ck("bf16 matmul (args)", lambda a,b: jax.jit(lambda p,q: p@q)(a,b), x, w)
# 2. bf16 CONSTANT weight baked into the jit (closure const)
wc = jnp.ones((64,128), bf16)
ck("bf16 matmul (const weight)", lambda a: jax.jit(lambda p: p @ wc)(a), x)
# 3. bf16 const + bias add (bf16 const bias)
bc = jnp.ones((128,), bf16)
ck("bf16 matmul+bias (const)", lambda a: jax.jit(lambda p: p @ wc + bc)(a), x)
# 4. bf16 layernorm-ish: mean/var reductions on bf16
ck("bf16 rmsnorm-ish", lambda a: jax.jit(lambda p: p * jax.lax.rsqrt(jnp.mean(p*p,axis=-1,keepdims=True)+jnp.bfloat16(1e-5)))(a), x)
# 5. grad through bf16 const matmul (training-like)
ck("bf16 grad(matmul const)", lambda a: jax.jit(jax.grad(lambda p: (p@wc).sum().astype(jnp.float32)))(a), x)
# 6. bf16 embedding (gather) — wte is bf16 const
wte = jnp.ones((1000,64), bf16); idx = jnp.arange(8, dtype=jnp.int32)
ck("bf16 embedding gather (const)", lambda: jax.jit(lambda i: jnp.take(wte, i, axis=0))(idx))
