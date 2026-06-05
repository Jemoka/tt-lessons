"""Same-script control: all-ones vs random cotangent, dynamic idx, V=1000 (non-aligned).
Decisively shows cotangent non-uniformity is the inf trigger (idx is a runtime arg = dynamic)."""
import os
os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("ARCH_NAME","blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY","1")
for k in ("TTXLA_LOGGER_LEVEL","TTMLIR_RUNTIME_LOGGER_LEVEL","TT_METAL_LOGGER_LEVEL"): os.environ.setdefault(k,"ERROR")
import numpy as np, jax, jax.numpy as jnp
V,D,B,T=1000,128,8,64
rng=np.random.default_rng(0)
wte=jnp.asarray(rng.standard_normal((V,D)).astype(np.float32)*0.02)
idx=jnp.asarray(rng.integers(0,V,size=(B,T)).astype(np.int32))
Cones=jnp.ones((B,T,D),jnp.float32)
Crand=jnp.asarray(rng.standard_normal((B,T,D)).astype(np.float32))
def loss(wte,idx,C): return jnp.sum(jnp.take(wte,idx,axis=0)*C)
def gn(g): return float(jnp.linalg.norm(g.astype(jnp.float32).ravel()))
for name in (["cpu","tt"] if "tt" in {d.platform for d in jax.devices()} else ["cpu"]):
    dev=jax.devices(name)[0]; wd,xd=jax.device_put(wte,dev),jax.device_put(idx,dev)
    gf=jax.jit(jax.grad(loss),device=dev)
    go=gf(wd,xd,jax.device_put(Cones,dev)); gr=gf(wd,xd,jax.device_put(Crand,dev))
    print(f"[{name}] V={V} ones-cotangent={gn(go):.4e}  random-cotangent={gn(gr):.4e}",flush=True)
print("EC2_DONE",flush=True)
