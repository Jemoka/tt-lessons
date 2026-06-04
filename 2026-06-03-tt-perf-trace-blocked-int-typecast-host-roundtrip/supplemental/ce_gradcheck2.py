# Verify the de-batch scatter fix: integer-label CE (flatten + take_along_axis)
# (1) LEGALIZES on TT now, (2) its grad is BIT-EXACT vs CPU and vs one_hot CE.
import os
os.environ.setdefault("ARCH_NAME","blackhole"); os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("TT_VISIBLE_DEVICES","2")
import jax, jax.numpy as jnp, numpy as np
print("devices:", jax.devices(), flush=True)
M,V=512,8192
rng=np.random.default_rng(0)
logits=jnp.asarray(rng.standard_normal((M,V)).astype(np.float32))
y=jnp.asarray(rng.integers(0,V,size=(M,)).astype(np.int32))
def ce_onehot(lg,yy):
    lp=jax.nn.log_softmax(lg,axis=-1); return -(lp*jax.nn.one_hot(yy,V,dtype=jnp.float32)).sum(-1).mean()
def ce_int(lg,yy):
    lp=jax.nn.log_softmax(lg,axis=-1); return -jnp.take_along_axis(lp,yy[:,None],axis=-1).squeeze(-1).mean()
# CPU reference grads
gc_oh=np.asarray(jax.jit(jax.grad(ce_onehot),backend="cpu")(logits,y))
# TT grads
try:
    g_oh=np.asarray(jax.jit(jax.grad(ce_onehot))(logits,y))
    print(f"[onehot]  TT legalizes; max|tt-cpu|={np.max(np.abs(g_oh-gc_oh)):.3e}",flush=True)
except Exception as e:
    print(f"[onehot]  FAIL {type(e).__name__}: {str(e)[:100]}",flush=True)
try:
    g_int=np.asarray(jax.jit(jax.grad(ce_int))(logits,y))
    print(f"[intlabel] TT LEGALIZES; max|tt-cpu|={np.max(np.abs(g_int-gc_oh)):.3e}  (must be ~0)",flush=True)
except Exception as e:
    print(f"[intlabel] FAIL(legalize) {type(e).__name__}: {str(e)[:140]}",flush=True)
