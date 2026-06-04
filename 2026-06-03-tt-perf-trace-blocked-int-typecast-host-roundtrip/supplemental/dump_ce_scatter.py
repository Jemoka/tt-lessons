# Dump the exact stablehlo.scatter dims for the flattened integer-label CE VJP.
import os
os.environ["ARCH_NAME"]="blackhole"; os.environ["JAX_PLATFORMS"]="cpu"  # CPU: just need the SHLO, no chip
import jax, jax.numpy as jnp
BT=4096; V=8192
def loss(logits, y):
    lp=jax.nn.log_softmax(logits.astype(jnp.float32),axis=-1)
    nll=-jnp.take_along_axis(lp, y[:,None], axis=-1).squeeze(-1)
    return nll.mean()
logits=jnp.zeros((BT,V),jnp.float32); y=jnp.zeros((BT,),jnp.int32)
g=jax.jit(jax.grad(loss))
txt=g.lower(logits,y).as_text()
import re
for ln in txt.splitlines():
    if "scatter" in ln.lower() and ("dimension_numbers" in ln or "scatter(" in ln or "stablehlo.scatter" in ln):
        print(ln.strip()[:300])
