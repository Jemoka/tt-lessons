# Compute-dense GPT MFU bench. For a big model a step >> dispatch floor, so
# block_until_ready step timing is valid. MFU = achieved / peak.
# Model FLOPs/step (fwd+bwd) ~= 6 * N_params * tokens (+ attention term).
import os, time
os.environ.setdefault("ARCH_NAME","blackhole"); os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("TT_VISIBLE_DEVICES","0")
import numpy as np, jax, jax.numpy as jnp, optax
from omegaconf import OmegaConf
from theseus.config import patch, configure
from theseus.model.models import GPT

# compute-dense config (vs synthetic 256/2/128)
N_EMBD=1536; N_LAYERS=12; N_HEAD=12; BLOCK=512; BATCH=8; VOCAB=32000; STEPS=20
print(f"cfg: d={N_EMBD} L={N_LAYERS} seq={BLOCK} batch={BATCH} vocab={VOCAB}", flush=True)

def main():
    with patch() as cfg:
        cfg.architecture = OmegaConf.create({
            "n_layers":N_LAYERS,"n_embd":N_EMBD,"n_head":N_HEAD,"block_size":BLOCK,
            "vocab_size":VOCAB,"dropout":0.0,"bias":True,"rope":True,"layer_norm_eps":1e-5,
            "intermediate_size":-1,
            "dtype":{"param":"float32","activation":"bfloat16"}})
        model=configure(GPT)
        key=jax.random.PRNGKey(0)
        params=model.init(key, jnp.zeros((1,BLOCK),jnp.int32))["params"]
        nparams=sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"params: {nparams/1e6:.1f}M", flush=True)
        opt=optax.adamw(3e-4); opt_state=opt.init(params)
        rng=np.random.default_rng(0)
        def gb():
            x=rng.integers(0,VOCAB,size=(BATCH,BLOCK)).astype(np.int32)
            return jnp.asarray(x), jnp.asarray(x)
        def loss_fn(p,x,y):
            logits,_=model.apply({"params":p},x,deterministic=True)
            lp=jax.nn.log_softmax(logits.astype(jnp.float32),axis=-1)
            oh=jax.nn.one_hot(y,VOCAB,dtype=jnp.float32)
            return -(oh*lp).sum(-1).mean()
        @jax.jit
        def step(p,os_,x,y):
            l,g=jax.value_and_grad(loss_fn)(p,x,y)
            u,os2=opt.update(g,os_,p); return optax.apply_updates(p,u),os2,l
        x,y=gb()
        p,os_,l=step(params,opt_state,x,y); l.block_until_ready()   # compile
        ts=[]
        for _ in range(STEPS):
            x,y=gb(); t0=time.perf_counter()
            p,os_,l=step(p,os_,x,y); l.block_until_ready()
            ts.append(time.perf_counter()-t0)
        med=float(np.median(ts)); tok=BATCH*BLOCK
        flops=6.0*nparams*tok       # standard fwd+bwd approx (excl attention)
        ach=flops/med/1e12
        print(f"median step: {med*1e3:.1f} ms | {tok/med:.0f} tok/s | achieved {ach:.1f} TFLOP/s (6ND)", flush=True)
        print(f"MFU vs 110 (bf16 matmul roofline): {100*ach/110:.1f}% | vs 332 bf16 peak: {100*ach/332:.1f}% | vs 664 fp8: {100*ach/664:.1f}%", flush=True)
main()
