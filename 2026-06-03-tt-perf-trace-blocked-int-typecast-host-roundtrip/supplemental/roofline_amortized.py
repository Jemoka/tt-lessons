# Best host-side estimate: unrolled chain of K dependent matmuls in ONE jit, then
# ONE hard sync (np.asarray of a tiny reduction so transfer is negligible). The
# single dispatch + single transfer amortize over K real serial matmuls.
import os, time
os.environ.setdefault("ARCH_NAME","blackhole"); os.environ.setdefault("JAX_PLATFORMS","tt,cpu"); os.environ.setdefault("TT_VISIBLE_DEVICES","0")
import jax, jax.numpy as jnp, numpy as np
print("devices:", jax.devices(), flush=True)
bf16=jnp.bfloat16
K=40
def bench(S):
    a=(jax.random.normal(jax.random.PRNGKey(1),(S,S))*0.03).astype(bf16)
    b=(jax.random.normal(jax.random.PRNGKey(2),(S,S))*0.03).astype(bf16)
    @jax.jit
    def chain(x,w):
        for _ in range(K):
            x=(x@w).astype(bf16)
        return x.astype(jnp.float32).sum()   # reduce to scalar -> transfer is 1 float
    _=np.asarray(chain(a,b))                  # compile+sync
    N=3; t0=time.perf_counter()
    for _ in range(N): r=float(np.asarray(chain(a,b)))   # hard sync on scalar
    per=(time.perf_counter()-t0)/(N*K)
    print(f"S={S:5d}: {per*1e3:8.3f} ms/matmul  {2.0*S**3/per/1e12:8.2f} TFLOP/s  (K={K} chain, scalar sync)", flush=True)
    return 2.0*S**3/per/1e12
print("=== amortized-chain bf16 matmul roofline ===", flush=True)
best=0
for S in [2048,4096,8192]:
    try: best=max(best,bench(S))
    except Exception as e: print(f"S={S} FAIL {str(e)[:70]}",flush=True)
print(f"PEAK_ACHIEVED_BF16 {best:.1f} TFLOP/s | vs 664 fp8: {100*best/664:.1f}% | vs ~332 bf16: {100*best/332:.1f}%", flush=True)
