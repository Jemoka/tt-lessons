"""Does TT embedding-backward (gather VJP / scatter) ACCUMULATE duplicate indices,
or overwrite? A gather W[idx] has VJP = scatter-add into W; when idx repeats a
token, that row's gradient must be the SUM of all its occurrences. If TT scatters
without accumulation, repeated rows get only ONE contribution -> silently wrong
training gradients.

loss(W) = sum(W[idx]); then dW[t,:] = count of t in idx (exact integers). We pick
idx with known repeat counts and compare CPU vs TT per row.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}", flush=True)

    vocab, dim = 16, 4
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((vocab, dim), dtype=np.float32))
    # idx with deliberate repeats: counts -> {2:3, 5:2, 9:1, 0:2}
    idx = jnp.asarray(np.array([2, 2, 2, 5, 5, 9, 0, 0], dtype=np.int32))
    counts = np.bincount(np.array(idx), minlength=vocab)

    def loss(Wmat):
        return jnp.sum(Wmat[idx])

    g_fn = jax.grad(loss)

    g_cpu = np.array(jax.jit(g_fn, device=cpu)(jax.device_put(W, cpu)))
    g_tt = np.array(jax.jit(g_fn, device=tt)(jax.device_put(W, tt)))

    # each row should equal its count (broadcast across dim). Print ALL rows,
    # including count==0, to localize any spurious/misplaced gradient.
    print("\nrow : count :  cpu_grad[:] :  tt_grad[:] :  match", flush=True)
    for t in range(vocab):
        c = int(counts[t])
        cg = g_cpu[t]
        tg = g_tt[t]
        ok = "OK" if np.max(np.abs(cg - tg)) < 1e-3 else "*** MISMATCH ***"
        print(f"{t:3d} : {c:5d} : {np.round(cg,3).tolist()} : {np.round(tg,3).tolist()} :  {ok}", flush=True)

    maxdiff = float(np.max(np.abs(g_cpu - g_tt)))
    print(f"\nmax|cpu-tt| grad = {maxdiff:.4f}", flush=True)
    # diagnostic: did TT clamp repeated rows to ~1 (overwrite) instead of count?
    rep = [t for t in range(vocab) if counts[t] > 1]
    if rep:
        tt_rep = {t: round(float(g_tt[t, 0]), 3) for t in rep}
        cpu_rep = {t: int(counts[t]) for t in rep}
        print(f"repeated rows  cpu(counts)={cpu_rep}", flush=True)
        print(f"repeated rows  tt(grad)   ={tt_rep}", flush=True)
        if all(abs(float(g_tt[t, 0]) - 1.0) < 1e-3 for t in rep):
            print("VERDICT: TT OVERWRITES (each repeated row = 1.0, not its count) "
                  "-> embedding_bw does NOT accumulate duplicates.", flush=True)
        elif maxdiff < 1e-3:
            print("VERDICT: TT ACCUMULATES correctly (matches CPU counts).", flush=True)
        else:
            print("VERDICT: TT grad differs but not a clean overwrite — inspect above.",
                  flush=True)


if __name__ == "__main__":
    main()
