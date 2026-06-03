"""Pure-JAX reproducer: TT matmul does not achieve fp32 accumulation precision.

Theseus-free. Compares CPU vs TT for plain fp32 matmuls and shows that:
  1. Error grows ~linearly with the contraction dimension K (signature of
     low-precision accumulation), even though operands are fp32.
  2. Forcing math_fidelity=hifi4 and fp32_dest_acc_en=true via compiler_options
     does NOT recover fp32 precision (identical to the opt0 default).

Run on a box with the TT PJRT plugin available:

    TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
        ARCH_NAME=blackhole python repro_tt_matmul_precision.py

Expected (Blackhole p150b, observed 2026-06-03):
    ksweep K=64 -> ~0.04, K=1024 -> ~0.68, K=4096 -> ~3.8, K=8192 -> ~10.5
    fidelity opt0-default == hifi4+fp32acc == opt2-default (all identical, large)
A correct fp32 matmul would keep max abs error ~1e-4 regardless of K.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np

cpu = jax.devices("cpu")[0]
tt = jax.devices("tt")[0]
rng = np.random.default_rng(0)
T = 32


def cmp(name, fn, host_args, compiler_options=None):
    cpu_args = [jax.device_put(jnp.asarray(a), cpu) for a in host_args]
    tt_args = [jax.device_put(jnp.asarray(a), tt) for a in host_args]
    y_cpu = np.asarray(jax.jit(fn, device=cpu)(*cpu_args), dtype=np.float64)
    if compiler_options is None:
        y_tt = jax.jit(fn, device=tt)(*tt_args)
    else:
        y_tt = jax.jit(fn, device=tt, compiler_options=compiler_options)(*tt_args)
    y_tt = np.asarray(y_tt, dtype=np.float64)
    d = np.abs(y_cpu - y_tt)
    print(f"[{name:34s}] max={d.max():.6e} mean={d.mean():.6e}", flush=True)


def main():
    mm = lambda a, b: jnp.einsum("tk,kn->tn", a, b)

    print("--- error vs contraction dim K (N=256) ---")
    for K in (64, 256, 1024, 4096, 8192):
        a = rng.standard_normal((T, K), dtype=np.float32)
        b = rng.standard_normal((K, 256), dtype=np.float32)
        cmp(f"ksweep K={K}", mm, (a, b))

    print("--- fidelity/accumulation overrides (K=4864) ---")
    a = rng.standard_normal((T, 4864), dtype=np.float32)
    b = rng.standard_normal((4864, 896), dtype=np.float32)
    for label, opts in [
        ("opt0 default", {"optimization_level": "0"}),
        ("opt0 hifi4 fp32acc", {"optimization_level": "0",
                                "math_fidelity": "hifi4",
                                "fp32_dest_acc_en": "true"}),
        ("opt0 lofi", {"optimization_level": "0", "math_fidelity": "lofi"}),
        ("opt2 default", {"optimization_level": "2"}),
    ]:
        cmp(f"fidelity {label}", mm, (a, b), compiler_options=opts)


if __name__ == "__main__":
    main()
