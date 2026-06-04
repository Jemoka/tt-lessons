"""Isolate whether optax.global_norm / a full reduce-to-scalar over the wte-sized
tensor [100288,256] reproduces the trainer's `reshape new_volume==old_volume`
(old_logical_volume=25673728 -> target=(1,)) FATAL on TT — and test TT-safe forms.

Adjudicates the open dispute (talk.md): is wte->[1] (a) optax.global_norm reducing
the wte gradient leaf, or (b) a runtime buffer-binding feeding wte to a scalar
reshape? This is theseus-free: if VARIANT A/B FATAL on TT with the wte shape, the
cause is the large-tensor reduce-to-scalar lowering (global_norm), independent of
any trainer donation/aliasing. If they all PASS, the trainer FATAL is NOT this op
-> supports the runtime-binding theory.

Also empirically finds a TT-safe global_norm (variants D/E: staged reductions).

Run (one healthy chip; chip 1 on tt-qb2 per talk.md):
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 \
    python /home/houjun/.agents/repro_global_norm_bigtensor.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import numpy as np
import jax
import jax.numpy as jnp

VOCAB, NEMB = 100288, 256  # the wte/embedding table shape from the dump


def _full_sumsq(x):                       # B: jnp.sum(x**2) -> scalar (full reduce)
    return jnp.sum(jnp.square(x))


def _global_norm_optax(x):                # A: exactly optax.global_norm over one leaf
    import optax
    return optax.global_norm([x])


def _sqrt_full(x):                        # C: sqrt(sum(x*x))
    return jnp.sqrt(jnp.sum(x * x))


def _staged_lastaxis(x):                  # D: reduce axis=-1 then sum (TT-safe candidate)
    return jnp.sqrt(jnp.sum(jnp.sum(jnp.square(x), axis=-1)))


def _staged_axis0(x):                     # E: reduce axis=0 then sum (TT-safe candidate)
    return jnp.sqrt(jnp.sum(jnp.sum(jnp.square(x), axis=0)))


VARIANTS = [
    ("A optax.global_norm([x])", _global_norm_optax),
    ("B jnp.sum(x**2) full", _full_sumsq),
    ("C sqrt(sum(x*x)) full", _sqrt_full),
    ("D staged sum(axis=-1) then sum", _staged_lastaxis),
    ("E staged sum(axis=0) then sum", _staged_axis0),
]


def main():
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}  tensor=[{VOCAB},{NEMB}] vol={VOCAB*NEMB}", flush=True)
    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal((VOCAB, NEMB)).astype(np.float32)) * 1e-3

    for name, fn in VARIANTS:
        x_cpu = jax.device_put(jnp.asarray(x_np), cpu)
        ref = float(jax.jit(fn, device=cpu)(x_cpu))
        # NOTE: a TT_FATAL is a SIGABRT Python can't catch; this breadcrumb prints
        # BEFORE the TT run so the last line before a core dump names the culprit.
        print(f"  >> trying on TT: {name}", flush=True)
        try:
            x_tt = jax.device_put(jnp.asarray(x_np), tt)
            got = float(jax.jit(fn, device=tt)(x_tt))
            rel = abs(got - ref) / (abs(ref) + 1e-30)
            print(f"  {name:34s} TT OK   cpu={ref:.6e} tt={got:.6e} rel={rel:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:34s} TT FAIL  cpu={ref:.6e}  err={repr(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()
