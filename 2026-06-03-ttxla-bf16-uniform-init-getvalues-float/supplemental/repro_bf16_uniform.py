"""Standalone JAX reproducer: tt-xla compiler aborts lowering a bf16
`jax.random.uniform` (the `tenstorrent.uniform` composite) because its
`low`/`high` bound constants are bf16 and the lowering reads them with
`getValues<float>()`.

A bf16 random-uniform (e.g. a bf16-param weight initializer, or any
`jax.random.uniform(dtype=bfloat16)`) emits bf16 `low`/`high` dense constants.
`TenstorrentUniformToRandConversionPattern` reads them with
`DenseElementsAttr::getValues<float>()`, which asserts unless the element type is
f32 -> MLIR assertion / core dump at compile time. An f32 uniform compiles fine.

Run on a box with the TT PJRT plugin:
  TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu \
      ARCH_NAME=blackhole python repro_bf16_uniform.py

Expected (Blackhole):
  - bf16 uniform  -> BEFORE fix: compiler abort (getValues<float> assertion);
                     AFTER fix: compiles + runs.
  - f32 uniform   -> compiles + runs before and after (the contrast).
Note: the abort is a C++ assert -> SIGABRT, which Python cannot catch; on a stock
plugin the bf16 case aborts the process.
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


def run(name, dtype, tt):
    # minval/maxval become low/high constants of the uniform; in a bf16 uniform
    # they are bf16 dense attrs -> the lowering's getValues<float> asserts.
    def f(key):
        u = jax.random.uniform(key, (64, 64), dtype=dtype, minval=-1.0, maxval=1.0)
        return u.sum()

    key = jax.device_put(jax.random.PRNGKey(0), tt)
    try:
        out = float(jax.jit(f, device=tt)(key))
        print(f"[{name:18s}] TT OK   sum={out:.4f}", flush=True)
    except Exception as e:  # a recoverable error path (if the build returns one)
        print(f"[{name:18s}] TT FAILED: {str(e).splitlines()[0][:120]}", flush=True)


def main():
    tt = jax.devices("tt")[0]
    run("f32 uniform", jnp.float32, tt)     # contrast: always compiles
    run("bf16 uniform", jnp.bfloat16, tt)   # trigger: aborts before the fix


if __name__ == "__main__":
    main()
