"""Dump the compiled TTNN IR for a plain f32 matmul to inspect operand/output
DataFormat and the matmul's compute_config — to learn why the tf32 unpack path
(which needs all-fp32 CBs + fp32_dest_acc_en) is not triggering on Blackhole.

Runs a tiny f32@f32 matmul on TT with TTXLA_LOGGER_LEVEL=DEBUG so the plugin
dumps vhlo/shlo/ttir/ttnn MLIR to stderr. We capture it; grep the ttnn stage for
the matmul op + its tensor types + any compute config.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
# DEBUG dumps MLIR at every stage to stderr:
os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    tt = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((32, 896), dtype=np.float32))
    b = jnp.asarray(rng.standard_normal((896, 896), dtype=np.float32))
    a = jax.device_put(a, tt)
    b = jax.device_put(b, tt)

    def mm(x, y):
        return x @ y

    print(">>> COMPILING f32 matmul (32,896)@(896,896) on TT <<<", flush=True)
    y = jax.jit(mm, device=tt)(a, b)
    y.block_until_ready()
    print(">>> DONE; dtype:", y.dtype, "shape:", y.shape, flush=True)


if __name__ == "__main__":
    main()
