import os

os.environ.setdefault("JAX_PLATFORMS", "tt,cpu")
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("TTXLA_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TTMLIR_RUNTIME_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]

    rng = np.random.default_rng(0)
    x_host = rng.standard_normal((1, 32, 896), dtype=np.float32)

    # This matches a Qwen-like gate projection shape. The transpose is the
    # important part: it yields an F-contiguous host buffer instead of a dense
    # row-major one.
    w_host = rng.standard_normal((4864, 896), dtype=np.float32).T

    print("w C contiguous:", bool(w_host.flags.c_contiguous), flush=True)
    print("w F contiguous:", bool(w_host.flags.f_contiguous), flush=True)
    print("w strides bytes:", w_host.strides, flush=True)

    x_cpu = jax.device_put(jnp.array(x_host), cpu)
    x_tt = jax.device_put(jnp.array(x_host), tt)
    w_cpu = jax.device_put(jnp.array(w_host), cpu)
    w_tt = jax.device_put(jnp.array(w_host), tt)

    w_roundtrip = np.array(jax.device_put(w_tt, cpu))
    print(
        "weight roundtrip max diff:",
        float(np.max(np.abs(w_roundtrip - w_host))),
        flush=True,
    )

    def dense(x, w):
        return jnp.einsum("bth,hf->btf", x, w)

    y_cpu = np.array(jax.jit(dense, device=cpu)(x_cpu, w_cpu))
    y_tt = np.array(jax.jit(dense, device=tt)(x_tt, w_tt))

    diff = np.abs(y_cpu - y_tt)
    print("matmul max diff:", float(np.max(diff)), flush=True)
    print("matmul mean diff:", float(np.mean(diff)), flush=True)


if __name__ == "__main__":
    main()
