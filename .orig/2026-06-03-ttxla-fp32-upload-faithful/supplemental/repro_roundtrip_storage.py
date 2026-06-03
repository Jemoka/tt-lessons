"""Pure host<->device roundtrip: does TT storage truncate fp32 -> bf16?

NO compute. Just device_put an fp32 array to TT and read it back, comparing to
the original host bytes. If the roundtrip is exact (~0), DRAM storage of the raw
buffer is fp32-faithful and any observed bf16 error comes from op/tilize time. If
the roundtrip shows ~1e-2 error, the host->device upload itself truncates fp32 to
bf16 (the suspected storage bug behind the pure-gather ~1.5e-2 finding).

Reference: bf16 has 8 mantissa bits, so round-trip-through-bf16 gives ~2^-8 (4e-3)
relative error; we print that as the expected signature of a bf16 truncation.

Run (chip 1):
  TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 \
    python /home/houjun/.agents/repro_roundtrip_storage.py
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
import ml_dtypes


def _bf16_truncate(x_f32):
    # what x would become if forced through bf16 and back
    return np.asarray(x_f32, dtype=ml_dtypes.bfloat16).astype(np.float32)


def _roundtrip(x_host, cpu, tt):
    # NO compute: device_put to TT, then device_put back to CPU.
    x_tt = jax.device_put(jnp.asarray(x_host), tt)
    x_back = np.array(jax.device_put(x_tt, cpu), dtype=np.float32)
    return x_back


def main() -> None:
    cpu = jax.devices("cpu")[0]
    tt = jax.devices("tt")[0]
    print(f"cpu={cpu}  tt={tt}", flush=True)
    rng = np.random.default_rng(0)

    print("pure roundtrip (device_put fp32 -> read back, NO compute):", flush=True)
    print(f"  {'shape':<22}{'roundtrip max':>16}{'vs-bf16-trunc max':>20}", flush=True)
    for shape in [(32,), (1, 896), (32, 896), (151936, 896), (4864, 896)]:
        x = rng.standard_normal(shape, dtype=np.float32)
        x_back = _roundtrip(x, cpu, tt)
        rt = float(np.max(np.abs(x_back - x)))
        # if storage truncated to bf16, x_back should equal _bf16_truncate(x)
        bf16_gap = float(np.max(np.abs(x_back - _bf16_truncate(x))))
        tag = "  <- looks bf16!" if rt > 1e-3 and bf16_gap < 1e-6 else ""
        print(f"  {str(shape):<22}{rt:>16.3e}{bf16_gap:>20.3e}{tag}", flush=True)

    # control: what bf16 truncation alone would cost on this data
    x = rng.standard_normal((32, 896), dtype=np.float32)
    print(
        f"\n  reference: |x - bf16(x)| max = "
        f"{float(np.max(np.abs(x - _bf16_truncate(x)))):.3e} "
        f"(the bf16-truncation signature)",
        flush=True,
    )


if __name__ == "__main__":
    main()
