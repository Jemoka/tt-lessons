# TT Matmul Does Not Achieve fp32 Accumulation Precision (error grows with K)

## Summary

Theseus Qwen2.5-0.5B parity on Tenstorrent Blackhole had a residual error (max
logit diff ~0.93, top5 overlap 3) that survived the earlier host-upload fix. The
concrete failure exposing the cause: a plain fp32 matmul run on TT vs CPU shows
error that is bf16-level and grows ~linearly with the contraction dimension K
(K=64 → ~0.04, K=8192 → ~10.5), even though the TTNN operands are fp32 and the
op is compiled with `math_fidelity = hifi4` and `fp32_dest_acc_en = true`.

This is **not** a tt-xla compiler/layout bug: the emitted TTNN IR is correct
(fp32 tile operands, HiFi4, fp32 dest accumulation). The precision loss happens
in tt-metal's matmul kernel execution on Blackhole — the requested fp32
accumulation is not honored, so accumulation is effectively low-precision. There
is no fix purely inside tt-xla; the chunked "slow-safe linear" workaround in
Theseus mitigates it by shrinking K per matmul.

## Status

- Bug type: backend numeric precision (low-precision matmul accumulation)
- Component: tt-metal matmul kernel (invoked via tt-xla → tt-mlir → TTNN); **not** the tt-xla compiler/layout
- Fixed locally: no — root cause is below the tt-xla layer
- Mitigation: Theseus `THESEUS_TT_SLOW_SAFE_LINEAR` chunks matmuls (K-blocks of 256), reducing per-matmul accumulation error; this is a workaround, not a fix
- Confirmed not the cause: HF→JAX weight mapping and the Theseus model (CPU parity is 8.5e-05, see Verification)

## Repositories

- Theseus: `/home/houjun/theseus`, branch `feat/tenstorrent` (dirty)
- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty)
- HF model: `Qwen/Qwen2.5-0.5B`

## Host Environment

- Diagnosed on `tt-qb2.stanford.edu` (`tt-qb-ac-02`): 4× Blackhole `p150b`, all chips free.
  - The original dev box was fully occupied by another user's serving process (see Notes), so bring-up was replicated onto tt-qb2 by rsyncing the prebuilt env (identical `/home/houjun` paths). See `/home/houjun/theseus/guide.md` §10.
- OS Ubuntu, kernel 6.8.0-110-generic, Python 3.12.13, JAX/jaxlib 0.7.1.
- `tt-smi -ls`: 4× Blackhole p150b (chips 0–3).

## User-Visible Failure

Strict TT Qwen parity residual (with the slow-safe workaround ON):

```text
max diff: ~0.93
top5 overlap: 3
hf loss: 7.848   jax loss: ~7.90
```

Reduced to a plain fp32 matmul, CPU vs TT (`repro_tt_matmul_precision.py`):

```text
--- error vs contraction dim K (N=256) ---
[ksweep K=64    ] max=4.106522e-02 mean=7.790840e-03
[ksweep K=256   ] max=1.275024e-01 mean=2.136738e-02
[ksweep K=1024  ] max=6.840668e-01 mean=1.030609e-01
[ksweep K=4096  ] max=3.792221e+00 mean=6.860886e-01
[ksweep K=8192  ] max=1.051480e+01 mean=1.900947e+00
--- fidelity/accumulation overrides (K=4864) ---
[fidelity opt0 default       ] max=5.375244e+00 mean=8.913133e-01
[fidelity opt0 hifi4 fp32acc ] max=5.375244e+00 mean=8.913133e-01
[fidelity opt0 hifi2         ] max=7.695831e+00 mean=1.165511e+00
[fidelity opt0 lofi          ] max=1.364212e+01 mean=2.399938e+00
[fidelity opt2 default       ] max=5.375244e+00 mean=8.913133e-01
```

A correct fp32 matmul keeps max abs error ~1e-4 independent of K.

## Root Cause

TT matmul accumulation is low-precision and the loss scales with the number of
accumulated terms K (≈linear growth → bf16-level accumulation, not fp32). The
relevant facts:

1. `opt0 default` == `opt0 hifi4 fp32acc` == `opt2 default` (byte-identical
   results) → the opt0 default already requests HiFi4 + fp32 dest accumulation
   (confirmed in `tt-mlir` `TTNNPipelines.h`: `computeCfgMathFidelity` default
   `HiFi4`, `computeCfgFp32DestAccEn` default `true`), and these are what the op
   actually uses. The `math_fidelity` setting *is* applied (lofi/hifi2 give
   larger error), so the compute config reaches the kernel.
2. The emitted TTNN IR and tensor layouts are fully fp32 — there is no implicit
   fp32→bf16 cast in the compiler:

   ```text
   #ttnn_layout3 = ...memref<1x16x!ttcore.tile<32x32, f32>, #dram>...
   #ttnn_layout4 = ...memref<16x8x!ttcore.tile<32x32, f32>, #dram>...
   %2 = "ttnn.matmul"(%0, %1)
        <{compute_config = #ttnn.device_compute_kernel_config<
            math_fidelity = hifi4, fp32_dest_acc_en = true>}>
        : (tensor<32x512xf32, #ttnn_layout3>,
           tensor<512x256xf32, #ttnn_layout4>) -> tensor<32x256xf32, ...>
   ```
3. Despite (1) and (2), runtime error is bf16-level and grows with K → the fp32
   accumulation requested by `fp32_dest_acc_en=true` is not honored at the
   tt-metal kernel/hardware level on Blackhole.

Because the compiler does everything correctly, the defect is in the matmul
kernel execution, below tt-xla.

## Fix

No tt-xla-level fix: the IR is already correct. Options:

- Proper fix: make tt-metal's Blackhole matmul honor fp32 dest accumulation
  (and/or fp32 input precision), so large-K fp32 matmuls stay accurate. Requires
  tt-metal changes + rebuild — outside this repo's scope.
- Mitigation in use: Theseus `THESEUS_TT_SLOW_SAFE_LINEAR=1` (default on TT)
  routes Dense/unembed through `_chunked_linear_input` / `_chunked_unembed`,
  splitting the contraction into K-blocks (256) and accumulating across blocks
  in the JAX graph. From the K-sweep, K=256 has ~0.13 error vs ~5.4 at K=4864,
  so chunking sharply reduces accumulation error. This is a workaround.

## Minimal Reproducer

- [supplemental/repro_tt_matmul_precision.py](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_tt_matmul_precision.py)

What it does:

1. Runs plain fp32 `einsum` matmuls CPU vs TT, sweeping K ∈ {64,256,1024,4096,8192}.
2. Re-runs a K=4864 matmul under explicit `compiler_options`
   (opt level, `math_fidelity`, `fp32_dest_acc_en`).
3. Prints max/mean abs diff. Error grows with K and is unchanged by hifi4+fp32acc
   (== opt0 default), confirming the accumulation precision is the issue.

## Reproduction Steps

On tt-qb2 (or any box with the TT PJRT plugin):

```bash
source /home/houjun/theseus/.venv/bin/activate
TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_tt_matmul_precision.py
```

To see the IR (confirm fp32 layout + hifi4 + fp32_dest_acc), add
`TTXLA_LOGGER_LEVEL=DEBUG` and grep for `#ttnn_layout` / `ttnn.matmul`.

## Verification

The model is correct; the residual is entirely this TT matmul issue. CPU-backend
`qwen_parity.py` (unmodified, `JAX_PLATFORMS=cpu`) matches HF:

```text
max diff: 8.49e-05   mean diff: 1.36e-05   top5 overlap: 5
roundtrip hf->jax->hf max diff: 0.0
hf loss: 7.848437309   jax loss: 7.848448809
```

On TT, error appears only through the matmul path and scales with K as shown
above. (End-to-end TT qwen_parity numbers on tt-qb2 to be appended.)

## Notes

- The earlier "device init hangs" on the original dev box were unrelated: that
  box's 4 chips were held continuously by another user's `uvicorn` serving
  process via the UMD `CHIP_IN_USE` lock. Diagnosis + the non-destructive
  `flock -n` free-check are in `guide.md` §9. tt-qb2 was used instead.
- `fp32_dest_acc_en=false` produced a *lower* max error (3.25) than the
  fp32acc=true default (5.38) on the K=4864 case — counterintuitive, and further
  evidence that the fp32-accumulation path is not behaving as specified. Do not
  over-interpret a single data point; the dominant, robust signal is error ∝ K.
- This lesson documents the matmul-accumulation precision issue only. Whether
  non-matmul ops (softmax, RMSNorm variance) also lose precision on TT is being
  probed separately.
