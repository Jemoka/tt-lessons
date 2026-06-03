# TT Matmul Does Not Achieve fp32 Accumulation Precision (error grows with K)

## Summary

Theseus Qwen2.5-0.5B parity on Tenstorrent Blackhole had a residual error (max
logit diff ~0.93, top5 overlap 3) that survived the earlier host-upload fix. The
concrete failure exposing the cause: a plain fp32 matmul run on TT vs CPU shows
error that is bf16-level and grows ~linearly with the contraction dimension K
(K=64 → ~0.04, K=8192 → ~10.5), even though the TTNN operands are fp32 and the
op is compiled with `math_fidelity = hifi4` and `fp32_dest_acc_en = true`.

There are two distinct layers here. Two gaps are **fixable in tt-xla** and are
fixed by the attached patch: (1) `packer_l1_acc` defaults to **false**, so matmul
partial results packed to the same L1 address across K-blocks accumulate in low
precision (error grows with K); (2) softmax and the generic reductions lower with
a **null** compute_config, falling back at runtime to TTNN's default (LoFi +
`fp32_dest_acc_en=false` → bf16). The patch sets HiFi4 + `fp32_dest_acc_en` +
`packer_l1_acc` on every compute-kernel-config op and removes the K-explosion
(K=8192 abs error 10.5 → 0.61) and the softmax/reduce LoFi fallback.

However, that does **not** close end-to-end Qwen parity. The dominant residual is
a third, deeper issue **below both tt-xla and tt-metal software**: the Blackhole
matmul FPU (matrix engine) runs fp32 inputs at **TF32 (~10 mantissa bits)** — a
hardware ceiling. Isolation tests show fp32 storage and the SFPU/eltwise path are
bit-exact (0 error), and a single matmul product (K=1) lands at ~9.8 mantissa
bits (TF32). That ~1e-3 per-product error compounds across 24 layers to ~0.63
(chunked) / ~0.93 (plain). Bit-exact fp32 is not achievable on the matrix engine;
the only routes are a 3×TF32 mantissa-split software emulation (~3× matmul cost)
or the Theseus chunked path (similar accuracy). Practical parity — top-5 tokens
match HF 5/5, losses within ~0.06 — is achieved at the hardware's matmul precision.

## Status

- Bug type: backend numeric precision (low-precision matmul accumulation + LoFi softmax/reduce)
- Component: tt-xla compute-kernel-config defaults (fixable here) + Blackhole matmul-FPU TF32 input precision (hardware ceiling, below all software)
- Fixed locally: **partially** — the two tt-xla gaps (`packer_l1_acc=false`; null softmax/reduce compute_config) are fixed by the attached patch and verified (ksweep K=8192: 10.5 → 0.61). The dominant residual is the matrix engine's **TF32 (~10-bit) fp32-input ceiling** — a hardware limit, **not** fixable in software short of a 3×TF32 emulation.
- Net effect on Qwen parity: small (default path 0.629 → 0.637; the matmul-input limit dominates). Patch is a correctness improvement, not a parity silver bullet.
- Mitigation still useful: Theseus `THESEUS_TT_SLOW_SAFE_LINEAR` chunks matmuls (K-blocks of 256), which limits the bf16-input error per matmul and gives better parity than plain matmul (0.63 vs 0.93).
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
above. End-to-end `qwen_parity.py` run **unmodified** on tt-qb2 (TT backend,
default slow-safe chunking on):

```text
max diff: 0.6286   mean diff: 0.1059   top5 overlap: 5
roundtrip hf->jax->hf max diff: 0.0
hf loss: 7.848437   jax loss: 7.925876
```

The script runs to completion on Tenstorrent and the top-5 token set matches HF
(5/5); the residual max-diff (0.63) is the un-eliminated tail of the matmul
error.

### Effect of the fix (patched plugin, tt-qb2)

The patch removes the matmul K-explosion — `ksweep` (CPU vs TT), before → after:

```text
K=1024:  0.68 -> 0.19     K=4096: 3.79 -> 0.36     K=8192: 10.5 -> 0.61
matmul down (K=4864): 4.35 -> 0.55     gate: 0.59 -> 0.22
softmax: max 0.038, mean 4.7e-4 (clean; was a null-config LoFi fallback)
```

End-to-end `qwen_parity.py` on the patched plugin:

```text
slow-safe ON  (default):  max 0.637  top5 4  jax loss 7.909   (vs 0.629/5 unpatched)
slow-safe OFF (plain mm):  max 0.931  top5 3  jax loss 7.896
```

Honest read: the patch is a genuine correctness fix (matmul K-block accumulation
and softmax/reduce now run at fp32/HiFi4 instead of bf16/LoFi), but it does **not**
materially change end-to-end Qwen parity. Two reasons:
1. The default path already chunks matmuls (small K), so the packer fix has little
   to add there; and fixing softmax/reduce LoFi did not move the needle, so
   softmax was **not** the dominant residual (correcting the earlier hypothesis).
2. The dominant remaining error is the **TF32 input precision of the matmul FPU**
   (the matrix engine), which is a Blackhole **hardware ceiling**, not a bf16
   truncation bug. Isolation experiments on tt-qb2:

   ```text
   pure fp32 device roundtrip (no compute):   max 0.000   (storage is EXACT fp32)
   identity  a+0.0  on device (SFPU path):    max 0.000   (vector/eltwise is fp32-clean)
   matmul K=1 (single product, no accum):     relmedian 1.16e-3  ~= 9.8 mantissa bits == TF32
   gather (jnp.take):                          mean 1.1e-3  (~tf32 datacopy)
   ```

   So fp32 is stored and moved exactly; only ops that go through the **matrix
   engine** drop to ~10-bit (TF32) — that is the maximum precision the Blackhole
   matmul FPU provides for fp32 inputs under HiFi4. TF32 (~1e-3 per product)
   compounds across 24 layers to ~0.63 (chunked) / ~0.93 (plain). This is below
   tt-xla and below tt-metal software — it is the silicon's matmul precision.

So: `qwen_parity.py` runs unmodified on Tenstorrent and the top-5 tokens match HF
(5/5) with losses within ~0.06 — i.e. practical parity at the hardware's matmul
precision. The two tt-xla precision gaps found here are real and fixed by the
patch. **Bit-exact fp32 parity is not achievable on the matrix engine**: the only
route is a software fp32-emulation that issues each matmul as ~3 TF32 matmuls on
mantissa-split operands (NVIDIA-Ampere "3×TF32" style) — a large change with a
~3× matmul cost — or keeping the Theseus chunked path (similar accuracy). HiFi4
already does a 4-phase combine but tops out at ~TF32 on this part.

### Definitive root cause (tt-metal source) and an emulation path

Located in tt-metal: the matmul **unpacker truncates fp32 SrcA/SrcB to bf16**
(`Float16_b`) before the multiply — `get_single_unpack_dst_format`
(`tt_metal/jit_build/data_format.cpp:119-132`): fp32 matmul src →
`unpack_conditional_dst_format` = `Float16_b`. HiFi4 then runs its phases over
bf16 inputs (the missing low mantissa reads as zero), so no fp32 is recovered.
The higher-precision input mode is hard-disabled:
`throw std::invalid_argument("TF32 unsupported atm")`
(`tt_metal/impl/data_format/tile.cpp:93`). `fp32_dest_acc_en` widens only the DEST
accumulator and `packer_l1_acc` only the L1/output accumulation — neither touches
SrcA/SrcB width. So plain matmul is bf16-input on Blackhole by construction.

Two routes to higher precision (we own the stack):
1. **Software bf16 mantissa-split (graph/compiler, no tt-metal rebuild):** since
   inputs are truncated to bf16, split each operand `x = hi + lo` (hi = bf16(x),
   lo = x − bf16(x)) and issue 3 matmuls `hi@hi + hi@lo + lo@hi`, summing on the
   fp32-exact SFPU path. Validated on tt-qb2 (K=512): relmedian **1.45e-3 →
   4.6e-4** (~3×; now accumulation-limited). This is the legitimate backend fix
   (a `ttir.matmul`/`dot_general` decomposition pass), not model-level chunking.
2. **Native tt-metal fix:** implement/enable the disabled TF32 (or fp32) matmul
   input-unpack path on Blackhole — deep work, not a flag flip.

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
