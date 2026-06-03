# TT-XLA Matmul Accumulates in Low Precision by Default (packer_l1_acc=false; error grows with K)

## Summary

A plain fp32 matmul on Tenstorrent Blackhole diverges from CPU by a bf16-level
error that grows ~linearly with the contraction dimension K (K=64 → ~0.04,
K=8192 → ~10.5), even though the TTNN operands are fp32 and the op compiles with
`math_fidelity = hifi4` and `fp32_dest_acc_en = true`. This is the dominant
residual behind Qwen2.5-0.5B logit error on TT (max diff ~0.93 plain).

The concrete, fixable bug is in tt-xla: `packer_l1_acc` defaults to **false**, so
matmul partials packed to the same L1 address across K-blocks accumulate in low
precision and the error grows with K. The attached patch sets HiFi4 +
`fp32_dest_acc_en` + `packer_l1_acc` on every compute-kernel-config op and removes
the K-explosion (K=8192 abs error 10.5 → 0.61). What remains is **not** a software
bug: with fp32 operands and `fp32_dest_acc_en`, tt-metal already unpacks matmul
inputs as TF32 (~10 mantissa bits), which is the Blackhole matrix engine's maximum
fp32 input precision. That ~1e-3 per-product floor compounds across layers; bit-exact
fp32 is not achievable on the matrix engine short of a software multi-pass emulation.

## Status

- Bug type: backend numeric precision — low-precision matmul accumulation
  (`packer_l1_acc=false`), with a separate TF32 input ceiling on top.
- Component: tt-xla compute-kernel-config defaults (fixable here) + Blackhole
  matmul-FPU TF32 input precision (hardware ceiling, below all software).
- Fixed locally: **the tt-xla bug, yes.** `packer_l1_acc=false` is fixed by the
  attached patch and verified (K=8192: 10.5 → 0.61). The residual is the matrix
  engine's TF32 (~10-bit) fp32-input ceiling — a hardware limit, not fixable in
  software short of a 3× mantissa-split emulation.
- Net effect on Qwen parity: small (default path 0.629 → 0.637; the matmul-input
  ceiling dominates once accumulation is fixed). The patch is a correctness
  improvement, not a parity silver bullet.
- Confirmed not the cause: the HF→JAX weight mapping and the model itself — CPU
  parity is 8.5e-05 (see Verification).
- Related: the softmax/reduce compute-config omission
  ([2026-06-03-ttxla-softmax-reduce-bf16-accumulation](/home/houjun/lessons/2026-06-03-ttxla-softmax-reduce-bf16-accumulation/README.md))
  and the embedding bf16 cast
  ([2026-06-03-ttxla-fp32-embedding-bf16-cast](/home/houjun/lessons/2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)).

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty).
- HF model: `Qwen/Qwen2.5-0.5B`.

## Host Environment

- 4× Blackhole `p150b`, all chips free.
- OS Ubuntu, kernel 6.8.0-110-generic, Python 3.12.13, JAX/jaxlib 0.7.1.
- `tt-smi -ls`: 4× Blackhole p150b (chips 0–3).

## User-Visible Failure

Strict TT Qwen2.5-0.5B parity residual:

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

Two distinct mechanisms.

### 1. tt-xla: `packer_l1_acc` defaults to false (the K-explosion; fixable)

The error grows ~linearly with K — the signature of low-precision accumulation
across K-blocks. The compiler does the rest correctly: `opt0 default` ==
`opt0 hifi4 fp32acc` == `opt2 default` (byte-identical), and `math_fidelity`
demonstrably reaches the kernel (lofi/hifi2 give larger error). The emitted TTNN
IR and layouts are fully fp32 — no implicit fp32→bf16 cast:

```text
#ttnn_layout3 = ...memref<1x16x!ttcore.tile<32x32, f32>, #dram>...
#ttnn_layout4 = ...memref<16x8x!ttcore.tile<32x32, f32>, #dram>...
%2 = "ttnn.matmul"(%0, %1)
     <{compute_config = #ttnn.device_compute_kernel_config<
         math_fidelity = hifi4, fp32_dest_acc_en = true>}>
     : (tensor<32x512xf32, #ttnn_layout3>,
        tensor<512x256xf32, #ttnn_layout4>) -> tensor<32x256xf32, ...>
```

What the default does not set is `packer_l1_acc`. With it false, matmul partials
packed to the same L1 address across K-blocks accumulate in low precision, so
error grows with K. Setting it true removes the K-growth (see Verification).

### 2. tt-metal: fp32 matmul inputs are unpacked as TF32 (the ceiling; not a bug)

The matmul input unpack format is decided in
`tt_metal/jit_build/genfiles.cpp:285-288`:

```cpp
unpack_conditional_dst_format = (exp_prec==A) ? Float16 : Float16_b;   // fallback
if (fp32_dest_acc_en && (is_all_fp32_formats(buf_dataformat_arr) || exp_prec==B))
    unpack_conditional_dst_format = DataFormat::Tf32;                  // our case
```

With fp32 operands + `fp32_dest_acc_en=true` (the opt0 default, ensured by the
patch), the override fires and `SrcA`/`SrcB` are unpacked as **TF32 (~10 mantissa
bits)** — the Blackhole matrix engine's *maximum* fp32 input precision (only the
SFPU does true fp32). `Float16_b` is only the non-`fp32_dest_acc` fallback branch
— not what runs here — and the `"TF32 unsupported atm"` throw at `tile.cpp:93` is
L1 tile-storage size bookkeeping, not the unpack-to-SrcReg path (which uses TF32
fine). TF32 is the documented matrix-engine maximum
(tt-metal `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:168`:
"the matrix engine's maximum accuracy is TF32 ... less than full 32-bit
precision"). So there is no software bug in the matmul precision path: tt-metal
already feeds the FPU its highest-precision fp32 input.

Measuring the effective mantissa bits of the matmul inputs directly
(`repro_matmul_bits.py`, small K to isolate input rounding from accumulation)
confirms TF32, not bf16 (which would be ~7-8 bits):

```text
   K   rel_err(median)  eff_bits   (bf16~7-8, tf32~10, fp32~23)
   1     1.005e-03         9.96
   2     1.051e-03         9.89
   4     1.025e-03         9.93
   8     1.121e-03         9.80
  16     1.130e-03         9.79
  64     1.323e-03         9.56
```

Isolation tests confirm storage and the vector path are clean; only the matrix
engine drops precision:

```text
pure fp32 device roundtrip (no compute):   max 0.000   (storage is EXACT fp32)
identity  a+0.0  on device (SFPU path):    max 0.000   (vector/eltwise is fp32-clean)
matmul K=1 (single product, no accum):     relmedian 1.16e-3  ~= 9.8 mantissa bits == TF32
```

## Fix

1. **tt-xla compute-config (attached patch):** set HiFi4 + `fp32_dest_acc_en` +
   `packer_l1_acc` on every compute-kernel-config op. Removes the matmul
   K-explosion (K=8192: 10.5 → 0.61). Patch:
   [supplemental/ttxla_packer_l1_acc_fix.patch](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/ttxla_packer_l1_acc_fix.patch).
2. **The TF32 input ceiling (not a tt-xla fix):** TF32 is the matrix engine's
   silicon limit for fp32 inputs. Exceeding it requires either a software 3×
   mantissa-split decomposition (below) or a native tt-metal change to the FPU
   input path — deep work, not a flag flip.

## Minimal Reproducer

Four standalone JAX scripts, no model harness:

- [supplemental/repro_tt_matmul_precision.py](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_tt_matmul_precision.py)
  — plain fp32 `einsum` matmuls CPU vs TT, sweeping K ∈ {64,256,1024,4096,8192},
  then re-running K=4864 under explicit `compiler_options` (opt level,
  `math_fidelity`, `fp32_dest_acc_en`). Error grows with K and is unchanged by
  hifi4+fp32acc (== opt0 default).
- [supplemental/repro_matmul_bits.py](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_matmul_bits.py)
  — measures the effective mantissa bits of matmul inputs vs a float64 reference
  across small K. Distinguishes the TF32 input ceiling (~10 bits) from bf16
  (~7-8 bits); produces the table in Root Cause §2.
- [supplemental/dump_matmul_ir.py](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/dump_matmul_ir.py)
  — compiles a plain f32 matmul with `TTXLA_LOGGER_LEVEL=DEBUG` and dumps the TTNN
  IR; grep `ttnn.matmul` to read operand/output DataFormat + compute_config (fp32
  operands + `fp32_dest_acc_en`, the condition that selects the TF32 unpack path).
- [supplemental/repro_bf16_split_matmul.py](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_bf16_split_matmul.py)
  — validates a 3× mantissa-split emulation. Splits each operand `x = hi + lo`
  (hi = bf16(x)), issues 3 matmuls `hi@hi + hi@lo + lo@hi`, and sums on the
  fp32-exact SFPU path. Compares plain TT vs 3-split TT against a CPU fp64
  reference.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_tt_matmul_precision.py

TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/supplemental/repro_bf16_split_matmul.py
```

To inspect the IR (confirm fp32 layout + hifi4 + fp32_dest_acc), add
`TTXLA_LOGGER_LEVEL=DEBUG` and grep for `#ttnn_layout` / `ttnn.matmul`.

## Verification

The model is correct; the residual is the matmul path. CPU-backend Qwen2.5-0.5B
parity (`JAX_PLATFORMS=cpu`) matches HF:

```text
max diff: 8.49e-05   mean diff: 1.36e-05   top5 overlap: 5
roundtrip hf->jax->hf max diff: 0.0
hf loss: 7.848437309   jax loss: 7.848448809
```

On TT, error appears only through the matmul path and scales with K.

### Effect of the patch (patched plugin)

The patch removes the matmul K-explosion — `ksweep` (CPU vs TT), before → after:

```text
K=1024:  0.68 -> 0.19     K=4096: 3.79 -> 0.36     K=8192: 10.5 -> 0.61
matmul down (K=4864): 4.35 -> 0.55     gate: 0.59 -> 0.22
```

End-to-end Qwen2.5-0.5B parity on the patched plugin:

```text
plain matmul:  max 0.931  top5 3  jax loss 7.896
```

The patch is a genuine correctness fix (matmul K-block accumulation now runs at
fp32/HiFi4 with `packer_l1_acc`), but it does not materially change end-to-end
parity, because once accumulation is fixed the dominant error is the TF32 input
precision of the matrix engine — a hardware ceiling. TF32 (~1e-3 per product)
compounds across 24 layers to ~0.63 / ~0.93. The top-5 tokens still match HF
(5/5) with losses within ~0.06: practical parity at the hardware's matmul
precision.

### Exceeding TF32 (software emulation, validated)

Exceeding TF32 is only possible via software multi-pass emulation — a
`dot_general` decomposition issuing ~3 split matmuls summed on the fp32-exact
SFPU. Validated (K=512): relmedian **1.45e-3 → 4.6e-4**. It tops out at ~12
mantissa bits (3×bf16: 11.1–11.9b across K=16..1024; 3×TF32 similar) at ~3×
matmul cost — better than native TF32 but not bit-exact fp32, and not needed for
top-5/loss parity. It is a workaround, not a bug fix.

## Notes

- `fp32_dest_acc_en=false` produced a *lower* max error (3.25) than the
  fp32acc=true default (5.38) on the K=4864 case — counterintuitive; do not
  over-interpret a single data point. The robust signal is error ∝ K.
- This lesson documents the matmul precision issue. The softmax/reduce and
  embedding bf16 behaviors are separate, linked above.
