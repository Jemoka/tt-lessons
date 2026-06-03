# TT-XLA Lowers Softmax and Generic Reductions Without an fp32 Compute Config, So They Accumulate in bf16

## Summary

On Tenstorrent Blackhole, an f32 `jax.nn.softmax` diverges from CPU by ~3.9e-2
(6.5% relative), and a bare f32 `reduce_sum` over 512 elements diverges by
~3.7e-2, even though every input/output stays f32 in the IR. Elementwise ops
(`exp`, `silu`, RoPE) are exact to f32 epsilon (~5e-7), and `rms_norm` is tight
(~1.6e-4). The split — reductions bad, elementwise exact, RMSNorm fine — points
at the reduction accumulator running in bf16.

Root cause: the TTIR→TTNN conversion patterns for `softmax` and the generic
reductions (`sum`/`max`/`min`) attach **no `compute_config`**, so at runtime they
fall back to TTNN's default `WormholeComputeKernelConfig` (LoFi math fidelity,
`fp32_dest_acc_en=false` → bf16 destination accumulation). `ttnn.rms_norm` is the
lone exception: a dedicated workaround pattern force-sets HiFi4 + `fp32_dest_acc_en`,
which is exactly why RMSNorm is precise and softmax is not. The fix is to attach
the same high-precision compute config to the softmax and reduction lowerings.

## Status

- Bug type: numeric precision (silent bf16 accumulation in f32 graphs).
- Component: `tt-mlir` TTIR→TTNN conversion (softmax + reduction patterns).
- Fixed locally: **No.** Root cause confirmed on hardware; fix is proposed but
  not yet implemented (the tt-mlir/tt-xla patch is owned by the primary bring-up
  agent). This lesson is the characterization.
- Related but separate: matmul operands appear to be stored bf16 in DRAM (error
  ∝ contraction dim K; also visible as ~1.5e-2 error on a pure embedding
  *gather*, which does no arithmetic). That stored-operand downcast is a distinct
  mechanism from the compute-config issue documented here and is being chased
  separately. Both must be fixed for plain-matmul + softmax Qwen parity with
  `THESEUS_TT_SLOW_SAFE_LINEAR=0`.

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, commit
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4`, worktree dirty (9 files; unrelated
  in-progress edits to CMakeLists.txt / buffer_instance.h / assert.h).
- `tt-mlir` (submodule) —
  `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit
  `412daacc440f10bb98ccc685c311b01f1fadab70`, worktree dirty (2 files).
- `theseus` — `/home/houjun/theseus`, branch `feat/tenstorrent`, commit
  `f085ca67fa68ef08d63668cd7f866b2b8147839e` (used only to source the op-probe
  shapes; the reproducer below is theseus-free).

## Host Environment

Observed on `tt-qb-ac-02` (tt-qb2.stanford.edu):

```text
Linux 5.15.0-179-generic x86_64
Python 3.10.12 (system); probe run under theseus uv venv Python 3.12
jax==0.7.1, jaxlib==0.7.1
ARCH_NAME=blackhole
```

`tt-smi -ls` board inventory:

```text
UMD Chip ID  Board Type  Device Series
0            Blackhole   p150b
1            Blackhole   p150b
2            Blackhole   p150b
3            Blackhole   p150b
```

Probe pinned to chip 1 via `TT_VISIBLE_DEVICES=1`. The
`TT_FATAL: ... logical eth core ... connects to a remote mmio device` lines at
startup are benign single-chip warnings; execution completes normally.

## User-Visible Failure

Per-op max/mean abs diff, CPU (f32) vs TT (f32 in, f32 out),
`scripts/tt_op_probe.py softmax rmsnorm silu rope attn gather`:

```text
[softmax            ] shape=(1, 14, 32, 32)   max=3.869e-02 mean=4.508e-04 relmax=6.469e-02
[rmsnorm            ] shape=(1, 32, 896)       max=1.245e-03 mean=8.504e-05 relmax=1.569e-04
[silu               ] shape=(1, 32, 4864)      max=4.768e-07 mean=1.237e-08 relmax=2.710e-07
[rope               ] shape=(1, 32, 14, 64)    max=4.768e-07 mean=1.260e-08 relmax=8.287e-05
[attn qk-softmax-av ] shape=(1, 14, 32, 64)    max=4.670e-02 mean=5.356e-03 relmax=7.491e+01
[take/embed (gather)] shape=(1, 32, 896)       max=1.553e-02 mean=1.127e-03 relmax=3.889e-03
```

Reading: softmax is ~400× worse (relative) than rms_norm; silu/rope are exact;
the full attention block (matmul→softmax→matmul) has the worst mean; the pure
gather shows bf16-magnitude error despite doing no arithmetic.

Pure-JAX controls isolate where softmax loses precision:

```text
control exp        [1,14,128,128]:  max 4.768e-07            (= f32 eps; EXACT)
control reduce_sum [1,14,512,512]:  max 3.727e-02  mean 7.791e-03
```

`exp` is exact; a bare `reduce_sum` loses 3.7e-2. The reduction, not the
exponential, carries the error.

## Root Cause

In `tt-mlir`, the TTIR→TTNN lowering builds `ttnn.softmax` and the generic TTNN
reductions with no `compute_config` attribute:

- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:558-570` — `SoftmaxOpConversionPattern`
  creates `ttnn::SoftmaxOp` from `input`, `dimension`, `numericStable` only; no
  compute config is passed.
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:283-296` — `ReductionOpConversionPattern`
  (sum/max/min) likewise attaches no compute config.

At runtime, a null compute config falls back to TTNN's default kernel config:

- `runtime/lib/ttnn/operations/utils/utils.cpp:427-442` —
  `createDeviceComputeKernelConfig` only sets `math_fidelity` /
  `fp32_dest_acc_en` *if the flatbuffer carries them* (`if (config->math_fidelity())`,
  `if (config->fp32_dest_acc_en())`); otherwise the TTNN default
  `WormholeComputeKernelConfig` applies — LoFi math fidelity and
  `fp32_dest_acc_en=false`, i.e. **bf16 destination accumulation**.

By contrast, `ttnn.rms_norm` is rewritten to carry a max-precision config:

- `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/RMSNormConfigRewritePattern.cpp:30-36`
  — force-sets `MathFidelity::HiFi4`, `fp32_dest_acc_en=true`, `packer_l1_acc=true`.

This asymmetry exactly matches the measured behavior: RMSNorm precise, softmax /
reduce not. The StableHLO→TTIR→TTNN type system itself is f32-clean (1:1 type
converters at `StableHLOToTTIRPass.cpp:57-58` and `TTIRToTTNNPass.cpp:61-63`;
constants and ToLayout preserve dtype), so this is a compute-precision default,
not a type cast.

## Fix

Not yet implemented. Proposed: mirror `RMSNormConfigRewritePattern` for the
softmax and reduction lowerings — attach a `DeviceComputeKernelConfigAttr` with
`MathFidelity::HiFi4` and `fp32_dest_acc_en=true` when building `ttnn.softmax`
and the `ttnn` sum/max/min ops (either in the conversion patterns in
`TTIRToTTNN.cpp`, or as analogous workaround rewrite patterns under
`lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/`). That restores fp32
accumulation for the reduction, the same way RMSNorm already gets it.

This is independent of the separate "matmul/stored operands are bf16 in DRAM"
issue; fixing the compute config alone will not recover matmul precision.

## Minimal Reproducer

`/home/houjun/lessons/2026-06-03-ttxla-softmax-reduce-bf16-accumulation/supplemental/repro_softmax_reduce_precision.py`
(theseus-free pure JAX). It:

1. Runs f32 `jax.nn.softmax` over attention-shaped logits `[1,14,S,S]` for
   S=32/128/512, CPU vs TT.
2. Runs a control elementwise `exp` (expected exact, ~5e-7).
3. Runs a control bare `reduce_sum` over the last axis (expected bf16-level,
   ~3e-2) — the key isolation.
4. Runs f32 `log_softmax` over vocab=151936 (the loss path).

Expected on the buggy build: `exp` ≈ f32 epsilon, but `reduce_sum` ≈ 3e-2 —
proving the reduce accumulator is bf16. After the proposed fix, `reduce_sum`
(and softmax) should drop toward f32 epsilon.

## Reproduction Steps

On a host with a free Blackhole chip (here tt-qb2 chip 1):

```bash
cd /home/houjun/theseus
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
export TTXLA_LOGGER_LEVEL=ERROR TTMLIR_RUNTIME_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR

# op-level survey (shapes from Qwen2.5-0.5B):
python scripts/tt_op_probe.py softmax rmsnorm silu rope attn gather

# pure-JAX isolation (exp vs reduce_sum):
python /home/houjun/lessons/2026-06-03-ttxla-softmax-reduce-bf16-accumulation/supplemental/repro_softmax_reduce_precision.py
```

## Verification

Before the fix (build at the SHAs above), the controls show:

```text
control exp        [1,14,128,128]:  max 4.768e-07
control reduce_sum [1,14,512,512]:  max 3.727e-02  mean 7.791e-03
```

i.e. elementwise exact, reduction bf16-level. After attaching a
HiFi4 + `fp32_dest_acc_en=true` compute config to the softmax/reduction
lowering, `reduce_sum` and `softmax` are expected to fall toward f32 epsilon
(rms_norm, which already carries that config, measures 1.6e-4 relative as the
reference for "fixed"). This section will be updated with after-fix numbers once
the patch lands.

## Notes

- The reproducer's own softmax numbers look small (2e-3 at S=32, shrinking with
  S) only because it feeds tiny logits (~N(0,1)/8) that make softmax nearly
  uniform; the op-probe's realistic-magnitude logits give the full 3.9e-2. The
  reduce_sum control, with O(1) summands, is the magnitude-independent signal.
- The `relmax 7.49e+01` on the attention op is an artifact of near-zero reference
  entries; the meaningful figures there are mean 5.4e-3 / max 4.7e-2.
- The pure-gather (`take/embed`) error of 1.5e-2 is *not* a reduction issue — a
  gather does no math — and is evidence for the separate stored-operand-bf16
  bug; it is noted here only to keep the two mechanisms distinct.
- Both source worktrees were dirty when measured (unrelated in-progress edits);
  none touch the softmax/reduction lowering, so the characterization is
  unaffected.
