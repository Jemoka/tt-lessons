# TT-XLA Lowers Softmax and Generic Reductions Without an fp32 Compute Config

## Summary

On Tenstorrent Blackhole, an f32 `jax.nn.softmax` diverges from CPU by ~3.9e-2
(6.5% relative), and a bare f32 `reduce_sum` over 512 elements diverges by
~3.7e-2, even though every input and output stays f32 in the IR. Elementwise ops
(`exp`, `silu`, RoPE) are exact to f32 epsilon (~5e-7) and `rms_norm` is tight
(~1.6e-4). The split — reductions bad, elementwise exact, RMSNorm fine — points
at the reduction running in low precision.

The IR-level cause is concrete: the TTIR→TTNN conversion patterns for `softmax`
and the generic reductions (`sum`/`max`/`min`) attach **no `compute_config`**, so
they fall back to TTNN's default `WormholeComputeKernelConfig` (LoFi,
`fp32_dest_acc_en=false` → bf16 accumulation), whereas `ttnn.rms_norm` alone
force-sets HiFi4 + `fp32_dest_acc_en`. That asymmetry is a real lowering omission.
A caveat learned after the fact (see Verification): attaching the same
high-precision config does **not** reduce the reduce/softmax error, because these
ops are input-bound — their operands are already bf16 by the time the reduction
runs, so promoting the accumulator cannot recover lost mantissa. The omission is
worth fixing for correctness, but it is not the lever that closes this gap.

## Status

- Bug type: numeric precision (silent bf16 accumulation in f32 graphs).
- Component: `tt-mlir` TTIR→TTNN conversion (softmax + reduction patterns).
- Fixed locally: **attempted, ineffective.** A HiFi4 + `fp32_dest_acc_en` +
  `packer_l1_acc` compute-config patch was built and deployed but left the
  reduce/softmax op error byte-identical (see Verification). These ops are
  input-bound, so an accumulator-only fix does not help them. The same patch *did*
  fix the matmul K-explosion, which was accumulation-bound — see
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md).
- Closing the reduce/softmax gap needs the deeper fix: keep operands fp32 fed to
  the FPU on Blackhole — a tt-metal kernel property below tt-xla.
- Related but distinct: the embedding op casts its fp32 weight to bf16, which
  shows up as ~1.5e-2 error on a pure gather (no arithmetic). That is a separate
  op workaround, documented in
  [2026-06-03-ttxla-fp32-embedding-bf16-cast](/home/houjun/lessons/2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md).

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, commit
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4`, worktree dirty (unrelated
  in-progress edits to CMakeLists.txt / buffer_instance.h / assert.h).
- `tt-mlir` (submodule) —
  `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit
  `412daacc440f10bb98ccc685c311b01f1fadab70`, worktree dirty (2 files; none touch
  the softmax/reduction lowering).

## Host Environment

Observed on a host with 4× Blackhole `p150b`:

```text
Linux 5.15.0-179-generic x86_64
Python 3.12 (probe venv)
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

Per-op max/mean abs diff, CPU (f32) vs TT (f32 in, f32 out), on Qwen2.5-0.5B
shapes:

```text
[softmax            ] shape=(1, 14, 32, 32)   max=3.869e-02 mean=4.508e-04 relmax=6.469e-02
[rmsnorm            ] shape=(1, 32, 896)       max=1.245e-03 mean=8.504e-05 relmax=1.569e-04
[silu               ] shape=(1, 32, 4864)      max=4.768e-07 mean=1.237e-08 relmax=2.710e-07
[rope               ] shape=(1, 32, 14, 64)    max=4.768e-07 mean=1.260e-08 relmax=8.287e-05
[attn qk-softmax-av ] shape=(1, 14, 32, 64)    max=4.670e-02 mean=5.356e-03 relmax=7.491e+01
[take/embed (gather)] shape=(1, 32, 896)       max=1.553e-02 mean=1.127e-03 relmax=3.889e-03
```

Reading: softmax is ~400× worse (relative) than rms_norm; silu/rope are exact;
the full attention block (matmul→softmax→matmul) has the worst mean. (The gather
error belongs to the separate embedding-cast bug, linked above.)

Standalone JAX controls isolate where softmax loses precision:

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
  `createDeviceComputeKernelConfig` sets `math_fidelity` / `fp32_dest_acc_en`
  only *if the flatbuffer carries them* (`if (config->math_fidelity())`,
  `if (config->fp32_dest_acc_en())`); otherwise the TTNN default
  `WormholeComputeKernelConfig` applies — LoFi fidelity and
  `fp32_dest_acc_en=false`, i.e. bf16 destination accumulation.

By contrast, `ttnn.rms_norm` is rewritten to carry a max-precision config:

- `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/RMSNormConfigRewritePattern.cpp:30-36`
  — force-sets `MathFidelity::HiFi4`, `fp32_dest_acc_en=true`, `packer_l1_acc=true`.

This asymmetry matches the measured behavior: RMSNorm precise, softmax/reduce not.
The StableHLO→TTIR→TTNN type system itself is f32-clean (1:1 type converters at
`StableHLOToTTIRPass.cpp:57-58` and `TTIRToTTNNPass.cpp:61-63`; constants and
ToLayout preserve dtype), so this is a compute-precision default, not a type cast.

## Fix

Proposed: mirror `RMSNormConfigRewritePattern` for the softmax and reduction
lowerings — attach a `DeviceComputeKernelConfigAttr` with `MathFidelity::HiFi4`
and `fp32_dest_acc_en=true` when building `ttnn.softmax` and the `ttnn`
sum/max/min ops (either in the `TTIRToTTNN.cpp` conversion patterns or as
workaround rewrite patterns under
`lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/`).

Caveat: as Verification shows, this restores fp32 accumulation but does **not**
reduce the measured reduce/softmax error, because those ops are input-bound on
Blackhole. The change is a correctness/consistency fix, not a precision win for
these ops on this hardware.

## Minimal Reproducer

[supplemental/repro_softmax_reduce_precision.py](/home/houjun/lessons/2026-06-03-ttxla-softmax-reduce-bf16-accumulation/supplemental/repro_softmax_reduce_precision.py)
— standalone JAX, no model harness. It:

1. Runs f32 `jax.nn.softmax` over attention-shaped logits `[1,14,S,S]` for
   S=32/128/512, CPU vs TT.
2. Runs a control elementwise `exp` (expected exact, ~5e-7).
3. Runs a control bare `reduce_sum` over the last axis (expected bf16-level,
   ~3e-2) — the key isolation.
4. Runs f32 `log_softmax` over vocab=151936 (the loss path).

Expected on the current build: `exp` ≈ f32 epsilon, but `reduce_sum` ≈ 3e-2 —
the reduce accumulator runs at bf16 magnitude.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
export TTXLA_LOGGER_LEVEL=ERROR TTMLIR_RUNTIME_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR
python /home/houjun/lessons/2026-06-03-ttxla-softmax-reduce-bf16-accumulation/supplemental/repro_softmax_reduce_precision.py
```

## Verification

The compute-config patch (HiFi4 + `fp32_dest_acc_en` + `packer_l1_acc` on all
`TTNNComputeKernelConfig` ops) was built, deployed, and the repro re-run on the
patched plugin. The controls are byte-identical before and after:

```text
                       BEFORE (unpatched)            AFTER (patched)
control exp            max 4.768e-07                 max 4.768e-07
control reduce_sum     max 3.727e-02  mean 7.791e-03 max 3.727e-02  mean 7.791e-03
softmax S=32           max 2.024e-03                 max 2.024e-03
log_softmax vocab      max 1.831e-04                 max 1.831e-04
```

The same patch dramatically fixed matmul accumulation (K=8192: 10.5 → 0.61,
K=4096: 3.79 → 0.36), proving the fp32-accumulation config took effect for
matmul. It left `reduce_sum`/`softmax` unchanged. Given the operands reach these
ops already at bf16 magnitude, the best-supported explanation is that they are
input-bound, not accumulation-bound: promoting the accumulator to fp32 cannot
recover precision the inputs already discarded. fp32 accumulation only helps when
the inputs still carry the mantissa, as in the matmul K-explosion case.

Caveat: the patched TTNN IR was not dumped to confirm the generic `ttnn.sum`
lowering actually received the config; the byte-identical result is equally
consistent with the reduce pattern not being covered by the patch. Either way,
the compute-config change is not the lever that closes the reduce/softmax gap.

## Notes

- The reproducer's softmax numbers look small (2e-3 at S=32, shrinking with S)
  only because it feeds tiny logits (~N(0,1)/8) that make softmax nearly uniform;
  realistic-magnitude logits give the full 3.9e-2. The `reduce_sum` control, with
  O(1) summands, is the magnitude-independent signal.
- The `relmax 7.49e+01` on the attention op is an artifact of near-zero reference
  entries; the meaningful figures there are mean 5.4e-3 / max 4.7e-2.
- Both source worktrees were dirty when measured (unrelated in-progress edits);
  none touch the softmax/reduction lowering.
