# TT-XLA Casts fp32 Embedding Weights to bf16 (Pure Gather Loses ~1.5e-2)

## Summary

A pure fp32 embedding *gather* on Tenstorrent Blackhole — `table[ids]`, no
arithmetic — returns values off by ~1.5e-2 from the host source. The cause is an
op-level operand cast: tt-metal's embedding kernel requires a BFLOAT16 weight
(`TT_FATAL`), and tt-mlir's `TTNNWorkaroundsPass` satisfies it by casting the
fp32 embedding weight and output to bf16. The lookup therefore returns
bf16-rounded values even though it does no math.

This was initially suspected to be a host→device upload that truncates fp32 to
bf16 in DRAM. A pure roundtrip probe (`device_put` fp32 → read back, no compute)
rules that out: the roundtrip is bit-exact (0.0) at every size including the
151936×896 embedding-table shape. Storage and upload preserve fp32; the loss is
the embedding op's bf16 operand workaround, not a storage bug.

## Status

- Bug type: numeric precision (op-level fp32→bf16 operand cast).
- Component: `tt-mlir` `TTNNWorkaroundsPass` embedding workaround; backed by
  tt-metal's BFLOAT16-only embedding kernel.
- Fixed locally: no. The cast is forced by a tt-metal `TT_FATAL`; recovering fp32
  embeddings needs an fp32-capable embedding kernel in tt-metal.
- Ruled out: host→device upload / DRAM storage truncation — verified fp32-faithful
  on hardware (see Root Cause #1).
- Distinct from: the matmul bf16-input precision issue, which is an FPU
  unpack/SrcReg matter, not storage — see
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md).
- Related (same embedding op, backward side): the embedding *gradient* tile-padding
  leak in
  [2026-06-03-ttxla-embedding-bw-tile-padding-grad](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md).
  This lesson is the forward bf16 cast; that one is the backward grad leak — same
  op, different failure, not a duplicate.

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, commit
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4` (worktree dirty, unrelated).
- `tt-mlir` (submodule) —
  `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit
  `412daacc440f10bb98ccc685c311b01f1fadab70`.
- `tt-metal` (nested submodule) — under
  `tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`.

## Host Environment

Linux 5.15.0-179-generic x86_64, 4× Blackhole p150b. jax/jaxlib 0.7.1,
`ARCH_NAME=blackhole`. Probe pinned to chip 1 (`TT_VISIBLE_DEVICES=1`). The
startup `TT_FATAL ... remote mmio device` lines are benign single-chip warnings;
execution completes.

## User-Visible Failure

A pure embedding gather of fp32 data shows bf16-magnitude error despite doing no
arithmetic:

```text
[take/embed (gather)] shape=(1, 32, 896)   max=1.553e-02 mean=1.127e-03
```

## Root Cause

Two checks: the upload path is fp32-faithful, and the gather error comes from an
op operand cast.

1. **The upload/storage path is exact** — a HW roundtrip with no compute is
   bit-exact at all sizes:

   ```text
   pure roundtrip (device_put fp32 -> read back, NO compute):
     shape                    roundtrip max   vs-bf16-trunc max
     (32,)                        0.000e+00           3.771e-03
     (1, 896)                     0.000e+00           7.753e-03
     (32, 896)                    0.000e+00           7.809e-03
     (151936, 896)                0.000e+00           1.562e-02
     (4864, 896)                  0.000e+00           1.549e-02
   ```

   `roundtrip max = 0` everywhere; a bf16 truncation would have produced the
   `vs-bf16-trunc` column (up to 1.5e-2). The source path confirms this: for
   `FLOAT32`, `createOwnedHostTensor` builds a Float32 TTNN tensor
   (`runtime/lib/ttnn/runtime.cpp:79-80`; FLOAT32 supported at
   `runtime/lib/common/utils.cpp:174`); `enqueue_write_mesh_tensor` asserts equal
   dtypes with no implicit conversion
   (`.../ttnn/core/tensor/tensor_impl.cpp:645`); and tilize respects the optional
   `output_dtype` rather than forcing bf16
   (`.../ttnn/operations/core/to_layout/to_layout_op.cpp:163`).

2. **The embedding op casts its operands to bf16** — the gather error is the
   embedding workaround:
   - `.../ttnn/operations/embedding/device/embedding_device_operation.cpp:36` —
     `TT_FATAL(weights.dtype() == DataType::BFLOAT16, "Weights tensor must have BFLOAT16 dtype")`.
   - `tt-mlir lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp:149-167`
     (`createEmbeddingOpOperandsWorkarounds`) satisfies that by casting the weight
     operand (lines 157-158) and the output operand (lines 161-162) to
     `ttcore::DataType::BFloat16`. An fp32 embedding table is therefore cast to
     bf16 at the op boundary.

Corroborating: in the same probe run, SFPU elementwise ops on fp32 (silu, exp,
RoPE) were exact to fp32 epsilon (~5e-7), so device tiled storage and SFPU compute
keep fp32. Only ops with explicit bf16 operand requirements (embedding) or the
matmul FPU path lose precision.

## Fix

No upload/storage fix is needed; that path is correct. The bf16 cast is forced by
tt-metal's `TT_FATAL`, so recovering fp32 embeddings requires an fp32-capable
embedding kernel in tt-metal. For a pure table read the bf16 lookup is a one-shot
~bf16 rounding (not a compounding error), so this may be acceptable depending on
the model's tolerance.

## Minimal Reproducer

[supplemental/repro_embedding_bf16_cast.py](/home/houjun/lessons/2026-06-03-ttxla-fp32-embedding-bf16-cast/supplemental/repro_embedding_bf16_cast.py)
— standalone JAX, no model harness. It:

1. Roundtrips fp32 arrays through TT with no compute (control) at several sizes,
   printing roundtrip max diff alongside what a bf16 truncation would have cost.
2. Runs an fp32 embedding gather `table[ids]` (151936×896 table) CPU vs TT and
   compares TT output against the host source and against a bf16-rounded source.

Expected: roundtrip max = 0 everywhere (storage exact); gather max ~1.5e-2 on TT
and ~0 vs the bf16-rounded source (TT returned bf16-rounded values).

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
export TTXLA_LOGGER_LEVEL=ERROR TTMLIR_RUNTIME_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR
python /home/houjun/lessons/2026-06-03-ttxla-fp32-embedding-bf16-cast/supplemental/repro_embedding_bf16_cast.py
```

## Verification

```text
roundtrip max = 0.000e+00 for shapes (32,), (1,896), (32,896), (151936,896), (4864,896)
embedding gather (fp32 table, no math): TT vs host max ~= 1.5e-2; TT vs bf16(host) ~= 0
```

The host→device→host path is bit-exact for fp32 (storage does not truncate), and
the gather returns bf16-rounded values, matching the bf16-cast operand workaround.

## Notes

- The roundtrip uses `device_put` both directions, keeping the buffer row-major.
  The SFPU-exact silu/exp results (which go through TILE layout and on-device
  compute) independently show tiled fp32 storage is faithful, so the negative
  storage result is not an artifact of skipping tilize.
- Both source worktrees were dirty (unrelated in-progress edits); none touch the
  upload or embedding-workaround code cited here.
