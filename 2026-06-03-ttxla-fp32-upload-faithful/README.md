# TT fp32 Host→Device Upload Is Faithful — the Embedding "bf16 in DRAM" Error Is an Op Workaround, Not a Storage Truncation

## Summary

While chasing why a pure fp32 embedding *gather* on Tenstorrent Blackhole shows
~1.5e-2 error (no arithmetic), the hypothesis was that the host→device upload
silently truncates fp32 to bf16 in DRAM. A pure roundtrip probe (`device_put`
fp32 → read back, no compute) disproves this: the roundtrip is **bit-exact
(0.0e+00) at every size tested, including the 151936×896 embedding-table shape**.
Storage and upload preserve fp32.

The gather error has a different cause: tt-metal's embedding kernel requires its
weight tensor to be BFLOAT16 (`TT_FATAL`), and tt-mlir satisfies this with an
op-specific workaround that casts the embedding weight and output to bf16 — the
same pattern as the SDPA bf16 cast. So the precision loss is an **op-level
operand workaround**, not a storage-path bug. This clears the "upload/storage"
lane: the dominant matmul bf16-input residual must come from the matmul
unpack/FPU path, not from how tensors are stored in DRAM.

## Status

- Finding type: negative result (rules out a hypothesis) + root-cause correction.
- Component: tt-runtime / tt-metal host→device upload (cleared); tt-mlir
  `TTNNWorkaroundsPass` embedding workaround (identified as the gather-error
  source).
- Storage truncation: **does not exist** on the upload path (verified on HW).
- Corrects an earlier claim: the embed-gather ~1.5e-2 was previously attributed
  to "stored operands bf16 in DRAM"
  (lesson [2026-06-03-ttxla-softmax-reduce-bf16-accumulation]); the real cause is
  the embedding op's bf16 weight workaround. The matmul bf16-input issue is a
  separate FPU-unpack matter (not storage), under investigation in the matmul
  unpacker lane.

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, commit
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4` (worktree dirty, unrelated).
- `tt-mlir` (submodule) —
  `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit
  `412daacc440f10bb98ccc685c311b01f1fadab70`.
- `tt-metal` (nested submodule) — under
  `tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`.
- `theseus` — `/home/houjun/theseus`, branch `feat/tenstorrent`,
  `f085ca67fa68ef08d63668cd7f866b2b8147839e` (the reproducer is theseus-free).

## Host Environment

`tt-qb-ac-02` (tt-qb2.stanford.edu), Linux 5.15.0-179-generic x86_64, 4×
Blackhole p150b. jax/jaxlib 0.7.1, ARCH_NAME=blackhole. Probe pinned to chip 1
(`TT_VISIBLE_DEVICES=1`). The startup `TT_FATAL ... remote mmio device` lines are
benign single-chip warnings; execution completes.

## User-Visible Failure

Earlier op probe (the symptom that started this lane): a pure embedding gather of
f32 data showed bf16-magnitude error despite doing no arithmetic.

```text
[take/embed (gather)] shape=(1, 32, 896)   max=1.553e-02 mean=1.127e-03
```

## Root Cause

Two independent checks show the upload path is fp32-faithful, and the gather
error comes from an op workaround:

1. HW roundtrip (no compute) is bit-exact at all sizes:

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
   `vs-bf16-trunc` column (up to 1.5e-2). Storage/upload preserves fp32.

2. Source path preserves fp32 (no downcast):
   - `runtime/lib/ttnn/runtime.cpp:79-80` — `createOwnedHostTensor` for
     `::ttnn::DataType::FLOAT32` builds a Float32 TTNN tensor
     (`createTTNNTensor<float>(...)`); FLOAT32 is a supported type
     (`runtime/lib/common/utils.cpp:174`), so it is not aliased away.
   - `third_party/tt-metal/.../ttnn/core/tensor/tensor_impl.cpp:645` —
     `enqueue_write_mesh_tensor` asserts
     `host_tensor.dtype() == device_tensor.dtype()` (no implicit conversion).
   - `third_party/tt-metal/.../ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:163`
     — tilize respects the optional `output_dtype`; it does not force bf16.

3. The gather error is the embedding op's own bf16 operand workaround:
   - `third_party/tt-metal/.../ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:36`
     — `TT_FATAL(weights.dtype() == DataType::BFLOAT16, "Weights tensor must have BFLOAT16 dtype")`.
   - `tt-mlir lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp:149-167`
     (`createEmbeddingOpOperandsWorkarounds`) satisfies that by casting the
     weight operand (line 157-158) and the output operand (line 161-162) to
     `ttcore::DataType::BFloat16`. So an fp32 embedding table is cast to bf16 at
     the op boundary — an op workaround, not a storage truncation.

Corroborating: in the same op-probe run, SFPU elementwise ops on f32 (silu, exp,
RoPE) were exact to f32 epsilon (~5e-7), so device tiled storage + SFPU compute
keep fp32. Only ops with explicit bf16 operand requirements (embedding) or the
matmul FPU path lose precision.

## Fix

No upload/storage fix is needed (the path is correct). Two op-level levers, for
the agents owning those lanes:

- Embedding: the bf16 cast is forced by tt-metal's `TT_FATAL`; recovering fp32
  embeddings needs an fp32-capable embedding kernel in tt-metal (or accepting the
  bf16 lookup, which for a pure table read is a one-shot ~bf16 rounding, not a
  compounding error).
- Matmul (the dominant Qwen residual): not addressed here; it is an FPU
  unpack/SrcReg precision matter, separate from storage — see the matmul unpacker
  investigation.

## Minimal Reproducer

`/home/houjun/lessons/2026-06-03-ttxla-fp32-upload-faithful/supplemental/repro_roundtrip_storage.py`
(theseus-free pure JAX). It `device_put`s fp32 arrays to TT and reads them back
with NO compute, at several sizes, and prints the roundtrip max diff alongside
what a bf16 truncation would have cost. Expected: roundtrip max = 0 everywhere.

## Reproduction Steps

```bash
cd /home/houjun/theseus
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
export TTXLA_LOGGER_LEVEL=ERROR TTMLIR_RUNTIME_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR
python /home/houjun/lessons/2026-06-03-ttxla-fp32-upload-faithful/supplemental/repro_roundtrip_storage.py
```

## Verification

```text
roundtrip max = 0.000e+00 for shapes (32,), (1,896), (32,896), (151936,896), (4864,896)
```

i.e. the host→device→host path is bit-exact for fp32; storage does not truncate.

## Notes

- The roundtrip uses `device_put` both directions, which keeps the buffer in
  row-major layout. The SFPU-exact silu/exp results (which DO go through TILE
  layout and on-device compute) independently show that tiled fp32 storage is
  also faithful, so the negative result is not an artifact of skipping tilize.
- This lane's purpose was to confirm or rule out an upload-time fp32→bf16
  truncation. It is ruled out. The remaining parity gap is the matmul FPU input
  precision, tracked separately.
- Both source worktrees were dirty (unrelated in-progress edits); none touch the
  upload or embedding-workaround code cited here.
