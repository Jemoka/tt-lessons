# TT-XLA Incorrectly Uploads Non-Row-Major Host Buffers

## Summary

TT-XLA produced wrong results when a host buffer passed into PJRT was not dense
row-major. The failure surfaced running Qwen2.5-0.5B on the TT PJRT backend:
several weight matrices imported from Hugging Face were Fortran-contiguous after
a transpose-heavy HF→JAX conversion, and uploading those buffers gave them the
wrong logical layout on device, corrupting large dense projections.

The fix detects non-dense-row-major host strides in
`BufferInstance::copyFromHost(...)`, materializes a dense row-major staging
buffer on the host, and uploads that compacted buffer instead.

## Status

- Bug type: backend correctness bug
- Component: `tt-xla` PJRT host-buffer upload path
- Fixed locally: yes
- Not resolved by this fix: a smaller residual logit error remains, traced
  separately to TT matmul precision (see
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md)).

## Repositories

- TT-XLA: `/home/houjun/tt-xla`
  - branch: `main`
  - commit: `03f29ed01a2bca27f5d8eaace659534016c7d0c4`
  - worktree: dirty
- HF snapshot used when this first surfaced:
  - model: `Qwen/Qwen2.5-0.5B`
  - snapshot: `060db6499f32faf8b98477b0a26969ef7d8b9987`

## Host Environment

- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-110-generic`
- Python: `3.12.13`
- JAX: `0.7.1`
- jaxlib: `0.7.1`
- Torch: `2.9.1+cpu`
- Transformers: `5.2.0`
- Device inventory: 4× Blackhole `p150b`

`tt-smi -ls` showed four available boards on host:

- chip 0: Blackhole `p150b`
- chip 1: Blackhole `p150b`
- chip 2: Blackhole `p150b`
- chip 3: Blackhole `p150b`

## User-Visible Failure

Before the fix, strict real-TT Qwen2.5-0.5B logit parity against Hugging Face was
badly wrong:

```text
max diff: 21.111913681030273
mean diff: 3.5360262393951416
top5 overlap: 0
hf loss: 7.848437309265137
jax loss: 18.352774227366726
```

The issue reduced to a smaller backend reproducer:

```text
exact_qwen_gate 2d max 7.550010681152344
weight_roundtrip max 0.500823974609375
w contiguous False f_contig True
weightT_roundtrip max 0.0
```

That pattern is the diagnosis:

- the same logical weights were correct when uploaded as a transposed row-major copy
- the incorrect path was tied to the original non-row-major host layout

## Root Cause

The PJRT host upload path in `BufferInstance::copyFromHost(...)` accepted
arbitrary host strides, but TT-XLA effectively assumed dense row-major semantics
before the data reached TTNN.

Two concrete problems were involved:

1. `calculateStrides(...)` treated the `num_byte_strides == 0` case incorrectly,
   returning all-ones strides instead of true dense row-major strides derived
   from shape.
2. Even when explicit non-row-major strides were present, the upload path did not
   first compact the host buffer into a dense row-major layout before
   constructing the runtime tensor.

This broke uploads for Fortran-contiguous arrays and other non-dense-row-major
views.

## Fix

The fix does three things:

1. compute true dense row-major strides from shape
2. detect when the incoming host strides are not dense row-major
3. materialize a dense row-major host buffer before creating the runtime tensor

Files changed:

- `pjrt_implementation/inc/api/buffer_instance.h`
- `pjrt_implementation/src/api/buffer_instance.cc`

Patch: [supplemental/ttxla_fix.patch](/home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/ttxla_fix.patch)

## Minimal Reproducer

[supplemental/repro_fortran_upload_matmul.py](/home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/repro_fortran_upload_matmul.py)
— standalone JAX, no model harness.

What it does:

1. builds a Fortran-contiguous weight tensor with a Qwen-like gate shape
2. uploads it to TT through `jax.device_put`
3. compares the TT roundtrip against the host source
4. runs a TT matmul on the same logical values

Expected after the fix: roundtrip diff stays at or near zero for the
F-contiguous tensor. Before the fix: the weight roundtrip is badly wrong.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed (so `jax.devices()` lists both
`tt` and `cpu`):

```bash
source .venv/bin/activate
python /home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/repro_fortran_upload_matmul.py
```

## Verification

After applying the TT-XLA fix and rebuilding the editable plugin, the isolated
gate probe improved from multi-unit error to normal low-error drift:

```text
before: plain same-input max diff 7.550010681152344
after:  plain same-input max diff 0.018653392791748047
```

End-to-end Qwen2.5-0.5B parity on TT also improved materially, but did not fully
converge:

```text
max diff: 0.9322552680969238
mean diff: 0.13998891413211823
top5 overlap: 3
hf loss: 7.848437309265137
jax loss: 7.902880500344669
```

That remaining gap is a separate bug — TT matmul precision, documented in
[2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md).

## Notes

- This lesson documents the host-upload/layout bug only.
- The reproducer also prints a matmul comparison, but the roundtrip check is the
  primary correctness signal for this bug.
- The TT-XLA worktree was dirty when this was diagnosed, so the supplemental
  patch contains only the minimal buffer-upload hunks relevant to this bug.
