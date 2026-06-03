# TT-XLA Incorrectly Uploads Non-Row-Major Host Buffers

## Summary

TT-XLA produced incorrect results when a host buffer passed into PJRT was not dense row-major. The concrete failure that exposed this was Qwen 0.5B parity in Theseus on Blackhole: HF weights were converted into JAX params correctly, but several imported weight matrices were Fortran-contiguous because of transpose-heavy HF->JAX mapping. Those buffers were uploaded to TT with the wrong logical layout, which corrupted large dense projections.

The fix was to detect non-dense-row-major host strides in `BufferInstance::copyFromHost(...)`, materialize a dense row-major staging buffer on the host, and then upload that compacted buffer instead.

## Status

- Bug type: backend correctness bug
- Component: `tt-xla` PJRT host-buffer upload path
- Fixed locally: yes
- Remaining Theseus Qwen parity issue after this fix: yes, smaller but still present

## Repositories

- Theseus repo: `/home/houjun/theseus`
  - branch: `feat/tenstorrent`
  - commit: `f085ca67fa68ef08d63668cd7f866b2b8147839e`
  - worktree: dirty
- TT-XLA repo: `/home/houjun/tt-xla`
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
- Device inventory: 4x Blackhole `p150b`

`tt-smi -ls` showed four available boards on host:

- chip 0: Blackhole `p150b`
- chip 1: Blackhole `p150b`
- chip 2: Blackhole `p150b`
- chip 3: Blackhole `p150b`

## User-Visible Failure

Before the fix, strict real-TT Theseus Qwen 0.5B parity was badly wrong:

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

That pattern matters:

- the same logical weights were correct if uploaded as a transposed row-major copy
- the incorrect path was tied to the original non-row-major host layout

## Root Cause

The PJRT host upload path in `BufferInstance::copyFromHost(...)` accepted arbitrary host strides, but TT-XLA was effectively relying on dense row-major semantics before the data reached TTNN.

Two concrete problems were involved:

1. `calculateStrides(...)` treated the `num_byte_strides == 0` case incorrectly and returned all-ones strides instead of true dense row-major strides derived from shape.
2. Even when explicit non-row-major strides were present, the upload path did not first compact the host buffer into a dense row-major layout before constructing the runtime tensor.

This broke uploads for Fortran-contiguous arrays and other non-dense-row-major views.

## Fix

The local fix does three things:

1. compute true dense row-major strides from shape
2. detect when the incoming host strides are not dense row-major
3. materialize a dense row-major host buffer before creating the runtime tensor

Files changed in the fix:

- `pjrt_implementation/inc/api/buffer_instance.h`
- `pjrt_implementation/src/api/buffer_instance.cc`

The patch is attached here:

- [supplemental/ttxla_fix.patch](/home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/ttxla_fix.patch)

## Minimal Reproducer

Use the attached script:

- [supplemental/repro_fortran_upload_matmul.py](/home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/repro_fortran_upload_matmul.py)

What it does:

1. creates a Fortran-contiguous weight tensor with a Qwen-like gate shape
2. uploads it to TT through JAX
3. compares TT roundtrip against the host source
4. optionally runs a TT matmul using the same logical values

Expected behavior after the fix:

- roundtrip diff stays at or near zero for the F-contiguous tensor

Expected behavior before the fix:

- weight roundtrip is badly wrong for the F-contiguous tensor

## Reproduction Steps

From `/home/houjun/theseus`:

```bash
source .venv/bin/activate
python /home/houjun/lessons/2026-04-23-ttxla-non-row-major-host-upload/supplemental/repro_fortran_upload_matmul.py
```

If you want the original end-to-end symptom:

```bash
source .venv/bin/activate
CONVERT_SHLO_TO_SHARDY=1 uv run --group ttxla python scripts/qwen_parity.py \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Hello world" \
  --max-length 32
```

## Verification

After applying the TT-XLA fix and rebuilding the editable plugin, the isolated gate probe improved from multi-unit error to normal low-error drift:

```text
before: plain same-input max diff 7.550010681152344
after:  plain same-input max diff 0.018653392791748047
```

End-to-end Theseus Qwen parity also improved materially, but did not fully converge:

```text
max diff: 0.9322552680969238
mean diff: 0.13998891413211823
top5 overlap: 3
hf loss: 7.848437309265137
jax loss: 7.902880500344669
```

That remaining gap is a separate bug.

## Notes

- This lesson documents the confirmed host-upload/layout bug only.
- The supplemental script prints a matmul comparison too, but the roundtrip check is the primary correctness signal for this bug. Separate TT dense-op numeric drift still exists elsewhere.
- The TT-XLA and Theseus worktrees were dirty when this was diagnosed, so the supplemental patch intentionally contains only the minimal buffer upload fix relevant to this bug.
