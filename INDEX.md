# Lesson index

Lessons grouped by theme, with related/duplicate relationships marked. A lesson
can appear in more than one group. No two lessons are exact duplicates; the
closest relations are flagged below.

## On-device training bring-up (sequential gaps)

Distinct backend gaps surface in order while bringing up on-device training; each
is exposed once the previous is cleared. With the last fix the full
`gpt/train/pretrain` trainer **completes end-to-end on Blackhole**.

1. [ttxla-scatter-not-legalized](2026-06-03-ttxla-scatter-not-legalized/README.md)
   — non-axis-0 `stablehlo.scatter` (RoPE gather VJP) failed to legalize. **Fixed.**
2. [ttxla-shardy-reshard-collectives-assert](2026-06-03-ttxla-shardy-reshard-collectives-assert/README.md)
   — Shardy `reshard_to_collectives` asserts on a size-1 shard axis. Framework
   workaround landed; compiler unpatched.
3. [ttxla-reshape-tilepadded-dim-flatten](2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)
   — `ttnn.reshape` FATAL flattening a tile-padded `n_head` dim in the RoPE
   gather grad. **Fixed** (RowMajor reshape workaround).
4. [ttxla-embedding-bw-reduce-rank-mismatch](2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/README.md)
   — `ttnn.embedding_bw` returns rank-4 vs IR rank-2, so the `global_norm` reduce
   leaves the full wte gradient in a scalar slot → reshape FATAL. **Fixed.** This
   is the real root cause of the clip symptom below.
   - [ttxla-trainer-clip-wte-global-id-misbind](2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/README.md)
     — the same FATAL diagnosed (incorrectly) as a serializer global-id mis-stamp;
     **RESOLVED/superseded** by the rank-mismatch fix, retained as a diagnostic trail.
5. [ttxla-eager-dynamic-slice-device-only-path](2026-06-03-ttxla-eager-dynamic-slice-device-only-path/README.md)
   — eager `dynamic_slice` routes to tt-metal's device-only slice path and aborts
   (checkpoint save). **Fixed** — trainer then completes end-to-end.

## Tile-padding in backward ops (same mechanism, different op)

A non-32-aligned dim padded to a 32×32 tile is mishandled in a backward op.
**Closely related — same root mechanism, distinct ops; not duplicates:**

- [ttxla-embedding-bw-tile-padding-grad](2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)
  — `ttnn.embedding_bw` leaks the cotangent from the padded region.
- [ttxla-reshape-tilepadded-dim-flatten](2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)
  — `ttnn.reshape` physical-volume mismatch on the padded dim.

## Embedding op (forward vs backward)

Same op, three distinct failures (not duplicates):

- [ttxla-fp32-embedding-bf16-cast](2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)
  — forward: fp32 weight cast to bf16 (op workaround).
- [ttxla-embedding-bw-tile-padding-grad](2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)
  — backward (`embedding_bw`): tile-padding gradient leak.
- [ttxla-embedding-bw-reduce-rank-mismatch](2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/README.md)
  — backward (`embedding_bw`): rank-4 runtime output vs rank-2 IR breaks a downstream reduce.

## Numeric precision (bf16 / TF32 on the FPU and accumulators)

- [tt-matmul-fp32-accumulation-precision](2026-06-03-tt-matmul-fp32-accumulation-precision/README.md)
  — `packer_l1_acc=false` accumulation bug (fixed) + TF32 matmul-input ceiling.
- [ttxla-softmax-reduce-bf16-accumulation](2026-06-03-ttxla-softmax-reduce-bf16-accumulation/README.md)
  — softmax/reductions lowered with no compute config.
- [ttxla-fp32-embedding-bf16-cast](2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)
  — forward embedding bf16 cast (also listed above).

## Compiler dtype handling

- [ttxla-bf16-uniform-init-getvalues-float](2026-06-03-ttxla-bf16-uniform-init-getvalues-float/README.md)
  — compiler aborts lowering a bf16 `tenstorrent.uniform` because the lowering reads
  its bounds with `getValues<float>()` (asserts on non-f32). **Fixed** (APFloat read).
  Consolidates a since-removed duplicate (`ttxla-uniform-rand-bf16-getvalues-float`).

## Performance

- [tt-perf-trace-blocked-int-typecast-host-roundtrip](2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip/README.md)
  — the real `gpt/train/pretrain` step is **~0.6% MFU (dispatch-bound)**, not fixed by a
  model tweak (needs a runtime-level fix; `enable_trace`/fp8 are ruled-out dead ends).
  The big-vocab one_hot CE → integer-label CE swap (legalizing the **batched** gather-VJP
  scatter, the batched extension of
  [ttxla-scatter-not-legalized](2026-06-03-ttxla-scatter-not-legalized/README.md)) is a
  **partial + shape-fragile** improvement (grad bit-exact; fails to compile at V=32000/seq=512),
  not the 20% win an earlier microbench suggested.

## Host upload / layout

- [ttxla-non-row-major-host-upload](2026-04-23-ttxla-non-row-major-host-upload/README.md)
  — non-row-major host buffers uploaded with the wrong layout (fixed). Its residual
  logit error is the matmul-precision lesson above.

## Suspected duplicates

- **Resolved duplicate:** `ttxla-uniform-rand-bf16-getvalues-float` (earlier,
  fix-proposed-only) was folded into the FIXED
  [ttxla-bf16-uniform-init-getvalues-float](2026-06-03-ttxla-bf16-uniform-init-getvalues-float/README.md)
  — same bug, same fix — and removed (original kept in `.orig/`).

No remaining exact duplicates. The closest related pair is
**ttxla-embedding-bw-tile-padding-grad** and **ttxla-reshape-tilepadded-dim-flatten**
(same tile-padding root cause, different ops) — kept separate because the failing
op, reproducer, and fix differ.
