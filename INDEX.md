# Lesson index

Lessons grouped by theme, with related/duplicate relationships marked. A lesson
can appear in more than one group. No two lessons are exact duplicates; the
closest relations are flagged below.

## On-device training bring-up (sequential gaps)

Three distinct backend gaps surface in order while bringing up on-device training;
each is exposed once the previous is cleared.

1. [ttxla-scatter-not-legalized](2026-06-03-ttxla-scatter-not-legalized/README.md)
   — non-axis-0 `stablehlo.scatter` (RoPE gather VJP) failed to legalize. **Fixed.**
2. [ttxla-shardy-reshard-collectives-assert](2026-06-03-ttxla-shardy-reshard-collectives-assert/README.md)
   — Shardy `reshard_to_collectives` asserts on a size-1 shard axis. Framework
   workaround landed; compiler unpatched.
3. [ttxla-reshape-tilepadded-dim-flatten](2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)
   — `ttnn.reshape` FATAL flattening a tile-padded `n_head` dim in the RoPE
   gather grad. **Fixed** (RowMajor reshape workaround).
4. [ttxla-trainer-clip-wte-global-id-misbind](2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/README.md)
   — a scalar `ttnn.reshape` is runtime-bound to the wte tensor (`global_id 1260`)
   in the `optax.clip_by_global_norm` graph. Open; whole-program (no standalone
   repro).

## Tile-padding in backward ops (same mechanism, different op)

A non-32-aligned dim padded to a 32×32 tile is mishandled in a backward op.
**Closely related — same root mechanism, distinct ops; not duplicates:**

- [ttxla-embedding-bw-tile-padding-grad](2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)
  — `ttnn.embedding_bw` leaks the cotangent from the padded region.
- [ttxla-reshape-tilepadded-dim-flatten](2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)
  — `ttnn.reshape` physical-volume mismatch on the padded dim.

## Embedding op (forward vs backward)

Same op, two distinct failures (not duplicates):

- [ttxla-fp32-embedding-bf16-cast](2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)
  — forward: fp32 weight cast to bf16 (op workaround).
- [ttxla-embedding-bw-tile-padding-grad](2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)
  — backward: tile-padding gradient leak.

## Numeric precision (bf16 / TF32 on the FPU and accumulators)

- [tt-matmul-fp32-accumulation-precision](2026-06-03-tt-matmul-fp32-accumulation-precision/README.md)
  — `packer_l1_acc=false` accumulation bug (fixed) + TF32 matmul-input ceiling.
- [ttxla-softmax-reduce-bf16-accumulation](2026-06-03-ttxla-softmax-reduce-bf16-accumulation/README.md)
  — softmax/reductions lowered with no compute config.
- [ttxla-fp32-embedding-bf16-cast](2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)
  — forward embedding bf16 cast (also listed above).

## Host upload / layout

- [ttxla-non-row-major-host-upload](2026-04-23-ttxla-non-row-major-host-upload/README.md)
  — non-row-major host buffers uploaded with the wrong layout (fixed). Its residual
  logit error is the matmul-precision lesson above.

## Suspected duplicates

None are exact duplicates. The closest pair is
**ttxla-embedding-bw-tile-padding-grad** and **ttxla-reshape-tilepadded-dim-flatten**
(same tile-padding root cause, different ops) — kept separate because the failing
op, reproducer, and fix differ.
