# TT-XLA Aborts in Shardy `reshard_to_collectives` (`isDone()`) on a Size-1 Shard Axis

## Summary

A GPT trains fine on Tenstorrent via a plain-`jit` path, but a trainer that
applies a **tensor-parallel sharding plan** aborts during compile with a hard C++
assertion — not a graceful legalization failure — deep in the Shardy dialect's
reshard→collectives export pass:

```text
python: .../shardy/dialect/sdy/transforms/export/reshard_to_collectives.cc:394:
  Value mlir::sdy::(anonymous namespace)::CollectiveInserter::insert():
  Assertion `isDone()' failed.
Signal: Aborted (6)
```

The trigger is the trainer's **tensor-parallel sharding plan**: weight
kernels are annotated `#sdy.sharding<@mesh, [{"shard"}, {}]>` (and the transpose
`[{}, {"shard"}]`) on a `<["batch"=1, "shard"=1]>` mesh. Even with `shard=1` (a
single device, no real parallelism) a matmul whose operands carry mismatched
shardings on the size-1 `"shard"` axis emits an `sdy.ReshardOp`; tt-xla lowers it
through Shardy's `reshard_to_collectives` pass, which asserts. The fix lives in
making that pass handle the degenerate (size-1 axis) reshard — or in not emitting
the reshard at all for single-device runs.

## Status

- Bug type: compiler crash (C++ `assert` abort) in an upstream Shardy export pass
  invoked by the tt-xla compile pipeline — not a tt-mlir conversion-legality gap.
- Component: Shardy `CollectiveInserter::insert`,
  `shardy/dialect/sdy/transforms/export/reshard_to_collectives.cc:394` (in
  `~/.cache/ttmlir-toolchain/src/shardy`). Reached via the tt-xla pass pipeline
  (`mlir::PassManager::run` → reshard_to_collectives).
- Fixed locally: **framework-side workaround landed** (the Shardy compiler itself
  is unpatched). The trainer gates its tensor-parallel sharding rules: when the
  shard mesh axis has size 1, every `"shard"` rule is rewritten to replicated, so
  no `sdy.ReshardOp` is emitted on a single device. All sharding-rule sites route
  through that one guard; it is inert when the shard axis is >1, and CPU-validated.
  This clears the Shardy crash; the trainer then hits a separate `ttnn.reshape`
  runtime gap (a distinct issue, not this bug). The standalone repro still aborts
  (it bypasses the framework guard) and documents the underlying compiler bug,
  which a defense-in-depth upstream fix would close (see Fix).
- Independent of the scatter fix
  ([2026-06-03-ttxla-scatter-not-legalized](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/README.md)):
  this aborts before any scatter is reached, and reproduces with no gather/scatter
  in the graph at all.
- Impact: on-device *training numerics* work (see the scatter lesson — RoPE scatter
  is bit-exact, a GPT trains via plain `jit`). What is blocked is any trainer that
  applies a tensor-parallel sharding plan unconditionally (including `n_shards=1`).
- Scope of trigger: a non-trivial reshard on a size-1 named mesh axis. A *trivial*
  single-axis reshard (`P("x",None) → P(None,"x")` on `<"x"=1>`) is optimized away
  and does **not** abort; the matmul-driven transpose reshard on a 2-axis
  `<"batch"=1,"shard"=1>` mesh does.

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty; carries
  the matmul-precision and scatter patches — neither is involved here).
- Shardy: bundled in the tt-mlir toolchain at
  `~/.cache/ttmlir-toolchain/src/shardy`; compiled into
  `third_party/tt-mlir/install/lib/libTTMLIRCompiler.so`.

## Host Environment

- Tenstorrent Blackhole `p150b` (single chip, `TT_VISIBLE_DEVICES=2`), Python 3.12,
  JAX/jaxlib 0.7.1, `ARCH_NAME=blackhole`.

## User-Visible Failure

A tensor-parallel-annotated training step on TT core-dumps during the first
compile:

```text
python: .../sdy/transforms/export/reshard_to_collectives.cc:394:
  ... CollectiveInserter::insert(): Assertion `isDone()' failed.
Signal: Aborted (6)
```

It aborts identically with or without `CONVERT_SHLO_TO_SHARDY=1` — the sharding
annotations route through Shardy regardless.

## Root Cause

A tensor-parallel sharding plan is attached to the model. The dumped SHLO
(`TTXLA_LOGGER_LEVEL=DEBUG`) shows a `<["batch"=1, "shard"=1]>` mesh with weight
kernels sharded on the `"shard"` axis:

```text
sdy.mesh @mesh = <["batch"=1, "shard"=1]>
... c_attn kernel : sdy.sharding<@mesh, [{}, {"shard"}]>
... c_proj kernel : sdy.sharding<@mesh, [{"shard"}, {}]>
... mlp  c_fc     : sdy.sharding<@mesh, [{}, {"shard"}]>
```

A matmul consuming a `[{}, {"shard"}]` operand and producing/consuming a
`[{"shard"}, {}]` value requires moving data across the `"shard"` axis — an
`sdy.ReshardOp`. Shardy's `reshard_to_collectives` export pass walks the reshard
factor-by-factor inserting collectives, and asserts `isDone()` (all factors
consumed) at the end. On the degenerate size-1 `"shard"` axis the inserter's
bookkeeping leaves the reshard "not done," tripping the assertion instead of
emitting a no-op. Because it is a release `assert`/abort, the whole process dies
with a core dump rather than a recoverable error.

The compiler itself is unpatched; a framework-side workaround is in place. Two
directions:

1. **Upstream (correct fix, not done):** make `CollectiveInserter` treat a reshard
   whose differing axes all have size 1 as a no-op (or complete its factor
   bookkeeping) so `isDone()` holds. This is in bundled Shardy
   (`reshard_to_collectives.cc`), so it needs a tt-mlir-toolchain rebuild.
2. **Framework-side (workaround, landed):** when there is a single shard/device
   (shard axis size 1), rewrite the `"shard"` sharding rules to replicated so no
   reshard is produced. This unblocks single-device TT training without touching
   the compiler, and is inert when the shard axis is >1.

The standalone numerics path (plain `jit`, no sharding plan) already trains on
device and is the current way to exercise TT training end-to-end.

## Minimal Reproducer

[supplemental/repro_sdy_reshard.py](/home/houjun/lessons/2026-06-03-ttxla-shardy-reshard-collectives-assert/supplemental/repro_sdy_reshard.py)
— standalone JAX, no model harness, no gather/scatter. It:

1. Builds a `<"batch"=1, "shard"=1>` mesh on one TT device.
2. `jit`s a `w1 @ w2` where `w1` is constrained to `[{}, {"shard"}]` and `w2`/the
   result to `[{"shard"}, {}]`, forcing a reshard on the size-1 `"shard"` axis.
3. Expected before fix: core dump with the `reshard_to_collectives.cc:394
   isDone()` assertion. Expected after fix: prints `RESHARD_OK`.

A control (a trivial `P("x",None)→P(None,"x")` reshard on a 1-axis `<"x"=1>` mesh)
prints `RESHARD_OK` — it is optimized away — which is why the bug needs the
matmul-driven transpose reshard above to surface.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=2 CONVERT_SHLO_TO_SHARDY=1 \
  python /home/houjun/lessons/2026-06-03-ttxla-shardy-reshard-collectives-assert/supplemental/repro_sdy_reshard.py
```

## Verification

```text
tt devices: [TTDevice(id=0, arch=Blackhole)]
python: .../reshard_to_collectives.cc:394: ... CollectiveInserter::insert():
  Assertion `isDone()' failed.
Aborted (core dumped)
```

The same assertion fires from both the standalone repro and the full
tensor-parallel-annotated trainer, confirming the reduction.

## Notes

- This is an *assertion abort*, so it appears even though `shard=1` makes the
  reshard semantically a no-op. A build with `NDEBUG` would likely skip the
  assert, but the underlying reshard handling would still be wrong. (The repro's
  `try/except` therefore cannot catch it on an assert build — the process aborts
  with SIGABRT; `RESHARD_FAILED` is only reached on a build that returns a
  recoverable error.)
- Single-device runs should not need Shardy collectives at all; the cheapest
  unblock is the framework-side workaround (skip sharding when `n_shards==1`).
- Same Shardy version ships for Wormhole and Blackhole; expect the same abort.
