# TT Trainer: wte Buffer Reaches a Scalar `ttnn.reshape` Slot (optax clip_by_global_norm graph)

> **RESOLVED — root cause was NOT a serializer gid mis-stamp.** This lesson's
> case-(b) "global_id mis-stamp" / pointer-collision theory was disproven by
> direct instrumentation (180 serialized reshapes, 0 with a mis-shaped input ref;
> the failing `ttnn::reshape` is invoked internally by a binary op's broadcast,
> not a serialized `ReshapeOp`). Actual root cause: `ttnn::embedding_bw` returns a
> **rank-4** tensor while the IR declares rank-2, so the downstream `global_norm`
> reduce (`dim_arg=[0,1]`) reduces only the leading unit dims and leaves the full
> `[100288,256]` wte gradient in a scalar slot — which then hits the broadcast
> reshape. Fixed in
> [2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/README.md).
> The analysis below is retained for the diagnostic trail; treat its conclusions
> as superseded.

## Summary

After the tile-padded reshape fix
([2026-06-03-ttxla-reshape-tilepadded-dim-flatten](/home/houjun/lessons/2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)),
on-device training of a GPT (`gpt/train/pretrain`) on Tenstorrent
advances further but still aborts at runtime in a `ttnn.reshape`:
`reshape_common.cpp:50: new_volume == old_volume … Invalid arguments to reshape`.
Runtime instrumentation shows the reshape reads a tensor of **logical volume
25,673,728 = the wte embedding table `[100288, 256]`** (vocab × n_embd) and tries
to reshape it to `target=[1]` — i.e. a **scalar-target reshape is fed the wte
tensor**.

The wte-shaped value is written (once, no overwrite — not a tensor-pool
collision-overwrite) at flatbuffer `global_id 1260` by the **`div` of
`optax.clip_by_global_norm`'s `clip_fn`** (the gradient-clipping transform in the
optimizer's AdamW chain, `optax.clip_by_global_norm(1.0)`, inside
`TrainState.apply_gradients`). A later scalar-target `ttnn.reshape` (IR-typed to
read a scalar) reads `global_id 1260` and gets the wte tensor → FATAL. This is a
**whole-program global-id / buffer-binding divergence specific to the optimizer's
clip-and-apply graph**: `optax.clip_by_global_norm` runs **correctly in isolation
on TT** (see Verification), so the bug only manifests inside the full train-step
module.

Not fixed. Precisely localized (producer op, global_id, and that it is neither a
reduce, donation, nor a func-call-input ordering bug). The fix is in the tt-mlir
flatbuffer serialization / global-id assignment for this graph shape, or the
runtime binding — a deep compiler/runtime investigation.

## Status

- Bug type: backend correctness — a scalar `ttnn.reshape` is bound at runtime to
  the wte tensor (`global_id 1260`) in the optimizer's clip graph; runtime FATAL.
- Component: tt-mlir TTNN flatbuffer serialization (global-id assignment) /
  tt-runtime tensor-pool binding; triggered by `optax.clip_by_global_norm` within
  the full `jit(train_step)` graph. Producer op: `clip_fn` `div`
  (`optax/_src/.../_clipping.py:105`, inside the optimizer's `apply_gradients`).
- Fixed locally: **No.** Localized; not yet root-caused to a specific id-assignment
  line. **A `getOperandThroughDPSOps` type-guard fix was TRIED and does NOT work**
  (see Fix): adding `if (next.getType() != value.getType()) break;` to stop the DPS
  walk at a shape-changing boundary was built, deployed (sha256-verified), and the
  `wte→(1,)` FATAL **persisted** on the full trainer (helper-verified, chip 1).
  Logic confirms why: with that guard `getOperandThroughDPSOps` always returns a
  value whose type == `op.getInput()` type, so a call-site net-type-check is a no-op
  too — **the misbind is type-INVISIBLE** (a buffer/global_id aliasing that does not
  manifest as a type change at the DPS-walk level). Qwen2.5-0.5B inference stayed
  byte-identical with the guard deployed (safe defensive no-op, not a regression — kept).
- Eliminated by direct experiment (NOT the cause):
  - **global_norm / large reduce-to-scalar** — gating the logging `grad_norm`
    (the trainer's logging grad-norm) did not change it, and a standalone-JAX `global_norm`/`sum`/`sqrt(sum)`
    on the exact `[100288,256]` shape **passes** on TT (all variants, rel ≤ 1e-3).
  - **input donation** — `donate_argnums=()` in `__make_train_step` left the FATAL
    unchanged.
  - **func-call input-arg ordering** — `global_id 1260` is **not** a func-call
    input slot (207 func-call binds logged; none at 1260).
  - **tensor-pool collision-overwrite** — `global_id 1260` is written exactly once
    (`overwrite=0`).
- Standalone-unreproducible: `optax.clip_by_global_norm(1.0).update(grads, state)`,
  inline clip math, and a TT-safe staged clip **all pass on TT** on a wte-shaped
  grad tree (see Verification). The bug needs the full train-step graph.
- Scope: training only (the optimizer update). Inference unaffected.
  This is the **fourth** gap in the on-device training bring-up sequence, after the
  scatter legalization fix
  ([2026-06-03-ttxla-scatter-not-legalized](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/README.md)),
  the Shardy guard
  ([2026-06-03-ttxla-shardy-reshard-collectives-assert](/home/houjun/lessons/2026-06-03-ttxla-shardy-reshard-collectives-assert/README.md)),
  and the tile-padded reshape fix
  ([2026-06-03-ttxla-reshape-tilepadded-dim-flatten](/home/houjun/lessons/2026-06-03-ttxla-reshape-tilepadded-dim-flatten/README.md)).

## Repositories

- TT-XLA / tt-mlir: `/home/houjun/tt-xla` (+ submodule `third_party/tt-mlir`),
  patched `libTTMLIRCompiler.so` (scatter + tile-padded-reshape fixes). The runtime
  instrumentation used to pin this (below) is reverted; deployed `libTTMLIRRuntime.so`
  is clean.
- Trainer: the optimizer is an AdamW chain with `optax.clip_by_global_norm(1.0)`;
  single-device trainer gates (sharding/scan) are required to reach this point.

## Host Environment

- Tenstorrent Blackhole `p150b`, single chip (`TT_VISIBLE_DEVICES=1`). Other chips
  were intermittently TLB-exhausted by repeated hard-killed runs (`tt_tlb_alloc -12`).
  Python 3.12, JAX/jaxlib 0.7.1, `ARCH_NAME=blackhole`.

## User-Visible Failure

```text
RESHAPE_DBG misbind: in_global_id=1260 out_global_id=1261 in_logical_volume=25673728 target=[1]
{TT_FATAL @ .../reshape_view/reshape_common.cpp:50: new_volume == old_volume
  Invalid arguments to reshape
jaxlib._jax.XlaRuntimeError: INTERNAL: Error code: 13
```

## Root Cause

Pinned by progressive runtime instrumentation (staged in supplemental):

1. **`reshape.cpp` (RESHAPE_DBG):** the failing `ttnn.reshape` reads
   `op->in()->global_id() == 1260`, whose runtime tensor has logical volume
   25,673,728 (= wte `[100288,256]`), with `target=[1]`.
2. **`types.cpp` (GID_DBG):** `global_id 1260` is written **once**
   (`overwrite=0`) with the wte-shaped tensor — not a pool collision-overwrite.
3. **`program_executor.cpp` (FUNCCALL_DBG):** `1260` is **not** among the 207
   func-call input-slot global_ids — so not a func-call-input ordering mismatch.
4. **`program_executor.cpp` (OPEXEC_DBG):** the op whose execution writes `1260`
   is `jit(train_step)/div_workaround` from
   `optax.clip_by_global_norm.<locals>.update_fn.<locals>.clip_fn`
   (`_clipping.py:105`), inside `TrainState.apply_gradients` (the optimizer update).

So the clip transform's per-leaf `div` (the clipped wte gradient, `[100288,256]`)
is stored at `global_id 1260`, and a *different* IR op — a scalar-target
`ttnn.reshape` (helper IR scan: every `[1]`-target reshape in the IR has a scalar
input) — reads `global_id 1260` at runtime. Two distinct MLIR values (the clip's
wte-grad output and a scalar reshape input) share `global_id 1260` in the
serialized program, and only the clip writes it → the reshape reads the wte tensor.
This is a whole-program global-id assignment / buffer-binding divergence; it does
not occur for `clip_by_global_norm` outside the full train-step graph.

## Fix

Not implemented. The misbind is **type-invisible** (a buffer/global_id aliasing that
does not show up as a shape/type change), so type-based serializer guards do NOT fix
it (see below). Candidate directions:
- **DPS type-guard — TRIED, DOES NOT WORK.** `FlatbufferObjectCache` assigns
  `global_id` per MLIR Value (`obj.getAsOpaquePointer()`, monotonic), so it is not a
  cache keying collision. The hypothesis was that op-input serialization
  (`getOperandThroughDPSOps(input)`, `TTNNToFlatbuffer.cpp`) walks the scalar
  reshape's input across a *shape-changing* DPS-init boundary onto the clip-div
  `[100288,256]` output. Fix tried: add `if (next.getType() != value.getType())
  break;` in `getOperandThroughDPSOps` (`FuncOpToProgram.h`). Built + deployed
  (sha256-verified) → **FATAL persisted** (helper-verified on the full trainer).
  Reason: with the guard, the walk only ever traverses *same-type* steps, so its
  result type always equals `op.getInput()`'s — i.e. the divergence is **not** a
  type change at any DPS step, and a call-site net-type-check is equally a no-op.
  Qwen2.5-0.5B inference byte-identical with the guard (safe; kept as defensive no-op).
- **Real direction (type-invisible aliasing):** the scalar reshape's recorded
  `TensorRef` ends up being the clip-div output's `global_id` without a type change
  — i.e. two Values genuinely share a buffer/dest at the *runtime* level
  (`getOrCreate`/`getOrCreateNoSharding` on `op.getResult()`, or DPS dest-buffer
  reuse where dest and result happen to be same-typed at the crossing). Pin it with
  the **defensive bind-time shape-assert** at `program_executor.cpp` (catch the
  mismatch loudly at the bind, naming the two global_ids), and/or trace the failing
  reshape's input `global_id` back to **every op that records that same global_id**
  in the serializer (not just the type-changing ones). The clip `div`/`where`/
  broadcast apply chain is still where to look.
  Mechanism detail (`getOperandThroughDPSOps`, `FuncOpToProgram.h`): it walks
  `while (isa<DestinationStyleOpInterface>(op)) value = dps.getDpsInitOperand(0)->get()`
  — i.e. follows the **DPS destination/init** operand up the chain, **asserting
  `getNumResults()==1` and `getNumDpsInits()==1`** (asserts are compiled out in the
  Release plugin). A DPS op in the clip graph that violates those assumptions (e.g.
  the `where`/select used to apply the clip, or a `div` whose init it shares),
  would silently walk `getDpsInitOperand(0)` to the wrong operand and land on the
  clip-div `[100288,256]` output. Reproduce/confirm by dumping the train_step TTNN
  IR and tracing the failing reshape's input back through its DPS-init chain.
- **CONVERGED root (helpers + CPU audit), and the recommended fix.** The reduce
  lowering is clean at every documented stage, so the `[1]`-reshape is a *later*
  rank-0/scalar materialization whose buffer/global_id binds to the reduce **input**
  in-context:
  - source SHLO: `[100288,256]` feeds `stablehlo.reduce`(add across [0,1]) → rank-0
    `f32` (global_norm sum-of-squares) — **no reshape** (verified on CPU).
  - StableHLO→TTIR: `StableHLOToTTIRReduceOpConversionPattern` →
    `ttir.sum keep_dim=false` (`StableHLOToTTIRPatterns.cpp:299-313`) — no reshape.
  - TTIR→TTNN: `ReductionOpConversionPattern` → `ttnn.sum` (`TTIRToTTNN.cpp:283-295`)
    — no reshape. → the scalar→`[1]` reshape is introduced by a rank-0/scalar→1D
    materialization step downstream, and the TTNN IR dump confirms **no
    `[100288,256]→[1]` reshape exists; all 71 `[1]`-target reshapes have scalar
    operands** — so at the IR level the reshape is correct; the **runtime gid 1260
    binds the big buffer to that scalar reshape's slot** (a cache/global_id
    mis-stamp specific to the in-context clip-apply graph; the plain reduce passes
    standalone).
  - **Recommended fix (consumer shape-guard, implementable):** in
    `createReshapeOp` (`TTNNToFlatbuffer.cpp:~2704`) — and the CpuOp/FuncCall input
    loops — when the cached input `TensorRef`'s shape ≠ the op's operand
    `RankedTensorType` shape, stamp a **fresh** correctly-shaped ref for that operand
    instead of the polluted cache entry. Converts the silent mis-read into a correct
    ref regardless of how the cache got polluted. (My per-step DPS type-guard is a
    no-op here because the resolved value is already same-typed; the guard must
    compare the *cached ref's* shape to the operand, not the walk's per-step types.)
  **Likely mechanism (why two different Values share gid 1260 type-invisibly):**
  the cache keys on `obj.getAsOpaquePointer()`, and MLIR **reuses freed Value/Op
  addresses**. If the clip-div `[100288,256]` Value is erased during a rewrite and
  its address is reused for the scalar reshape-input Value, `cache.at(scalar)`
  returns the **stale clip-div entry** (gid 1260) — a classic opaque-pointer-reuse
  cache bug, type-invisible, and in-context-only (the clip-apply graph has the
  erase/realloc pattern; standalone reduce/clip don't). This is consistent with all
  observations: IR operand is scalar, gid stamped to 1260, single write, no type
  change in the DPS walk. The consumer shape-guard above defends against it; a deeper
  fix would invalidate cache entries on Value erasure (or key on something stable
  across rewrites).
  **VALIDATION BLOCKED:** confirming the fix / grabbing the exact `RESHAPE_IRCHECK`
  line needs a hardware run, pending device availability.
- **Defensive runtime guard:** at `program_executor.cpp` bind/insert time, assert
  the bound tensor's logical shape matches the `TensorRef`'s expected shape — this
  converts the confusing downstream reshape FATAL into a clear "global_id N shape
  mismatch (expected …, got …)" at the point of divergence.
- **Workaround (unblock training now):** the trigger is the optimizer's
  `optax.clip_by_global_norm`. Removing it from the adamw chain (or a TT-safe clip
  formulation) should sidestep the colliding graph — to be confirmed (the
  confirmation run was blocked by intermittent device launch flakiness; see Notes).

## Minimal Reproducer

**No standalone reproducer** — the bug is whole-program (HW-verified). Three
standalone JAX attempts all run **clean on HW** (clip is not independently buggy):
[supplemental/repro_clip_by_global_norm.py](/home/houjun/lessons/2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/supplemental/repro_clip_by_global_norm.py)
— A: `optax.clip_by_global_norm(1.0).update`, B: inline clip math, C: TT-safe
staged clip; all PASS on TT on a wte-shaped grad tree.
[supplemental/repro_global_norm_bigtensor.py](/home/houjun/lessons/2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/supplemental/repro_global_norm_bigtensor.py)
— `global_norm`/`sum`/`sqrt(sum)` on `[100288,256]`; all PASS on TT.

The only confirmed reproducer is the full trainer (run from a venv that has the TT
PJRT plugin installed; the project trainer CLI, since the bug is whole-program):
```bash
source .venv/bin/activate
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=1 \
  <project-trainer> run gpt/train/pretrain <synthetic-pretrain-config> training.tokens=8000
```

## Verification

```text
# pure-JAX clip repro on TT — all pass (clip not independently buggy):
A optax.clip_by_global_norm.update   TT OK   rel=2.63e-03
B inline clip math                   TT OK   rel=2.63e-03
C TT-safe staged clip                TT OK   rel=1.89e-03
# pure-JAX global_norm on [100288,256] — all pass:
A optax.global_norm([x])  TT OK rel=5.0e-4   B sum(x**2) TT OK   C sqrt(sum) TT OK
# trainer (full graph) — FATAL:
RESHAPE_DBG misbind: in_global_id=1260 ... in_logical_volume=25673728 target=[1]
```

## Notes

- The runtime diagnostics that pinned this (`reshape.cpp`/`types.cpp`/
  `program_executor.cpp` `LOG_WARNING`s) are staged as
  [supplemental/runtime_instrumentation.patch](/home/houjun/lessons/2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/supplemental/runtime_instrumentation.patch);
  they rebuild `libTTMLIRRuntime.so` via a fast surgical relink (recompile the one
  .cpp → `ar r` its `libTTRuntime*.a` → relink, ~0.4s) — `OPEXEC_DBG` logs every op
  (verbose), so use it only to pin, then revert.
- Two earlier mis-reads were corrected by experiment: "it's global_norm" (gating
  the trainer's *logging* grad_norm did nothing; the real global_norm is
  *inside* clip in `apply_gradients`) and "it's func-call arg ordering" (1260 is not a
  func-call input). Both are recorded above so a future reader does not repeat them.
- The full-trainer confirmation of the clip-removal workaround was blocked by
  intermittent device launch flakiness (processes dying at startup with no log;
  unrelated to the bug).
