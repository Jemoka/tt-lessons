# TT Trainer: wte Buffer Reaches a Scalar `ttnn.reshape` Slot (optax clip_by_global_norm graph)

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
  line.
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

Not implemented. Candidate directions:
- **Compiler (most likely):** `FlatbufferObjectCache` assigns `global_id` per MLIR
  Value (`obj.getAsOpaquePointer()`, monotonic `nextGlobalId()`) — so it is **not** a
  cache keying collision; `global_id 1260` is exactly one Value (the clip-div
  output). Therefore the scalar `ttnn.reshape`'s **serialized input resolves to that
  Value**. Op input serialization uses `getOperandThroughDPSOps(input)`
  (`TTNNToFlatbuffer.cpp`, e.g. FuncCall at :687 and per-op TensorRef lookups) — the
  prime suspect is a **DPS-resolution mis-mapping** in the clip-and-apply graph: a
  scalar reshape operand (typed scalar in IR) walks a destination-passing-style
  chain that lands on the clip-div `[100288,256]` output, so its recorded
  `global_id` becomes 1260. Audit `getOperandThroughDPSOps` against the clip
  `div`/`where`/broadcast ops.
  Mechanism detail (`getOperandThroughDPSOps`, `FuncOpToProgram.h`): it walks
  `while (isa<DestinationStyleOpInterface>(op)) value = dps.getDpsInitOperand(0)->get()`
  — i.e. follows the **DPS destination/init** operand up the chain, **asserting
  `getNumResults()==1` and `getNumDpsInits()==1`** (asserts are compiled out in the
  Release plugin). A DPS op in the clip graph that violates those assumptions (e.g.
  the `where`/select used to apply the clip, or a `div` whose init it shares),
  would silently walk `getDpsInitOperand(0)` to the wrong operand and land on the
  clip-div `[100288,256]` output. Reproduce/confirm by dumping the train_step TTNN
  IR and tracing the failing reshape's input back through its DPS-init chain.
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
