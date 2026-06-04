# TT Perf: GPT Training Step is Dispatch-Bound (0.6% MFU); `enable_trace` Blocked by an si32→ui32 Index Typecast Host Round-Trip

## Summary

On-device GPT training on Blackhole runs at **~0.6% MFU** — the step is **~190× slower than the matmul roofline implies** (measured 6.12 s vs ~33 ms of matmul at the achievable 110 TFLOP/s). The matmul engine itself is healthy (110 TFLOP/s ≈ 63% of the bf16-HiFi4 peak of 175), so the gap is **per-op host-dispatch overhead**, not compute. The fix for dispatch-bound execution is TT-metal **trace** (`enable_trace`), which replays a captured op sequence with ~zero per-op host involvement.

`enable_trace` currently **fails to compile** a GPT step: `ttnn.capture_or_execute_trace: All output tensors of trace function must be on device`. Root cause: the si32→ui32 embedding/gather **index** typecast is lowered as a host round-trip (`ttnn.from_device → host typecast → ttnn.to_device`) because on-device dtype-cast requires a tilized layout and the index tensor is row-major. That mid-graph `from_device` violates trace's all-on-device requirement (and is also a per-step host sync feeding the dispatch overhead).

## Status

- **Bug type:** performance (dispatch-bound) + a lowering choice that blocks `enable_trace`.
- **Component:** tt-mlir `lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp` — device-input row-major integer typecast path emits a host round-trip.
- **Fixed:** NO (diagnosed + scoped). The fix is feasible but touches a central layout pass → needs careful qwen-parity regression testing; deferred to a focused change.
- **Impact:** blocks the single biggest MFU lever (trace) for the training step; also a per-step device↔host sync.
- **Not the blocker:** matmul throughput (healthy 63%), fp8/fidelity (only raise the matmul ceiling we aren't hitting). MFU is overhead-bound.

## Repositories

- `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir` — `TTMLIR_GIT_HASH=412daacc…`, dirty.
- 4× Blackhole p150b. jax/jaxlib 0.7.1. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`.

## User-Visible Failure

Compute-dense GPT step (d=1024, 12 layers), measured by the collective:

```text
achieved ≈ 0.7 TFLOP/s  → MFU 0.4% (vs bf16-HiFi4 peak 175) / 0.6% (vs achievable 110)
step 6.12 s vs ~33 ms at the matmul roofline → ~190× overhead
backward dominates ~82% of step time; optimizer cheap → dispatch-bound
```

`enable_trace=true` compile failure + the blocking IR:

```text
error: 'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device
%27 = "ttnn.from_device"(%16) : 64x1xsi32   device -> host
%28 = "ttnn.typecast"(%27) -> u32 : si32 -> ui32   (ON HOST)
%29 = "ttnn.to_device"(%28, %0) : ui32 back to device
```

## Root Cause

1. **Dispatch-bound execution.** Each ttnn op incurs a host dispatch (~0.13 ms floor measured on this backend). A 12-layer fwd+bwd+optimizer step is tens of thousands of ops → the device idles between tiny ops. `enable_trace` is the canonical fix (capture once, replay on-device).
2. **Trace is blocked by a host round-trip on the index typecast.** TTNN embedding/gather wants `uint32` indices; JAX provides `int32` (si32). The si32→ui32 cast on a **row-major** index tensor is lowered in `TTNNDecomposeLayouts` as `from_device → host typecast → to_device`, because **on-device typecast requires a tilized layout** (`TTNNDecomposeLayouts.cpp:739` "Device typecast only supports tilized tensors") and the index is row-major. Trace forbids any tensor leaving the device mid-graph → capture fails.

Note: both `Int32` and `UInt32` **are** tilize/untilize-able on device (`canTilizeDataTypeOnDevice`/`canUntilizeDataTypeOnDevice`, lines 151-168), so an on-device path is feasible — the pass simply doesn't choose it for the row-major device-input typecast case.

## Second trace blocker (confirmed): RNG `threefry_fold_in` host path

A uint32-index JAX workaround (feed ui32 token ids so no si32→ui32 cast is emitted) does **not** unblock trace — it fails in a *different* spot: `jax._src.prng threefry_fold_in` / `random_fold_in` (the model's PRNG key path) → `XlaRuntimeError 13` under `enable_trace`. So the step has **≥2 mid-graph host round-trips** trace can't capture: (1) the si32→ui32 index typecast (below), and (2) the RNG fold_in path. Both are backend/compiler fixes; trace-on MFU is not measurable from JAX until both keep their tensors on-device. (This means a model-side dtype tweak alone cannot unblock trace.)

## CORRECTION (decisive): `enable_trace` is a DEAD END; the real lever is the big-vocab CE

Later measurement overturned the trace hypothesis:
- A **clean matmul-MLP training step** (24 layers, no embedding/index/RNG/vocab) **already hits 35.3% MFU** (61.8 TFLOP/s vs 175) — so matmul+dispatch is NOT the bottleneck and the 20% target is met on the matmul core.
- **`enable_trace` made that step 4.4× SLOWER** (35%→8%) — trace hurts compute-bound steps. Do not pursue it for MFU.
- The real GPT trainer is **0.6% MFU** because of GPT-specific overheads stacked around the matmuls. Dominant: the **one_hot cross-entropy over the big vocab** (+80 ms alone: 35%→5.2%), a memory-bound `one_hot[B,T,V]`+grad. It exists only as a workaround for integer-label CE, whose `take_along_axis` gather-VJP emits a `stablehlo.scatter` that fails to legalize (the batched extension of [2026-06-03-ttxla-scatter-not-legalized](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/README.md)).

**The lever = legalize the CE gather-VJP scatter so integer-label CE works (drops the one_hot).** This is a backend scatter-legalization extension, qwen-safe (inference has no training scatter). The si32→ui32 typecast + RNG host round-trips (below) are secondary (sync removal), and the trace-unblock they'd enable is moot since trace hurts.

### The CE scatter is a BATCHED scatter; model-side flatten does NOT avoid it (HW-verified)
The integer-label CE gradient emits `stablehlo.scatter` with **batching dims** (`input_batching_dims=[0]`, `scatter_indices_batching_dims=[0]`, `index_vector_dim=2`, `inserted_window_dims=[1]`, `update_window_dims=[0]`), rejected at `StableHLOToTTIRPatterns.cpp:6093` (`checkBasicLegality`, first check). Reshaping `logits [B,T,V]→[B*T,V]` + `take_along_axis` (route 1) was tested on HW and **still emits a batched scatter** (batch over the BT rows) → same `failed to legalize 'stablehlo.scatter'`. So there is **no pure model-side escape**; the backend must handle the batched scatter.

### Fix (scoped): de-batch the scatter in the conversion pattern
In `StableHLOToTTIRScatterOpConversionPattern` (`StableHLOToTTIRPatterns.cpp:6086`): when batching dims are present, **rewrite them to explicit indices** — iota over each batch dim, concat into `scatter_indices` along `index_vector_dim`, move the batch dims from `input_batching_dims` into `scatter_dims_to_operand_dims`, and clear the batching dims. This reduces the batched CE scatter to the **multi-dimensional** scatter form the pattern already supports (:6118-6144, requires `index_vector_dim`=last and `scatter_dims ⊇ inserted_window`). Then `extractElementWiseScatterIndices`/matchAndRewrite build the flat index against the de-batched multi-dim shape. **qwen-safe** (no inference scatter); **training-correctness-critical** — must verify the loss gradient is bit-exact (existing `repro_scatter_legalize.py` pattern) + a training smoke converges, before landing. This is the precise, bounded next backend task and the highest-value MFU lever.

### Turnkey de-batch spec (exact, captured from the flattened CE VJP)

Flattened integer-label CE (`logits [BT,V]`, labels `[BT]`, `take_along_axis` then mean) — grad SHLO (captured CPU, `dump_ce_scatter.py`):
```
stablehlo.scatter(operand[BT,V], indices, updates) {
  inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2,
  input_batching_dims = [0], scatter_indices_batching_dims = [0] }   <-- batch over the BT rows
```
**De-batch (reduces exactly to the supported multi-dim path):**
1. `iota_b` = iota over the batch operand-dim (size BT), shaped to the indices' non-vector layout and placed at `index_vector_dim`.
2. `new_indices = concatenate([iota_b, scatter_indices], dim=index_vector_dim)` → index vector now `[batch_idx, col_idx]`.
3. New dim numbers: `scatter_dims_to_operand_dims = [0,1]`, `inserted_window_dims = [0,1]`, `index_vector_dim = last`, **batching dims cleared**.
4. Result satisfies the existing multi-dim checks (6118-6144: `index_vector_dim`=last ✓; `scatter_dims ⊇ inserted_window` {0,1}⊇{0,1} ✓) → `flattenMultiDimScatterIndices` + matchAndRewrite handle it unchanged.

Implementation locus: `checkBasicLegality` (`:6093`, allow batching dims when de-batchable) + a preprocessing step in `matchAndRewrite` that builds `new_indices` (iota+concat) and the remapped dim numbers before the existing index flattening. Generalize to N batch dims by iota+concat per batch dim. **Verify:** grad bit-exact vs CPU (`repro_scatter_legalize.py` pattern) + a training smoke converges + Qwen2.5-0.5B inference byte-identical, before landing.

### IMPLEMENTED + HW-VERIFIED: de-batch legalizes the CE scatter (qwen-safe); next layer is an L1 overflow

The de-batch normalization is implemented in `StableHLOToTTIRScatterOpConversionPattern`
(`maybeDeBatchScatter` + inline lowering; impl snapshot in
`supplemental/debatch_scatter_impl.cpp.txt`). Key mechanics learned:
- The new (de-batched) `stablehlo.scatter` is **not re-visited** by the dialect-conversion driver,
  so it must be lowered **inline in the same `matchAndRewrite`** (return the new op from the
  de-batch helper and continue with it; `checkBasicLegality` switched from `adaptor` to `op`).
  Relying on re-matching silently leaves the new scatter unlowered (`failed to legalize`).
- `ScatterDimensionNumbersAttr::get(ctx, updateWindowDims, insertedWindowDims, inputBatchingDims,
  scatterIndicesBatchingDims, scatterDimsToOperandDims, indexVectorDim)`; iota/concat via
  `rewriter.create`; region moved with `inlineRegionBefore`.

**Verified on Blackhole (chip 2):**
- The batched integer-label CE scatter **now legalizes** (SHLO→TTIR→TTNN succeeds; the prior
  `failed to legalize 'stablehlo.scatter'` is gone). The IR blocker — the lever everyone identified
  — is solved.
- **Qwen2.5-0.5B inference byte-identical** with the de-batch compiler (max diff 0.4292325973510742, top5 5) —
  the change only affects scatters with batching dims; inference has none. No regression.

**Next layer (downstream, separate):** the de-batched scatter then hits a tt-metal **L1 overflow at
runtime** — the multi-dim scatter lowering flattens the operand to 1D ([512,8192]→4.19M) and the
ttnn scatter's circular buffers `grow to 67285840 B > max L1 1572864 B`. So legalization is correct
but the existing flatten-to-1D multi-dim scatter lowering doesn't scale to large operands. Fixing
that (DRAM/streamed memory config, or a non-flattening scatter lowering) is the next bounded step to
realize the MFU win. Grad-bit-exactness gate `supplemental/ce_gradcheck2.py` is staged.

## Fix (proposed, not yet landed) — secondary host round-trips

Lower the row-major device-input integer typecast **on-device** instead of via host: `to_layout(tilize) → ttnn.typecast(si32→ui32) → to_layout(untilize)`, all in device memory. Since si32/ui32 are device-tilizable, this is valid and removes the `from_device`, unblocking trace and removing a per-step sync. Locus: the device-input typecast handler in `TTNNDecomposeLayouts.cpp` (the `!output.isTilized()` / row-major typecast branch that currently bounces through host).

Alternatives: (a) feed `uint32` token indices from the framework so no cast is needed (model-side, not always controllable); (b) make the embedding/gather lowering accept si32 indices directly.

**Caveat:** `TTNNDecomposeLayouts` is central to every model; any change must be verified to keep Qwen2.5-0.5B inference byte-identical (max diff 0.4292325973510742, top5 5) and all training gaps green. That regression discipline is why this is staged rather than rushed.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
# standalone MFU bench (minimal GPT, big-vocab one_hot CE):
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python /home/houjun/lessons/2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip/supplemental/mfu_bench.py
# matmul roofline anchor:
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python /home/houjun/lessons/2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip/supplemental/roofline_amortized.py
# dump the batched CE-VJP scatter dims (CPU only):
python /home/houjun/lessons/2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip/supplemental/dump_ce_scatter.py
# trace-blocker: compile a step with enable_trace and observe the from_device error
#   (set via compiler_options or the TTXLA_COMPILE_OPTIONS env bridge — see Notes)
```

## Verification

```text
matmul roofline (anchor): 110 TFLOP/s = 63% of bf16-HiFi4 peak (175)  [healthy]
training MFU:             0.4–0.6%  [dispatch-bound, ~190x overhead]
enable_trace:            FAILS to compile (from_device on si32 index)  [the blocker]
```

(The fix's effect — trace compiles, MFU jumps — is pending the staged change + qwen regression check.)

## Notes

- **MFU denominators (sourced):** bf16-HiFi4 = 175, bfp8-HiFi2 = 351, bfp4-LoFi = 702 TFLOP/s board peak (130 cores × {1.35, 2.7, 5.4}); ~83% achievable. The "664 BlockFP8" headline is the fp8/LoFi tier. 20% MFU vs bf16 peak = 35 TFLOP/s (reachable iff dispatch overhead is killed); vs 664 = 133 TFLOP/s (needs fp8 emission + matmul-bound).
- **Timing methodology:** host wall-clock can't measure TT compute naively (≈0.13 ms dispatch floor; full readback is transfer-bound). Use a jitted unrolled dependency-chain + scalar hard-sync (`supplemental/roofline_amortized.py`); for a compute-dense full step, the step ≫ dispatch floor so plain `block_until_ready` timing is valid.
- **Knob plumbing:** `enable_trace`, `experimental_weight_dtype` (bfp_bf8/bf4), `math_fidelity`, `ttnn_perf_metrics_enabled` are all JAX-settable PJRT compile options, but a trainer that calls bare `jax.jit` can't pass them. A tt-xla env-var bridge (`TTXLA_COMPILE_OPTIONS="enable_trace=true,…"`, `compile_options.cc`) was added so the trainer can set them with zero trainer edits.

- **Implementation note (landing strategy):** the existing scatter helpers read `srcOp` directly, so land the de-batch as a *pre-normalization* — emit a new `stablehlo.scatter` with cleared batching dims + iota-concat-augmented indices + merged dim numbers, replace `srcOp` with it, and let the existing multi-dim path lower the new op (don't thread new dims through the helpers). The full drop-in shape is: iota per batch dim, ordered concat, merged scatter/inserted dims, reduce=add, with a multi-update guard.
