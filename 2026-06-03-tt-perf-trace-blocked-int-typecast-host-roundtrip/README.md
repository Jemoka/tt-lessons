# TT Perf: Real GPT Training is Dispatch-Bound (~0.6% MFU) — Limitation Analysis + a Partial CE Fix

> **NOT a full win — read the corrected Status.** The real `theseus gpt/train/pretrain` step on
> Blackhole is **~0.6% MFU (dispatch-bound)**, not ~20%. An earlier "33.8% / GOAL ACHIEVED" claim
> was from a **standalone matmul microbench**, not the real trainer — corrected below. The
> integer-label CE fix is a real but partial + shape-fragile improvement (removes one of several
> stacked overheads; fails to compile at some shapes). `enable_trace` and fp8 are ruled-out dead
> ends for MFU.

## Summary

Investigation of on-device GPT training MFU on Blackhole. The matmul engine is healthy (110 TFLOP/s
≈ 63% of the bf16-HiFi4 peak; a *clean* standalone matmul-MLP step hits 35% MFU) — but the **real
GPT training step is ~0.6% MFU**: it is **dispatch-bound** (≈0.13 ms host dispatch × tens of
thousands of ops, multiplied by `slow_safe_linear` matmul chunking, plus mandatory per-step host
round-trips for the embedding index typecast and RNG `fold_in`). The matmul HW is fine but starved,
and the cure (on-device trace/graph replay) is blocked (won't compile these graphs, and was 4.4×
*slower* even where it did). This is a runtime-execution-model limitation, not fixable by a model
tweak.

A secondary finding: the big-vocab **one_hot cross-entropy** is one large memory-bound sink in the
step. Swapping it for integer-label `take_along_axis` CE (grad bit-exact vs CPU; qwen byte-identical)
required legalizing the gather-VJP's batched scatter (backend fix below) — a real improvement, but
**partial** (one of several overheads; doesn't lift the dispatch-bound trainer near 20%) and **not
shape-robust** (compiles on the trainer's synthetic config + a V=8192 gradcheck, but fails at
V=32000/seq=512). `enable_trace` (the initial hypothesis) and fp8 are documented below as ruled-out
dead ends for MFU.

## Status

- **⚠️ CORRECTED (2026-06-04): ~20% MFU is NOT achieved on the real trainer.** Honest measured state:
  - **The real `theseus gpt/train/pretrain` step is ~0.6% MFU** (measured: d=1536/L=12/seq=512/vocab=32000, slow-safe ON, one_hot CE → 9687 ms/step, 1.0 TFLOP/s). It trains correctly but is **dispatch-bound**, not compute-bound.
  - The **33.8% figure was a STANDALONE microbench** (clean matmul-MLP + vocab head, no slow-safe chunking, none of the GPT's per-op/host-roundtrip overheads) — NOT the real trainer. Earlier "GOAL ACHIEVED" claims based on it were an overstatement; corrected here.
  - **The fundamental limitation:** TT-XLA executes the step op-by-op from the host (~0.13 ms dispatch/op × tens of thousands of ops), multiplied by `slow_safe_linear` matmul chunking and the per-step host round-trips (si32→ui32 index, RNG `fold_in`). The matmul HW is fine (~63% of peak; clean step 35%) but starved. The cure (on-device trace/graph-replay) is **doubly blocked**: trace won't compile these graphs (host round-trips violate all-on-device) AND trace made a compute-bound step 4.4× *slower*. No model tweak fixes this — it needs a runtime-level fix (working trace or heavy op fusion).
- **CE lever — partial win, not a full fix:** the one_hot→integer-label CE swap removes one big memory-bound sink and its grad is **bit-exact vs CPU** (`max|tt-cpu|=6.119e-09`), and `qwen_parity` stays byte-identical. BUT (a) it's only one of several stacked overheads, so it does NOT lift the dispatch-bound real trainer near 20%; and (b) it is **not shape-robust** — it compiles+runs on the trainer's synthetic config and a V=8192 gradcheck, but **fails (XlaRuntimeError 13) at V=32000/seq=512**. So it's a real but incomplete/fragile improvement, not a closed win.
- **The CE backend mechanism** (`StableHLOToTTIRScatterOpConversionPattern`): route a batched scatter with `scatter_dims_to_operand_dims.size()==1` to the existing single-dim element-wise path (batching/row dims are position-aligned `update_scatter_dims`, handled by `extractElementWiseScatterIndices`' remap) — no flatten, CBs per-row. `checkBasicLegality` allows batching for single-scatter-dim and skips the `index_vector_dim==1` / `index.rank≤update.rank` checks when batched (non-batched RoPE path unchanged). The shape-robustness gap (V=32000 failure) is unresolved — likely a remaining L1/large-V or multi-batch-dim case.
- **Ruled-out leads (do not pursue for MFU):** `enable_trace` (4.4× slower; also blocked by the host round-trips); fp8/fidelity (only raises the matmul ceiling, which was never the bottleneck).

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
