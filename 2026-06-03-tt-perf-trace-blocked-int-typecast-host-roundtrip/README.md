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
- tt-qb2, 4× Blackhole p150b. jax/jaxlib 0.7.1. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`.

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

## Fix (proposed, not yet landed)

Lower the row-major device-input integer typecast **on-device** instead of via host: `to_layout(tilize) → ttnn.typecast(si32→ui32) → to_layout(untilize)`, all in device memory. Since si32/ui32 are device-tilizable, this is valid and removes the `from_device`, unblocking trace and removing a per-step sync. Locus: the device-input typecast handler in `TTNNDecomposeLayouts.cpp` (the `!output.isTilized()` / row-major typecast branch that currently bounces through host).

Alternatives: (a) feed `uint32` token indices from the framework so no cast is needed (model-side, not always controllable); (b) make the embedding/gather lowering accept si32 indices directly.

**Caveat:** `TTNNDecomposeLayouts` is central to every model; any change must be verified to keep `qwen_parity.py` byte-identical (max diff 0.4292325973510742, top5 5) and all training gaps green. That regression discipline is why this is staged rather than rushed.

## Reproduction Steps

```bash
ssh houjun@tt-qb2.stanford.edu
cd /home/houjun/theseus && source .venv/bin/activate
# dispatch-bound MFU (compute-dense step):
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=2 python /home/houjun/.agents/mfu_bench.py
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
- **Knob plumbing:** `enable_trace`, `experimental_weight_dtype` (bfp_bf8/bf4), `math_fidelity`, `ttnn_perf_metrics_enabled` are all JAX-settable PJRT compile options, but theseus calls bare `jax.jit`. A tt-xla env-var bridge (`TTXLA_COMPILE_OPTIONS="enable_trace=true,…"`, `compile_options.cc`) was added so the real trainer can set them with zero theseus edits.
