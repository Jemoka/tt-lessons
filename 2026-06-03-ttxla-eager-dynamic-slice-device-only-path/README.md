# TT-XLA Eager `dynamic_slice` Routes to tt-metal's Device-Only Slice Path and Aborts

## Summary

After the embedding-backward rank fix, the theseus `gpt/train/pretrain` trainer ran the on-device training step but aborted in **checkpoint save** — `int(self.key[0])`, indexing the on-device PRNG key — with `XlaRuntimeError: INTERNAL: Error code 13`. The underlying tt-metal abort was `TT_FATAL: slice_dim and num_devices must be provided for device-only tensor args slice`.

Root cause: JAX implements eager (non-jit) `slice_p` via `dynamic_slice_p` with the start index as a runtime scalar tensor; tt-mlir lowers `stablehlo.dynamic_slice` to `ttir.slice_dynamic`, whose runtime op calls the tt-metal **tensor-args** `ttnn::slice` overload. That overload takes tt-metal's "device-only tensor args" path (multi-device shard semantics) which requires `slice_dim`/`num_devices` and aborts when they are absent. The fix materializes the small begins/ends index tensors to host and calls the static-index `ttnn::slice` overload instead.

## Status

- **Bug type:** wrong runtime op-overload / dispatch path (hard abort).
- **Component:** tt-mlir TTNN runtime, `ttir.slice_dynamic` op (`runtime/lib/ttnn/operations/data_movement/slice.cpp`).
- **Fixed locally:** yes — runtime materializes dynamic-slice indices to host and uses the static-index `ttnn::slice` overload. Verified on hardware.
- **Milestone:** with this fix the full theseus `gpt/train/pretrain` trainer **completes end-to-end on Blackhole** ("Job 'gpt/train/pretrain' completed successfully"), after the prior six gaps (scatter, Shardy size-1, while/scan, loss-flatten, tile-padded reshape, embedding_bw rank) were cleared.
- **Inference unaffected:** `qwen_parity.py` emits no `slice_dynamic` (RoPE uses static slices), so inference output is unchanged.

## Repositories

- `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir` — `TTMLIR_GIT_HASH=412daacc440f10bb98ccc685c311b01f1fadab70`, worktree dirty (this fix + earlier landed fixes). The PJRT plugin dynamically links `install/lib/libTTMLIRRuntime.so`, so a runtime-only rebuild suffices.
- `/home/houjun/theseus` — training harness; `gpt/train/pretrain`, `configs/scratch/synthetic_pretrain.yaml`. Not modified for this fix.

## Host Environment

- `tt-qb2.stanford.edu`, 4× Blackhole p150b. clang-20 toolchain at `/opt/ttmlir-toolchain`.
- jax/jaxlib 0.7.1. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`.

## User-Visible Failure

```text
File ".../theseus/training/base.py", line 868, in train -> self.save(...)
File ".../theseus/job.py", line 232, in save_tree_and_metadata_from_path
    "jax_random": int(self.key[0]),
File ".../jax/_src/lax/slicing.py", line 1474, in _slice_impl
    return dispatch.apply_primitive(dynamic_slice_p, x, *start_indices, ...)
jaxlib._jax.XlaRuntimeError: INTERNAL: Error code: 13
```

Underlying tt-metal abort (captured via a flushed probe / repro):

```text
TT_FATAL: slice_dim and num_devices must be provided for device-only tensor args slice
  (ttnn/.../data_movement/slice/slice.cpp:316)
```

## Root Cause

The compiled program for the eager op is:

```text
func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [1] : (tensor<8xi32>, tensor<i32>) -> tensor<1xi32>
```

The start index `%arg1` is a **program input** (a runtime scalar tensor), not a constant — JAX's eager `_slice_impl` dispatches `slice_p` as `dynamic_slice_p` with the start passed as an argument. `StableHLOToTTIRDynamicSliceOpConversionPattern` (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:4471`) always lowers this to `ttir.slice_dynamic` with begins/ends as **tensors**. The runtime `runSliceDynamicOp` then calls the tt-metal tensor-args overload `ttnn::slice(in, beginsTensor, endsTensor, ...)`, which sets `use_device_only_path` and asserts `slice_dim && num_devices` — a multi-device shard-slice path that does not apply to a single logical slice.

Under jit, `x[0]` instead folds to a static `stablehlo.slice`/`ttir.slice_static` (constant begins/ends), which uses the normal path and works — which is why only eager indexing failed.

## Fix

`runtime/lib/ttnn/operations/data_movement/slice.cpp`, `runSliceDynamicOp`: materialize the (small) begins/ends index tensors to host int vectors (`ttnn::from_device` + `to_vector`, handling int32/uint32) and call the **static-index** overload `ttnn::slice(in, beginsSpan, endsSpan, stepSpan, memoryConfig)` — identical to `runSliceStaticOp`. This takes the normal slice path; correctness holds for arbitrary (data-dependent) indices at the cost of a tiny host sync of the index tensors. One file. Patch: `/home/houjun/lessons/2026-06-03-ttxla-eager-dynamic-slice-device-only-path/supplemental/slice_dynamic_host_index_fix.patch`.

## Minimal Reproducer

`/home/houjun/lessons/2026-06-03-ttxla-eager-dynamic-slice-device-only-path/supplemental/repro_eager_dynamic_slice.py` (pure JAX):

1. `a = jax.device_put(jnp.arange(8, int32))`.
2. `int(a[0])` — eager integer index (no jit).
3. **Before fix:** `TT_FATAL: slice_dim and num_devices ...` → `XlaRuntimeError Error 13`.
4. **After fix:** returns `0`. (jit'd `x[0]` worked both before and after — only eager failed.)

## Reproduction Steps

```bash
ssh houjun@tt-qb2.stanford.edu
cd /home/houjun/theseus && source .venv/bin/activate
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python /home/houjun/lessons/2026-06-03-ttxla-eager-dynamic-slice-device-only-path/supplemental/repro_eager_dynamic_slice.py
# full trainer:
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python -u -m theseus.cli run gpt/train/pretrain \
  configs/scratch/synthetic_pretrain.yaml ~/theseus training.tokens=8000
```

Runtime-only surgical rebuild: recompile `slice.cpp` with the command from `build/compile_commands.json`, `ar r` the object into `build/runtime/lib/ttnn/operations/libTTRuntimeTTNNOps.a`, relink `libTTMLIRRuntime.so`, copy to `install/lib/`.

## Verification

```text
BEFORE: [FAIL] eager int(a[0]): XlaRuntimeError INTERNAL Error 13 (TT_FATAL slice_dim/num_devices)
AFTER : [PASS] eager int(a[0]) = 0;  [PASS] key[0]/key[1:2]/big[0]/big[3:5] all pass
TRAINER: Job 'gpt/train/pretrain' completed successfully   (was: abort in checkpoint save)
```

`qwen_parity.py` regression check: inference emits no `slice_dynamic`, so output is unchanged (byte-identical baseline max diff 0.4292325973510742, top5 5). [Re-run pending an sshd window; the change touches only the `slice_dynamic` runtime path which inference does not exercise.]

## Notes

- The static `ttnn::slice` overload is the same one `runSliceStaticOp` already uses successfully, so this aligns the dynamic path with the known-good static path rather than introducing a new mechanism.
- A compiler-side fold of constant-start `dynamic_slice` → `slice_static` would NOT have fixed this case: the eager start index is a runtime program input, not a constant.
- The host materialization adds a small device→host sync per dynamic slice. For the trainer this is rare (checkpoint metadata, RNG key indexing); a hot-loop data-dependent dynamic slice would pay the sync each call. Acceptable for correctness; revisit if a perf-critical dynamic slice appears.
