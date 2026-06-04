# TT-XLA `ttnn.gather` Rejects int32 Index Tensors (Missing UInt32 Operand Workaround)

## Summary

Running the real GPT trainer (`gpt/train/pretrain`) on Blackhole crashed inside the
JIT'd `train_step` with an opaque `jax.errors.JaxRuntimeError: INTERNAL: Error code: 13`. With
`JAX_TRACEBACK_FILTERING=off` and the runtime logger on, the real cause surfaced as a tt-metal
device-op assertion: `Index tensor must be of type UINT32 or UINT16. Got: DataType::INT32`
(`gather_device_operation.cpp:59`). The trigger is the integer-label cross-entropy loss
(`jnp.take_along_axis(log_probs, targets[...,None])`), whose forward gather lowers to `ttnn.gather`
with a signed-`int32` index tensor — but tt-metal's gather device op only accepts `uint32`/`uint16`.

The fix is one missing operand-dtype workaround: `ttnn.gather` had no `getOperandsWorkarounds()`
override, so its `int32` index reached the device unchanged. Adding a `createGatherOpOperandsWorkarounds`
factory that forces the index operand `si32 → ui32` (and binding it on `TTNN_GatherOp`) makes the
TTNN-workarounds pass auto-insert the cast, exactly as embedding and `update_cache` already do.

## Status

- **Bug type:** missing backend op-legalization workaround (operand dtype), not a miscompile.
- **Component:** tt-mlir, `TTNNWorkarounds` pass + `TTNN_GatherOp` definition (TTNN dialect).
- **Fixed locally:** yes — `libTTMLIRCompiler.so` rebuilt and deployed to the install tree
  (`third_party/tt-mlir/install/lib/libTTMLIRCompiler.so`).
- **General:** fixes any JAX program whose `gather`/`take_along_axis` produces an int32 index, not
  just the trainer that surfaced it.
- **qwen-safe:** the workaround only rewrites a `ttnn.gather` operand dtype; Qwen2.5-0.5B inference
  does not change (it has no training-loss gather), so parity is unaffected.
- **Not resolved by this fix:** the separate big-vocab integer-label CE *backward* scatter shape work
  and the dispatch-bound real-trainer MFU
  ([2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip](/home/houjun/lessons/2026-06-03-tt-perf-trace-blocked-int-typecast-host-roundtrip/README.md))
  are orthogonal. This forward-gather bug surfaced after switching the loss to integer-label CE.

## Repositories

- **tt-mlir** (the fix): `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit
  `412daacc440f10bb98ccc685c311b01f1fadab70`, worktree dirty (carries this fix plus prior in-flight
  scatter/reshape/bf16 fixes).
- **tt-xla** (PJRT plugin host): `/home/houjun/tt-xla`, commit `03f29ed01a2bca27f5d8eaace659534016c7d0c4`.
- The bug surfaced from a model whose loss uses integer-label CE (`take_along_axis`), which is what
  emits the gather.

## Host Environment

- Ubuntu 24.04, Linux 6.8.0-110-generic, Python 3.12, Clang 20.
- jax==0.7.1, jaxlib==0.7.1; tt-xla PJRT plugin (editable install pointing at the source tree).
- Device: Tenstorrent Blackhole p150b (4× p150b).
- `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`, `CONVERT_SHLO_TO_SHARDY=1`.

## User-Visible Failure

The trainer died in `train_step` with only the opaque code:

```text
  File ".../training/base.py", line 750, in train
    self.state, loss, train_meta, grad_norm = train_step(...)
  ...
    results = self.xla_executable.execute_sharded(input_bufs)
jax.errors.JaxRuntimeError: INTERNAL: Error code: 13
```

With `JAX_TRACEBACK_FILTERING=off` + `TT_METAL_LOGGER_LEVEL=ERROR`, the real device-op assertion
appeared above the Python traceback:

```text
TT_FATAL: Index tensor must be of type UINT32 or UINT16. Got: DataType::INT32
{TT_FATAL @ .../ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp:59:
 tensor_args.input_index_tensor.dtype() == DataType::UINT32 ||
 tensor_args.input_index_tensor.dtype() == DataType::UINT16
```

The minimal reproducer (`take_along_axis` with an int32 index) fails the same way on the old compiler,
isolating the cause to the gather op's index dtype — independent of the trainer, the optimizer, and the
loss backward.

## Root Cause

tt-metal's `ttnn::gather` device op requires its index tensor to be `UINT32` or `UINT16`. In tt-mlir,
TTNN ops normalize device-required operand layouts/dtypes in the `TTNNWorkarounds` pass via each op's
`getOperandsWorkarounds()` (an `TTNN_WorkaroundInterface` method). `ttnn.embedding`,
`ttnn.embedding_backward`, and `ttnn.update_cache` all declare a workaround that forces their index
operand to `UInt32`. `ttnn.gather` declared **none**, so it inherited the interface's default (empty
workarounds) and its `int32` index passed straight through to the device op, tripping the assertion at
runtime.

The `stablehlo.gather` produced by `take_along_axis`'s indices is signed `int32` (JAX default integer
type for token ids), which lowers through `ttir.gather_dim` → `ttnn.gather` unchanged
(`GatherDimOpConversionPattern` forwards `adaptor.getIndex()` with no cast). Nothing downstream
corrected the dtype because the workaround was simply missing.

**There are two sub-problems, and fixing only the first is not enough.** Declaring the workaround on
the op (sub-problem 1) does nothing on its own, because the `TTNNWorkarounds` pass only *applies*
operand workarounds to an op whose name is in an `enabledOps` allow-set. That set is built two ways
(`TTNNWorkaroundsPatterns.cpp`, `TTNNWorkarounds::runOnOperation`): when the optimizer is **off** it is
*all* TTNN dialect ops, but when the optimizer is **on** it is a small curated literal
`enabledOpsForWorkaroundWithOptimizer = {WhereOp, FullOp, EmbeddingOp, ScatterOp}`. tt-xla compiles
with the optimizer **on**, so `ttnn.gather` was filtered out by the allow-set and its
`getOperandsWorkarounds()` was never even consulted. The first deploy (workaround factory + binding
only) therefore failed on-device with the identical `Got: DataType::INT32` assertion — the symptom is
indistinguishable, so verify on hardware, not just that the symbol links. (Embedding/Scatter never hit
this because they are already in the curated set.)

## Fix

Mirror the embedding / `update_cache` index workaround for gather:

1. `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` — add
   `TTNNOperandsWorkaroundsFactory::createGatherOpOperandsWorkarounds(RankedTensorType indexType)`:
   leaves operand 0 (input) and the output unconstrained, and sets operand 1 (index) to
   `DataType::UInt32` when it is `Int32`.
2. `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h` — declare the factory method.
3. `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` — give `TTNN_GatherOp` an `extraClassDeclaration`
   `getOperandsWorkarounds()` that calls the new factory with `getIndex().getType()`.
4. `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` — add
   `ttnn::GatherOp::getOperationName()` to the `enabledOpsForWorkaroundWithOptimizer` literal set, so
   the workaround actually runs under the optimizer (the step the first deploy was missing).

The `TTNNWorkarounds` pass then inserts the `si32 → ui32` typecast on the index automatically. Token
ids are non-negative, so the reinterpret is value-safe (no signed values to lose).

Patch: `/home/houjun/lessons/2026-06-04-ttxla-ttnn-gather-int32-index-uint32-workaround/supplemental/ttnn_gather_uint32_workaround.fulldiff.patch`
(the relevant hunks are the three `GatherOp` / `createGatherOpOperandsWorkarounds` additions; the
reshape-workaround hunks in the same files are unrelated prior in-flight work).

## Minimal Reproducer

`/home/houjun/lessons/2026-06-04-ttxla-ttnn-gather-int32-index-uint32-workaround/supplemental/repro_take_along_axis_int32_gather.py`

1. Builds float32 logits `[2,128,100288]` and int32 targets `[2,128]` (token-id-like).
2. Computes `-log_softmax(...)` gathered at the targets via `jnp.take_along_axis` — emits the
   `ttnn.gather` with an int32 index.
3. Runs it under `jax.jit` on the `tt` device and on `cpu`, and compares.

- **Before fix:** the `tt` run aborts with `Index tensor must be of type UINT32 or UINT16. Got:
  DataType::INT32`.
- **After fix:** the gather runs; `max|tt - cpu|` is at the bf16/tf32 HW-precision floor
  (`< 1e-3`), prints `GATHER_REPRO_PASS`.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole CONVERT_SHLO_TO_SHARDY=1 TT_VISIBLE_DEVICES=<free-chip> \
TTXLA_LOGGER_LEVEL=ERROR TTMLIR_RUNTIME_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR \
  python -u /home/houjun/lessons/2026-06-04-ttxla-ttnn-gather-int32-index-uint32-workaround/supplemental/repro_take_along_axis_int32_gather.py
```

A full `gpt/train/pretrain` step reproduces it too (integer-label CE loss): on the old compiler it
crashes in `train_step` with Error code 13; on the fixed compiler the step runs.

### Surgical rebuild (no full superbuild)

The build tree's `rules.ninja` is absent, so `ninja` cannot drive the build. Manual lane:

```bash
cd /home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir
# 1. Regenerate the op header (extraClassDeclaration lands in TTNNOps.h.inc):
mlir-tblgen -gen-op-decls -I include -I build/include \
  -I ~/.cache/ttmlir-toolchain/include \
  -I ~/.cache/ttmlir-toolchain/src/stablehlo -I ~/.cache/ttmlir-toolchain/src/stablehlo-build \
  include/ttmlir/Dialect/TTNN/IR/TTNNOps.td -o build/include/ttmlir/Dialect/TTNN/IR/TTNNOps.h.inc
# 2. Recompile the two affected objects (exact commands from build/compile_commands.json):
#    lib/Dialect/TTNN/IR/{TTNNOps.cpp, TTNNWorkaroundsPass.cpp}
# 3. ar r into the dialect archive the compiler lib links:
llvm-ar r build/lib/libMLIRTTNNDialect.a \
  build/lib/Dialect/TTNN/IR/CMakeFiles/obj.MLIRTTNNDialect.dir/{TTNNOps,TTNNWorkaroundsPass}.cpp.o
# 4. Relink libTTMLIRCompiler.so (raw clang++ link line, NOT a ninja target).
# 5. Deploy: cp build/lib/libTTMLIRCompiler.so ../../install/lib/  (the plugin loads from install/).
```

## Verification

Done in two on-device rounds, which is itself the key lesson:

- **Round 1 (workaround factory + binding only):** deployed `libTTMLIRCompiler.so` with the new symbol
  confirmed present (`nm -C | grep createGatherOpOperandsWorkarounds`) and the plugin confirmed
  resolving to it (`ldd`). The repro **still aborted with the identical** `Got: DataType::INT32`. This
  proved the binding alone is inert — the op was being filtered out by the optimizer `enabledOps`
  allow-set before its workaround was consulted.
- **Round 2 (after adding `GatherOp` to `enabledOpsForWorkaroundWithOptimizer`):** rebuilt + redeployed.

Expected Round-2 result:

```text
GATHER_REPRO shape=(2, 128) max|tt-cpu|=<~1e-4..1e-3>
GATHER_REPRO_PASS
```

_(Round-2 on-device confirmation was in flight when qb2 sshd dropped; update this block with the
measured `max|tt-cpu|` once the run is read back. The first round is fully verified as FAILING, which
is what pinned the allow-set sub-problem.)_

Takeaway: a missing-workaround fix is not verified by the symbol linking or the IR header
regenerating — only by the op actually running on hardware, because the allow-set filter makes a
correctly-declared-but-unlisted workaround fail with a symptom identical to no fix at all.

## Notes

- Why it presented as a scatter bug at first: the integer-label CE also has a *backward* path whose
  gather-VJP is a `stablehlo.scatter` (a separate, already-documented shape-fragility). This bug is in
  the **forward** gather and is independent — the device asserts at gather construction, before any
  backward op runs.
- Scope: the workaround only fires for `Int32` indices. `UInt16` indices (already accepted by the
  device) are untouched; non-integer index types would be a different (verifier) error.
- The `fulldiff.patch` was taken from a dirty tree; only the `GatherOp` /
  `createGatherOpOperandsWorkarounds` hunks belong to this lesson.
