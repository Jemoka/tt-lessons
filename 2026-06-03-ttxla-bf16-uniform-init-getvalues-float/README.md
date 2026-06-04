# TT-XLA Compiler Aborts on bfloat16 Params: `getValues<float>` on a bf16 uniform-init Constant

## Summary

Training a GPT with **fully bf16 parameters** (`dtype.param=bfloat16`) core-dumped in the *compiler* (not the runtime) with an MLIR assertion: `DenseElementsAttr::getValues() [T = float]: Assertion 'succeeded(range) && "element type cannot be ..."'`. The mixed-precision config (fp32 params + bf16 activations) — the normal way to train bf16 — compiled and trained fine; only bf16 *params* triggered it.

Root cause: `TenstorrentUniformToRandConversionPattern` (the `tenstorrent.uniform` → `ttir.rand` lowering) read the uniform initializer's `low`/`high` constant operands with `getValues<float>()`, which asserts when the dense attr's element type is not f32. A bf16-param model's weight initializer emits bf16 `low`/`high` constants. The fix reads them as `APFloat` and converts to f32, robust to any float element type.

## Status

- **Bug type:** compiler crash (MLIR assertion / abort) on a valid input dtype.
- **Component:** tt-mlir, `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` (`TenstorrentUniformToRandConversionPattern`).
- **Fixed locally:** yes — `getValues<float>()` → `getValues<APFloat>()` + `APFloat::convert(IEEEsingle)`. Verified on hardware: fully-bf16 GPT now compiles and runs (no abort).
- **Inference unaffected / no regression:** `qwen_parity.py` byte-identical after the change (max diff 0.4292325973510742, top5 5).
- **Separate remaining issue (NOT this bug):** with bf16 *params*, on-device training loss is `-inf` from step 0 — a **TT-specific numerical overflow** in the fully-bf16 forward pass (CPU fully-bf16 trains fine: 11.49→6.78). This is a distinct, deeper backend numerics bug, documented in Notes; this lesson only covers the compiler abort. Standard mixed precision (fp32 params + bf16 activations) is unaffected and is the recommended config anyway.

## Repositories

- `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir` — `TTMLIR_GIT_HASH=412daacc440f10bb98ccc685c311b01f1fadab70`, worktree dirty. Compiler-only change; `libTTMLIRCompiler.so` rebuilt and deployed to `install/lib`.

## Host Environment

- `tt-qb2.stanford.edu`, 4× Blackhole p150b. clang-20 toolchain. jax/jaxlib 0.7.1, flax, optax. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`.

## User-Visible Failure

```text
DTYPE param=bfloat16 act=bfloat16
... (device init) ...
python: /home/houjun/.cache/ttmlir-toolchain/include/mlir/IR/BuiltinAttributes.h:423:
  auto mlir::DenseElementsAttr::getValues() const [T = float]:
  Assertion `succeeded(range) && "element type cannot be ..."'
Process received signal: Aborted (6)   (core dumped)
```

Mixed precision (`param=float32 act=bfloat16`) compiled and trained fine (loss 11.56→6.98), so only bf16 *params* triggered the abort.

## Root Cause

`TenstorrentUniformToRandConversionPattern::matchAndRewrite` (lowering the `tenstorrent.uniform` composite to `ttir.rand`) extracts the uniform bounds from its `low`/`high` constant operands:

```cpp
auto lowAttr  = rewriter.getF32FloatAttr(lowValue.getValues<float>()[0]);
auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<float>()[0]);
```

`DenseElementsAttr::getValues<float>()` requires the attr's element type to be f32 and asserts otherwise. When the model's params are bf16, flax's weight initializers run in bf16, so the `tenstorrent.uniform` bounds are bf16 dense constants → the assert fires and the process aborts. This is why fp32-param models (and the matmul/bias/grad bf16 *activation* repros) never hit it: their uniform-init constants are f32.

## Fix

Read the bounds as `APFloat` (the storage type of any `DenseFPElementsAttr`) and convert to f32 via `APFloat::convert(IEEEsingle)`, which handles bf16 (and any other float type) without asserting:

```cpp
auto toF32 = [](llvm::APFloat v) -> float {
  bool losesInfo = false;
  v.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return v.convertToFloat();
};
auto lowAttr  = rewriter.getF32FloatAttr(toF32(*lowValue.getValues<llvm::APFloat>().begin()));
auto highAttr = rewriter.getF32FloatAttr(toF32(*highValue.getValues<llvm::APFloat>().begin()));
```

One file. Patch: `/home/houjun/lessons/2026-06-03-ttxla-bf16-uniform-init-getvalues-float/supplemental/bf16_uniform_getvalues_fix.patch`.

## Minimal Reproducer

`/home/houjun/lessons/2026-06-03-ttxla-bf16-uniform-init-getvalues-float/supplemental/smoke_bf16.py` — real GPT, dtypes from env `SMOKE_PARAM`/`SMOKE_ACT`:

1. `SMOKE_PARAM=bfloat16 SMOKE_ACT=bfloat16` → **before fix:** compiler abort; **after fix:** compiles + runs.
2. `SMOKE_PARAM=float32 SMOKE_ACT=bfloat16` (mixed) → trains fine before and after.

`supplemental/repro_bf16_compile.py` is the bisection harness (bf16 matmul / const weight / bias / rmsnorm / grad / embedding) — all of which PASS, confirming the trigger is specifically the bf16 *uniform initializer* constant, not bf16 compute in general.

## Reproduction Steps

```bash
ssh houjun@tt-qb2.stanford.edu
cd /home/houjun/theseus && source .venv/bin/activate
SMOKE_PARAM=bfloat16 SMOKE_ACT=bfloat16 ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python /home/houjun/lessons/2026-06-03-ttxla-bf16-uniform-init-getvalues-float/supplemental/smoke_bf16.py
```

Compiler-only surgical rebuild: recompile `StableHLOLegalizeCompositePass.cpp` with the command from `build/compile_commands.json`, `ar r` into `build/lib/libTTMLIRStableHLOToTTIR.a`, relink `libTTMLIRCompiler.so`, copy to `install/lib/`.

## Verification

```text
BEFORE: param=bf16 act=bf16 -> compiler Aborted (getValues<float> assertion)
AFTER : param=bf16 act=bf16 -> compiles + runs (loss is -inf, see Notes — a separate bug)
        param=fp32 act=bf16  -> trains: loss 11.56 -> 6.98 (unchanged)
qwen_parity (default): max diff 0.4292325973510742, top5 5  (byte-identical; no regression)
```

## Notes

- **The `-inf` is a separate, still-open bug.** After this compiler fix, a fully-bf16 model compiles but its on-device training loss is `-inf` from step 0 — an overflow in the bf16 forward pass that is **TT-specific** (the identical fully-bf16 model trains fine on CPU: 11.49→6.78). Suspected: a bf16-accumulated reduction (unembed matmul over vocab=100288, or softmax/normalization) overflowing where fp32 accumulation would not. Candidate for the compute_config (HiFi4 + fp32_dest_acc) treatment on the remaining reduce/matmul ops, or a forward op keeping bf16 output where fp32 is needed. Left for a follow-up / delegation.
- **Practical guidance:** train bf16 GPTs in the standard mixed-precision form (fp32 master params + bf16 activations) — it works on TT today and is best practice regardless. Fully-bf16 *params* is uncommon; this fix unblocks its *compilation*, and the `-inf` numerics remain to be chased.
- The `APFloat::convert` approach is the general-correct way to read a float dense-attr element of unknown type; the same `getValues<float>()` anti-pattern elsewhere (e.g. composite clamp bounds) would have the same bf16 hazard.
