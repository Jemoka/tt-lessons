# TT-XLA Aborts Lowering `tenstorrent.uniform` for Fully-bf16 Models (`getValues<float>()` on a bf16 attr)

## Summary

Compiling a fully-bf16 model (params AND activations bfloat16) for Tenstorrent
crashes the compiler — not the runtime — during StableHLO→TTIR lowering:
`mlir/IR/BuiltinAttributes.h:423 DenseElementsAttr::getValues() [T=float]:
Assertion 'succeeded(range) && "element type cannot be ..."'` → core dump. The
offending op is the `tenstorrent.uniform` composite (random-uniform / dropout
RNG), whose lowering reads its scalar `low`/`high` bound constants with
`getValues<float>()`. That accessor requires f32 storage; in a fully-bf16 model
the bounds are bf16, so the typed read asserts and aborts.

The fix is two lines: read the bounds type-agnostically as `APFloat`
(`getValues<llvm::APFloat>()[0].convertToDouble()`) — an idiom already used
elsewhere in the same conversion library. Standard mixed precision
(params fp32, activations bf16 — the recommended config) is unaffected because its
constants stay f32.

## Status

- Bug type: compiler crash (assertion/abort) in StableHLO→TTIR composite lowering;
  typed `DenseElementsAttr` accessor used on a non-f32 attr.
- Component: `tt-mlir`
  `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp:261-262`
  (`TenstorrentUniformToRandConversionPattern`, `tenstorrent.uniform` → `ttir.rand`).
- Fixed locally: **No** — root-caused + exact fix identified (below); not yet built/verified.
- Scope: only **fully-bf16-parameter** models (uncommon — fp32 master weights with
  bf16 activations is best practice and works). Mixed precision and qwen_parity are
  unaffected.
- Pinned by **static analysis** (no DEBUG-build backtrace needed): `getValues<float>()`
  occurs exactly twice in the whole `lib/Conversion/` tree, both at these two lines.

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, `03f29ed01` (dirty; session fixes applied).
- `tt-mlir` (submodule) — `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, `412daacc4`.

## Host Environment

4× Blackhole p150b (tt-qb2), jax/jaxlib 0.7.1, ARCH_NAME=blackhole. The abort is at
compile time (lowering), so it reproduces regardless of chip availability.

## User-Visible Failure

A fully-bf16 GPT smoke (params=bfloat16, activations=bfloat16) aborts during compile:

```text
mlir/IR/BuiltinAttributes.h:423: DenseElementsAttr::getValues() [T = float]:
  Assertion `succeeded(range) && "element type cannot be iterated as the given type"'
  failed.   (core dumped, during StableHLO→TTIR lowering)
```

Mixed precision (params=fp32, act=bfloat16) on the same model trains fine
(loss 11.56 → 6.98, monotonic) — so the trigger is bf16 *parameter* constants, not
bf16 activations.

## Root Cause

`TenstorrentUniformToRandConversionPattern` lowers the `tenstorrent.uniform`
composite to `ttir.rand`, extracting the uniform's scalar `low`/`high` bounds from
constant operands (`StableHLOLegalizeCompositePass.cpp`):

```cpp
auto lowValue  = mlir::cast<DenseFPElementsAttr>(lowOp.getValue());
auto highValue = mlir::cast<DenseFPElementsAttr>(highOp.getValue());
...
auto lowAttr  = rewriter.getF32FloatAttr(lowValue.getValues<float>()[0]);   // :261
auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<float>()[0]);  // :262
```

`DenseElementsAttr::getValues<float>()` builds an iterator that asserts the
attribute's element type is f32 (`BuiltinAttributes.h:423`). In a fully-bf16 model
the `low`/`high` constants are **bf16**, so the typed read fails the assertion and
aborts. These two lines are the *only* `getValues<float>()` (or `<double>`) calls in
the entire `lib/Conversion/` tree, so this is the whole gap — no other site needs
auditing.

## Fix

Read the bounds type-agnostically via `APFloat` (works for bf16/f16/f32 — bf16 ⊂ f32
⊂ f64, lossless for these scalar bounds). This mirrors an idiom already present in
the same library at `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:3858`
(`valAttr.getValues<APFloat>()[0].convertToDouble()`):

```cpp
auto lowAttr  = rewriter.getF32FloatAttr(lowValue.getValues<llvm::APFloat>()[0].convertToDouble());
auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<llvm::APFloat>()[0].convertToDouble());
```

Build: surgical recompile of `StableHLOLegalizeCompositePass.cpp.o` → relink
`libTTMLIRCompiler.so` (no full rebuild needed).

## Minimal Reproducer

`supplemental/smoke_bf16.py` (the primary's smoke; theseus GPT+RoPE+AdamW). Set
`SMOKE_PARAM=bfloat16 SMOKE_ACT=bfloat16` to trigger the abort; `SMOKE_PARAM=float32
SMOKE_ACT=bfloat16` (mixed precision) trains fine as the contrast. A fully theseus-free
repro would be: lower any `jax.random.uniform`-containing graph with bf16 `minval`/`maxval`
constants through tt-xla.

## Reproduction Steps

```bash
cd /home/houjun/theseus && source .venv/bin/activate
TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  SMOKE_PARAM=bfloat16 SMOKE_ACT=bfloat16 \
  python /home/houjun/lessons/2026-06-03-ttxla-uniform-rand-bf16-getvalues-float/supplemental/smoke_bf16.py
# -> compiler abort at StableHLOLegalizeCompositePass.cpp:261 (getValues<float> on bf16)
# Contrast (works): SMOKE_PARAM=float32 SMOKE_ACT=bfloat16 ...
```

## Verification

Before fix: fully-bf16 smoke aborts in lowering (assertion above). Mixed precision
trains (loss 11.56 → 6.98). After the APFloat fix, the fully-bf16 smoke is expected to
compile and train. (Verification on chip 1 pending the rebuilt `libTTMLIRCompiler.so`.)

## Notes

- This is a low-priority gap: fp32 master weights + bf16 activations is best practice
  and already works on TT (native bf16 matmul + fp32 accumulation via the
  packer_l1_acc/fp32_dest_acc fix). Fully-bf16 *params* is uncommon.
- General lesson: typed `DenseElementsAttr::getValues<T>()` is a footgun for any T that
  pins the storage element type — prefer `getValues<APFloat>()`/`getValues<APInt>()` and
  convert, so the pattern survives bf16/f16 inputs. The two sites here were the only
  offenders in the conversion library, but the same care applies to any new composite
  lowering that reads constant attrs.
