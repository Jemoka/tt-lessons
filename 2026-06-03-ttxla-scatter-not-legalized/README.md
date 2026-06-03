# TT-XLA Cannot Legalize `stablehlo.scatter` — Blocks On-Device Training

## Summary

A GPT pretraining run compiles and runs fine on CPU but fails to compile its
**training step** on Tenstorrent: the backward pass emits `stablehlo.scatter`
(scatter-add, the VJP of a forward gather), and tt-xla's SHLO→TTIR conversion
legalizes scatter only for a **narrow set of shapes**, not the general case.

Precise finding (the important nuance — verified op-by-op on hardware):

- **Axis-0 scatter (embedding-table gradient): SUPPORTED.** tt-xla has a dedicated
  `StableHLOToTTIREmbeddingBackwardOpConversionPattern`; `jnp.take(W, idx, axis=0)`
  (token embedding) trains on TT. *Caveat:* with repeated indices the TT result
  differed from CPU by up to 1.0 — duplicate token positions may not accumulate
  (scatter-add) correctly; see Notes.
- **Non-axis-0 / last-axis scatter: NOT SUPPORTED.** `RoPE.rotate_half`'s
  `jnp.take(x, perm, axis=-1)` (and `take_along_axis` / advanced indexing on a
  non-0 axis) has a VJP scatter whose `dimension_numbers` the general pattern's
  `checkBasicLegality` rejects (it requires single-dim scatter on operand-dim 0
  with `index_vector_dim == 1`). These abort the compile:
  `error: failed to legalize operation 'stablehlo.scatter'`.
- Integer-label cross-entropy (`optax.softmax_cross_entropy_with_integer_labels`)
  also gathers; use a one-hot + `log_softmax` loss (scatter-free) to avoid it.

So embedding gradients train on TT, but any model that gathers on a non-0 axis in
the forward (e.g. RoPE via `jnp.take`) cannot be trained on-device until the
general scatter legality check is broadened. Forward/inference is unaffected —
scatter only appears in gradients (which is why qwen_parity forward works).

## Status

- Bug type: backend lowering gap — narrow `stablehlo.scatter` legality in SHLO→TTIR,
  *plus* a second dim-0 assumption in the index-construction that follows.
- Component: tt-mlir, two spots in
  `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:
  (1) `StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality` (~`:6086`)
  and (2) `extractElementWiseScatterIndices` (~`:6463`).
- Fixed locally: **partially.** The legality check (1) was relaxed and the
  plugin rebuilt (clang-20, patched `libTTMLIRCompiler.so`); a non-axis-0 scatter
  now **passes legalization** — but it then fails downstream in (2), which still
  assumes scatter dim 0 and emits a wrong `ttir.repeat` (shape mismatch). So the
  relax is **necessary but not sufficient**; non-axis-0 scatter still cannot
  compile end-to-end.
- No regression: with the patched/rebuilt `.so`, qwen_parity is byte-identical to
  baseline (max diff 0.637, top5 4, jax loss 7.909) — the relax is inert for
  inference (scatter only appears in gradients).
- Impact: TT inference/forward works; embedding-gradient training (axis-0) works;
  on-device training of models that **gather on a non-0 axis in the forward**
  (e.g. RoPE via `jnp.take`) is still blocked, now one layer deeper.
- Reproduced: standalone pure-JAX (gather/take VJP fails, slice/concat OK, embed
  axis-0 OK), and via a GPT pretraining step. After the relax, the take-VJP error
  moved from "failed to legalize stablehlo.scatter" to a `ttir.repeat` shape
  mismatch in index construction.

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty; carries
  the matmul-precision patch from
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md),
  independent of this gap).
- tt-mlir submodule: `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`
  (dirty — `StableHLOToTTIRPatterns.cpp` carries the legality relax from this
  lesson). Rebuilt with clang-20 to produce the patched
  `install/lib/libTTMLIRCompiler.so` (327 MB); the stock `.so` is backed up at
  `/tmp/libTTMLIRCompiler.so.orig.bak`. The fully-resolved link command is saved
  at `/tmp/ttmlir_link_cmd.sh` for fast future relinks without a full superbuild.

## Host Environment

- 4× Blackhole `p150b`, Python 3.12.13, JAX/jaxlib 0.7.1.
- Run on chip 0 (`TT_VISIBLE_DEVICES=0`, `CONVERT_SHLO_TO_SHARDY=1`).

## User-Visible Failure

A GPT training step on TT aborts during compile. First the loss gather:

```text
loc(... "softmax_cross_entropy_with_integer_labels" ... "jit(step)/transpose(jvp(jit(_take)))" ...)
  error: failed to legalize operation 'stablehlo.scatter'
ERR| Failed to convert from SHLO to TTIR module
XlaRuntimeError: INTERNAL: Error code: 13
```

After switching to a one-hot (scatter-free) loss, the next gather — RoPE — fails
the same way:

```text
loc(... "RotaryPosEncoding.rotate_half" (rope.py:45) ... ) error: failed to legalize 'stablehlo.scatter'
```

`rope.py:45` is `jnp.take(x, self._rotate_half_indices, axis=-1)`.

## Root Cause

A forward `gather` (`jnp.take`, `take_along_axis`, advanced indexing) has a VJP
that is a `scatter`-add. tt-xla *does* have SHLO→TTIR scatter patterns
(`StableHLOToTTIRPatterns.cpp`): a cache-fill/update pattern, an
`EmbeddingBackward` pattern, and a general `StableHLOToTTIRScatterOpConversionPattern`
gated by `checkBasicLegality` (`:6086`). The general gate is **narrow**: for
single-dimensional scatter it requires `index_vector_dim == 1` and
`scatter_dims_to_operand_dims[0] == 0` (i.e. scatter into operand dim 0). Gathers
on the GPT pretraining path:

1. `optax.softmax_cross_entropy_with_integer_labels` — gathers the target-class
   logit; VJP scatters. Avoid via a one-hot + `log_softmax` loss.
2. `RoPE.rotate_half` (`rope.py:45`) — `jnp.take(x, perm, axis=-1)`; VJP scatters
   on the **last** axis → `scatter_dims_to_operand_dims[0] != 0` → rejected by
   `checkBasicLegality` → **fails to legalize**. (A slice/concat `rotate_half` is
   bit-identical and scatter-free on CPU, but introduced a separate tt-metal
   compile crash on TT in the forward — see Notes — so it is not a clean fix.)
3. **Token embedding** `jnp.take(wte, idx, axis=0)` — VJP scatters into operand
   dim 0 → matches the `EmbeddingBackward` pattern → **legalizes and runs on TT.**

So the blocker for on-device transformer training is specifically **non-axis-0
scatter** (item 2), not scatter in general; the unavoidable embedding-gradient
scatter (item 3) already works.

## Fix

Two changes are required; only the first was made here.

**(1) Relax the legality gate — DONE.**
`StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality`
(`StableHLOToTTIRPatterns.cpp:~6086`) rejected single-dim scatter unless
`scatter_dims_to_operand_dims[0] == 0`. That `notifyMatchFailure` was removed
(replaced with `return success();`), and the plugin was rebuilt
(clang-20 — see Notes on the toolchain). A non-axis-0 scatter now legalizes.
Patch: [supplemental/ttmlir_scatter_legality_relax.patch](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/ttmlir_scatter_legality_relax.patch).

**(2) Generalize index construction — NOT DONE (the remaining blocker).**
`extractElementWiseScatterIndices` (`StableHLOToTTIRPatterns.cpp:~6463`) builds
the `ttir.scatter` index tensor assuming scatter dim 0: it reshapes the index by
appending trailing `1`s (placing the scattered content on dim 0) and then
`ttir.repeat`s it along the `update_window_dims`. For a last-axis scatter (RoPE
`rotate_half`, index shape `(64,)`, update shape `(8,64)`) this produces a
`ttir.repeat` from `(64,1)` that cannot reach `(8,64)`:

```text
error: 'ttir.repeat' op Input tensor shape (64,1) does not repeat to output (8,64)
```

Finishing non-axis-0 scatter means rebuilding this index tensor against the actual
scatter dim (place content on `scatter_dims_to_operand_dims[0]`, repeat along the
other window dims) — and then confirming the `ttnn.scatter` runtime
(`runtime/.../data_movement/scatter.cpp`) handles `dim > 0`. The TTIR→TTNN→runtime
scatter stack exists, but its dim-0 assumptions have not been audited past the
legality check.

Partial model-side mitigations (insufficient alone, but reduce scatter count):

- Loss: one-hot + `log_softmax` instead of integer-label gather.
- RoPE: slice/concat `rotate_half` instead of `jnp.take` — but this introduced a
  *separate* tt-metal forward crash on TT (see Notes), so it is not a clean
  workaround.

These do not remove the embedding-gradient scatter, so they do not enable TT
training by themselves.

## Minimal Reproducer

[supplemental/repro_scatter_legalize.py](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py)
— standalone JAX. Takes the gradient on CPU vs TT of (a) a gather-based
`rotate_half`, (b) a slice-based `rotate_half`, and (c) an embedding lookup
`take(W, idx)` — the unavoidable case (3). (a) and (c) abort the TT compile; (b)
compiles and is bit-exact.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py
```

## Verification

Before the relax (stock plugin):

```text
[gather (take)   ] TT grad FAILED: INTERNAL: Error code: 13   (last-axis scatter: failed to legalize stablehlo.scatter)
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
[embed grad      ] TT grad OK   max|cpu-tt|=1.000e+00   (axis-0 scatter via EmbeddingBackward; see duplicate-index caveat)
```

After the relax (patched/rebuilt plugin) the last-axis gather VJP gets **past
legalization** and fails one layer deeper, confirming the legality gate was the
first (but not only) blocker:

```text
[gather (take)   ] TT grad FAILED: 'ttir.repeat' op Input (64,1) does not repeat to (8,64)   (index construction still assumes dim 0)
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
[embed grad      ] TT grad OK   max|cpu-tt|=1.000e+00
```

Regression check — qwen_parity on the patched/rebuilt plugin is byte-identical to
the pre-patch baseline (scatter is inert for inference):

```text
max diff: 0.6373   mean diff: 0.0933   top5 overlap: 4   roundtrip 0.0
hf loss: 7.8484    jax loss: 7.9089
```

## Notes

- The GPT pretraining path itself is correct and trains fine on CPU (loss
  11.68 → 6.61). This lesson is specifically about *on-device TT training*.
- Forward-only TT use (inference) is unaffected — scatter only arises in
  gradients.
- **Embedding duplicate-index caveat:** the axis-0 embedding-gradient scatter
  legalizes, but with repeated indices the TT gradient differed from CPU by up to
  1.0 — duplicate token positions may be **overwritten rather than scatter-added**.
  If real, that silently corrupts training gradients (the most common case — a
  token appearing twice in a batch). Worth a focused correctness probe before
  trusting TT embedding training.
- **Model-side rotate_half rewrite is not a clean workaround:** replacing RoPE's
  `jnp.take` with a bit-identical slice/concat (0.0 diff on CPU) removed the
  scatter but triggered a *separate* tt-metal compile crash in the forward on TT
  (INTERNAL err 13 with a C++ stack dump, no MLIR legalize error). Reverted to
  protect qwen forward parity (rope.py is shared). So the real fix must be the
  backend scatter-legality broadening, not a model rewrite.
- Same arch/LLK on Wormhole and Blackhole; expect the same gap there.
