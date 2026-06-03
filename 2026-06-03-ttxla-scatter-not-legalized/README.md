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
  also gathers; use a one-hot + `log_softmax` loss (scatter-free, what the theseus
  model does) to avoid it.

So embedding gradients train on TT, but any model that gathers on a non-0 axis in
the forward (e.g. RoPE via `jnp.take`) cannot be trained on-device until the
general scatter legality check is broadened. Forward/inference is unaffected —
scatter only appears in gradients (which is why qwen_parity forward works).

## Status

- Bug type: backend lowering gap — narrow `stablehlo.scatter` legality in SHLO→TTIR.
- Component: tt-mlir `StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality`
  (`StableHLOToTTIRPatterns.cpp:6086`).
- Fixed locally: **no** (needs a tt-mlir rebuild). Isolated precisely: axis-0
  scatter (embedding grad) already legalizes; non-axis-0 scatter (RoPE `jnp.take`
  VJP) does not.
- Impact: TT inference/forward works; embedding-gradient training works; on-device
  training of models that **gather on a non-0 axis in the forward** (e.g. RoPE via
  `jnp.take`) is blocked.
- Reproduced: standalone pure-JAX (gather/take VJP fails, slice/concat OK, embed
  axis-0 OK), and via a GPT pretraining step.

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty; the
  patched plugin from
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md)
  was running, but the gap is independent of that patch).

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
   logit; VJP scatters. Avoid via one-hot + `log_softmax` (the theseus model loss).
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

Broaden `StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality`
(`StableHLOToTTIRPatterns.cpp:6086`) to accept scatter on a non-0 operand dim
(map to `ttir.scatter` with the right `dim`), so last-axis gather VJPs (RoPE etc.)
legalize. The full TTIR→TTNN→runtime scatter stack already exists
(`ttir.scatter`→`ttnn.scatter`→`runtime/.../data_movement/scatter.cpp`). This is a
tt-mlir change + rebuild. Not done here (the tt-mlir superbuild is impractical to
rebuild in this environment; see the matmul-precision lesson's build notes).

Partial model-side mitigations (insufficient alone, but reduce scatter count):

- Loss: one-hot + `log_softmax` instead of integer-label gather.
- RoPE: slice/concat `rotate_half` instead of `jnp.take`.

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

```text
[gather (take)   ] TT grad FAILED: INTERNAL: Error code: 13   (last-axis scatter: failed to legalize stablehlo.scatter)
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
[embed grad      ] TT grad OK   max|cpu-tt|=1.000e+00   (axis-0 scatter via EmbeddingBackward; see duplicate-index caveat)
```

The last-axis gather VJP aborts the TT compile; the slice/concat gradient is
bit-exact; the axis-0 embedding gradient **legalizes and runs** (its ~1.0 diff is
the duplicate-index accumulation caveat, not a legalization failure).

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
