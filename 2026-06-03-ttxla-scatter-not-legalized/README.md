# TT-XLA Cannot Legalize `stablehlo.scatter` — Blocks On-Device Training

## Summary

Theseus GPT pretraining runs fine on CPU but fails to compile its **training
step** on Tenstorrent: the backward pass emits `stablehlo.scatter` (scatter-add),
which tt-xla's SHLO→TTIR conversion cannot legalize. Concretely, three distinct
gathers in the GPT forward each produce a scatter in their gradient and abort the
compile one after another: the integer-label cross-entropy loss, `RoPE.rotate_half`
(`jnp.take` of a permutation), and — unavoidably — the **token-embedding lookup**
(whose VJP scatter-adds gradients back into the embedding table).

Forward / inference is unaffected (Qwen parity ran fine), because scatter only
appears in the gradient. The one-line shape of the situation: tt-xla legalizes
`gather`/`stablehlo.dynamic_gather` for the forward but has no lowering for
`stablehlo.scatter`, so **any model that gathers in the forward cannot be trained
on-device** until scatter is supported.

## Status

- Bug type: backend lowering gap (unsupported op in SHLO→TTIR).
- Component: tt-xla / tt-mlir SHLO→TTIR conversion (`stablehlo.scatter`).
- Fixed locally: **no.** Documented + isolated. Model-side workarounds remove
  *some* scatters (one-hot loss, slice-based rotate_half) but **cannot** remove
  the embedding-gradient scatter — so a real fix requires scatter legalization.
- Impact: TT inference/forward works; TT on-device **training** of any
  embedding/gather-using model (i.e. essentially all transformers) is blocked.
- Reproduced: standalone pure-JAX (no theseus) + via theseus GPT pretraining.

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty; the
  patched plugin from the matmul-precision lesson is what was running).
- Theseus: `/home/houjun/theseus`, branch `feat/tenstorrent` (dirty).

## Host Environment

- tt-qb2 (`tt-qb-ac-02`): 4× Blackhole `p150b`, Python 3.12.13, JAX/jaxlib 0.7.1.
- Run on chip 0 (`TT_VISIBLE_DEVICES=0`, `CONVERT_SHLO_TO_SHARDY=1`).

## User-Visible Failure

`theseus run gpt/train/pretrain` (or a standalone train step) on TT aborts during
compile. First the loss gather:

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
that is a `scatter`-add. tt-xla lowers the forward gather but has **no SHLO→TTIR
legalization for `stablehlo.scatter`**, so the gradient graph fails to compile.
Sources of gather (hence scatter-in-backward) on the GPT pretraining path:

1. `optax.softmax_cross_entropy_with_integer_labels` — gathers the target-class
   logit; VJP scatters. *Avoidable*: use one-hot + `log_softmax` (the theseus
   model's own loss already does this).
2. `RoPE.rotate_half` (`rope.py:45`) — `jnp.take` of an index permutation; VJP
   scatters. *Avoidable*: `concatenate([-x2, x1])` slicing is bit-identical and
   scatter-free (verified, 0.0 diff).
3. **Token embedding** `jnp.take(wte, idx)` — VJP scatter-adds gradients into the
   embedding rows. **Not avoidable** — this is the definition of an embedding
   gradient. So even with (1) and (2) fixed, training still emits a scatter.

Because of (3), on-device training of a transformer is impossible without scatter
support; (1)/(2) are not enough.

## Fix

Proper fix (etiological): add `stablehlo.scatter` legalization to the tt-mlir
SHLO→TTIR conversion (lower to a TTIR scatter / `embedding_bw`-style op). This is
a tt-mlir change + rebuild, not a flag flip, and unblocks all gather-VJP cases at
once. Not done here.

Partial model-side mitigations (insufficient alone, but reduce scatter count):

- Loss: one-hot + `log_softmax` instead of integer-label gather.
- RoPE: slice/concat `rotate_half` instead of `jnp.take`.

These do not remove the embedding-gradient scatter, so they do not enable TT
training by themselves.

## Minimal Reproducer

[supplemental/repro_scatter_legalize.py](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py)
— standalone JAX. Takes the gradient of (a) a gather-based `rotate_half` and (b) a
slice-based one, on CPU vs TT.

## Reproduction Steps

```bash
source /home/houjun/theseus/.venv/bin/activate
TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py
```

## Verification

```text
[gather (take)   ] TT grad FAILED: INTERNAL: Error code: 13   (failed to legalize stablehlo.scatter)
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
```

The gather gradient aborts the TT compile; the slice/concat gradient compiles and
is bit-exact vs CPU.

## Notes

- The synthetic GPT pretraining path itself is correct and trains fine on CPU
  (loss 11.68 → 6.61; see the synthetic-data harness). This lesson is specifically
  about *on-device TT training*, not the harness.
- Forward-only TT use (inference, qwen_parity) is unaffected — scatter only
  arises in gradients.
- Same arch/LLK on Wormhole and Blackhole; expect the same gap there.
