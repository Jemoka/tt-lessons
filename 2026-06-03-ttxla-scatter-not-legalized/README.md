# TT-XLA Could Not Legalize Non-Axis-0 `stablehlo.scatter` — Blocked On-Device Training (FIXED)

## Summary

A GPT pretraining run compiled and ran fine on CPU but failed to compile its
**training step** on Tenstorrent: the backward pass emits `stablehlo.scatter`
(scatter-add, the VJP of a forward gather), and tt-xla's SHLO→TTIR conversion
only handled scatter on **operand dim 0**, not a general axis. The unavoidable
RoPE `rotate_half` gather (`jnp.take(x, perm, axis=-1)`) has a VJP that scatters
on the **last** axis, which aborted the compile (`failed to legalize
'stablehlo.scatter'`). Embedding-table gradients (axis-0 scatter) already worked
via a dedicated pattern, so the blocker was specifically non-axis-0 scatter.

**Fixed** with two scoped changes in tt-mlir's single-dim scatter path: relax the
legality gate to admit a non-0 scatter axis, and rebuild the `ttir.scatter` index
tensor against the *actual* scatter axis (using `update_scatter_dims`) instead of
assuming dim 0. With the rebuilt plugin a last-axis scatter is **bit-exact vs CPU**
and a full theseus GPT (with RoPE) **trains end-to-end on the device**.

## Status

- Bug type: backend lowering gap — single-dim `stablehlo.scatter` was hardcoded to
  operand dim 0 in two places of the SHLO→TTIR conversion.
- Component: tt-mlir,
  `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:
  (1) `StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality` (~`:6086`)
  and (2) `extractElementWiseScatterIndices` (~`:6468`).
- Fixed locally: **yes.** Both spots patched; plugin rebuilt with clang-20
  (patched `libTTMLIRCompiler.so`). Patch:
  [supplemental/ttmlir_scatter_nonzero_axis_fix.patch](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/ttmlir_scatter_nonzero_axis_fix.patch).
- Verified: pure-JAX repro — take-axis=-1 grad on TT is bit-exact vs CPU
  (`max|cpu-tt| = 0.0`); a 2-layer GPT with RoPE trained 30 steps on TT, loss
  11.63 → 6.94 (finite, monotonically decreasing; ln(vocab)=11.52).
- No regression: the change is only reachable from the scatter rewrite, which is
  never emitted by inference (forward) — so qwen inference MLIR is provably
  unchanged. Empirically, qwen_parity on the rebuilt `.so` is deterministic and
  tracks HF tightly (default "Hello world" prompt: top5 overlap 5/5, jax loss
  5.534 vs hf 5.515, max diff 0.429 — within the TF32 matmul ceiling).
- Remaining: the axis-0 embedding-gradient scatter still shows a ~1.0 diff vs CPU
  with **repeated** indices (duplicate-token accumulation; pre-existing, separate
  from this fix — see Notes). The `ttnn.scatter` runtime handled `dim > 0` as-is;
  no runtime change was needed.

## Repositories

- TT-XLA: `/home/houjun/tt-xla`, branch `main`, commit `03f29ed01` (dirty; carries
  the matmul-precision patch from
  [2026-06-03-tt-matmul-fp32-accumulation-precision](/home/houjun/lessons/2026-06-03-tt-matmul-fp32-accumulation-precision/README.md),
  independent of this gap).
- tt-mlir submodule: `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`
  (dirty — `StableHLOToTTIRPatterns.cpp` carries both hunks of this lesson's fix;
  git hash `412daacc`). Rebuilt with clang-20 (surgical single-file recompile →
  `ar r libTTMLIRStableHLOToTTIR.a` → relink) to produce the patched
  `install/lib/libTTMLIRCompiler.so` (327 MB), installed on both this box and
  `tt-qb2`. The fully-resolved link command is saved at `/tmp/ttmlir_link_cmd.sh`
  and the single-file recompile at `/tmp/recompile_shlo.sh`, for fast relinks
  without a full superbuild.

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

Two scoped changes in the single-dim scatter path of
`StableHLOToTTIRPatterns.cpp`, both landed. Patch:
[supplemental/ttmlir_scatter_nonzero_axis_fix.patch](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/ttmlir_scatter_nonzero_axis_fix.patch).

**(1) Relax the legality gate.**
`StableHLOToTTIRScatterOpConversionPattern::checkBasicLegality` (~`:6086`)
rejected single-dim scatter unless `scatter_dims_to_operand_dims[0] == 0`. That
`notifyMatchFailure` was removed (replaced with `return success();`).
`matchAndRewrite` already passes `dim = scatter_dims_to_operand_dims[0]` straight
to `ttir.scatter`, so the op itself supports any axis.

**(2) Build the index tensor against the actual scatter axis.**
`extractElementWiseScatterIndices` (~`:6468`) previously reshaped the scatter
index by **appending trailing `1`s**, which places the index content on the
leading update axes — correct only when the scatter dim is 0. For a last-axis
scatter (RoPE `rotate_half`: `scatter_indices (8,1)`, `update (3,8)`,
`update_window_dims=[0]`) that produced a `ttir.repeat` from `(8,1)` that cannot
reach `(3,8)`. The fix uses the StableHLO scatter contract: `update_scatter_dims`
(update dims **not** in `update_window_dims`), in order, correspond to the batch
dims of `scatter_indices`. So the reshape now places each scatter-index batch dim
on its matching update scatter axis and makes the window dims size 1; the existing
`ttir.repeat` then broadcasts the window dims up to the update shape. The embedding
(axis-0) case reduces to the old shape, so it is unchanged. The `ttnn.scatter`
runtime handled `dim > 0` with no change.

Model-side mitigations explored but **not needed** with the backend fix:

- Loss: a one-hot + `log_softmax` loss avoids the integer-label gather's scatter.
  (Still a fine choice, but the backend now legalizes the gather form too.)
- RoPE: a slice/concat `rotate_half` avoids `jnp.take` — but it introduced a
  *separate* tt-metal forward crash on TT (see Notes), so the `jnp.take` form
  (now fully supported) is the right one to keep.

## Minimal Reproducer

[supplemental/repro_scatter_legalize.py](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py)
— standalone JAX. Takes the gradient on CPU vs TT of (a) a gather-based
`rotate_half` (last-axis scatter), (b) a slice-based `rotate_half`, and (c) an
embedding lookup `take(W, idx)` (axis-0 scatter). On the stock plugin (a) aborts
the TT compile; with the fix all three compile and (a)/(b) are bit-exact.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed:

```bash
source .venv/bin/activate
TT_VISIBLE_DEVICES=0 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  python /home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/supplemental/repro_scatter_legalize.py
```

## Verification

Three states of the same repro show the fix landing in two steps.

Before any change (stock plugin) — last-axis scatter aborts at legalization:

```text
[gather (take)   ] TT grad FAILED: INTERNAL: Error code: 13   (failed to legalize stablehlo.scatter)
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
[embed grad      ] TT grad OK   max|cpu-tt|=1.000e+00   (axis-0 EmbeddingBackward; dup-index caveat)
```

After change (1) alone (legality relax) — gets past legalization, fails one layer
deeper in index construction (this is why (2) is also needed):

```text
[gather (take)   ] TT grad FAILED: 'ttir.repeat' op Input (64,1) does not repeat to (8,64)
```

After changes (1)+(2) (final patch) — last-axis scatter runs and is **bit-exact**:

```text
[gather (take)   ] TT grad OK   max|cpu-tt|=0.000e+00
[slice/concat    ] TT grad OK   max|cpu-tt|=0.000e+00
[embed grad      ] TT grad OK   max|cpu-tt|=1.000e+00   (dup-index caveat unchanged)
```

End-to-end: a 2-layer theseus GPT **with RoPE** trains on the device (the train
step's backward emits the now-supported RoPE scatter):

```text
step   0 | loss 11.6270
step  10 | loss 9.1855
step  20 | loss 8.0371
step  29 | loss 6.9424
first loss 11.6270 -> last loss 6.9424  (finite=True, decreased=True)
ln(vocab) reference = 11.5158
```

Regression check — qwen_parity on the rebuilt plugin (default "Hello world"
prompt, deterministic across two runs). The change only touches scatter lowering,
which never appears in inference, so forward output is provably unchanged; the
parity tracks HF tightly:

```text
max diff: 0.4293   mean diff: 0.0897   top5 overlap: 5   roundtrip 0.0
hf loss: 5.5148    jax loss: 5.5339
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
- **Model-side rotate_half rewrite is not a clean workaround anyway:** replacing
  RoPE's `jnp.take` with a bit-identical slice/concat (0.0 diff on CPU) removes the
  scatter but triggers a *separate* tt-metal compile crash in the forward on TT
  (INTERNAL err 13 with a C++ stack dump, no MLIR legalize error). That confirmed
  the right fix was the backend scatter generalization (this lesson), not a model
  rewrite — and the `jnp.take` form now works end-to-end.
- The fix is scoped to the **single-dim** scatter path. Multi-dim scatter
  (`scatter_dims_to_operand_dims.size() > 1`) goes through a different, already-
  general code path (`flattenMultiDimScatterIndices`) and was not touched.
- Same arch/LLK on Wormhole and Blackhole; expect the fix to apply equally there.
