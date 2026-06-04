# TT ttnn.reshape Crashes Flattening a Tile-Padded Non-32-Aligned Dim (RoPE gather-grad, training)

## Summary

On Tenstorrent Blackhole, on-device training of a GPT with RoPE crashes at runtime
inside a `ttnn.reshape`:
`reshape_common.cpp:50: new_volume == old_volume … Invalid arguments to reshape`
(`infer_dims_for_reshape`). The failing op flattens a 4-D tensor whose last logical
dim is `n_head = 4` — which is **not a multiple of the 32×32 tile** — into a 2-D
tensor. In TILE layout that size-4 dim is padded up to a full 32-wide tile, so the
tensor's **physical** volume is 8× its logical volume; flattening to an unpadded
layout makes the physical volumes disagree and tt-metal aborts.

The reshape is emitted by the **gradient of the RoPE `rotate_half` gather**
(`jnp.take(x, idx, axis=-1)` over a constant index permutation) inside the
attention jvp of the training loss. It is a tt-mlir reshape-lowering bug: it emits
a direct `ttnn.reshape` between a tile-padded layout and an unpadded one without
routing through row-major / `to_layout` (which would drop the padding).

## Status

- Bug type: backend correctness — `ttnn.reshape` lowering does not account for
  tile padding when flattening a non-32-aligned dim; runtime FATAL.
- Component: tt-mlir TTIR→TTNN reshape lowering (the `ttnn.reshape` between
  `#ttnn_layout129` and `#ttnn_layout130` below); surfaced via the RoPE gather VJP.
- Fixed locally: **Yes** (verified). A `ttnn.reshape` operand workaround forces
  RowMajor layout when an operand is tile-padded (physical volume ≠ logical),
  routing the flatten through row-major so padding is dropped. See Fix. Confirmed:
  the RoPE gather-grad reshape (`64x8x128x4 → 64x4096`) now lowers and the trainer
  advances past it (the failure then moves to a separate optimizer clip-binding
  bug — see
  [2026-06-03-ttxla-trainer-clip-wte-global-id-misbind](/home/houjun/lessons/2026-06-03-ttxla-trainer-clip-wte-global-id-misbind/README.md)).
  qwen_parity inference is byte-identical (no regression — the workaround only
  fires on tile-padded reshapes, which inference does not hit).
- Trigger: a reshape that flattens a tile-padded dim whose size is not a multiple
  of 32 (here `n_head = 4`). A model with `n_head % 32 == 0` would avoid it.
- Scope: training (the gather-grad path). Inference is unaffected (no such reshape
  in the forward graph; inference parity intact). This is the **third** distinct
  gap in the TT training bring-up, after the scatter legalization fix
  ([2026-06-03-ttxla-scatter-not-legalized](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/README.md))
  and the Shardy guard
  ([2026-06-03-ttxla-shardy-reshard-collectives-assert](/home/houjun/lessons/2026-06-03-ttxla-shardy-reshard-collectives-assert/README.md)).
- Related (same tile-padding family):
  [2026-06-03-ttxla-embedding-bw-tile-padding-grad](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)
  — a different op (`ttnn.embedding_bw`), same root mechanism (a non-32-aligned dim
  padded to a tile leaking/mismatching in a backward op).

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`, `03f29ed01` (dirty; scatter-fix
  + Shardy-guard patches applied).

## Host Environment

4× Blackhole p150b, jax/jaxlib 0.7.1, ARCH_NAME=blackhole. Run pinned to chip 1
(`TT_VISIBLE_DEVICES=1`), patched plugin. Model: GPT n_embd 256, **n_head 4**
(→ head_dim 64), 2 layers, block_size 128, RoPE, vocab 100288; batch 8, seq 128.

## User-Visible Failure

A GPT training step on TT aborts in the first train step:

```text
TT_FATAL @ .../reshape_view/reshape_common.cpp:50: new_volume == old_volume
info: Invalid arguments to reshape
 --- ttnn::operations::data_movement::detail::infer_dims_for_reshape(...)
 --- ttnn::reshape(...)
 --- tt::runtime::ttnn::operations::data_movement::run(ReshapeOp const*, ...)
```

## Root Cause

The failing op, from the train_step TTNN IR dump (`TTXLA_LOGGER_LEVEL=DEBUG`):

```text
%137 = "ttnn.reshape"(%136) <{shape = [64, 4096]}>
       : (tensor<64x8x128x4xbf16, #ttnn_layout129>) -> tensor<64x4096xbf16, #ttnn_layout130>
  loc("jit(train_step)/jvp(GPT)/GPT.decode/blocks_0/attn/attn.preprocess_qkv/jit(_take)")
```

Source: RoPE `rotate_half` —
`rotated = jnp.take(x, rotate_half_indices, axis=-1)` (constant indices). The
reshape is in the gradient (jvp) of that gather.

The two layouts (the proof):

```text
#ttnn_layout129 (64x8x128x4):  memref<2048x1   x !ttcore.tile<32x32, bf16>, #dram>
   logical (65536, 4)  ->  PADDED physical (65536, 32)   ->  phys volume 2,097,152
#ttnn_layout130 (64x4096):     memref<2x128    x !ttcore.tile<32x32, bf16>, #dram>
   physical (64, 4096) == logical                        ->  phys volume   262,144
```

The size-4 (`n_head`) dim is padded to a full 32-wide tile (`…x1x tile<32x32>`), so
the input's physical volume (2,097,152) is 8× the target's (262,144).
`infer_dims_for_reshape` checks `new_volume == old_volume` on the *physical* volumes
and aborts. (No static *logical* volume mismatch exists — all 180 reshapes verify;
the mismatch is purely tile padding.) Same tile-padding family as the
embedding-backward grad bug
([2026-06-03-ttxla-embedding-bw-tile-padding-grad](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/README.md)).

## Fix

**Implemented + verified.** A `ttnn.reshape` operand workaround in
`TTNNWorkaroundsPass`: force RowMajor layout for the reshape operands when either
operand is tile-padded, so the flatten runs in row-major (physical == logical, no
padding) and is re-tilized afterward by the workaround framework. Row-major reshape
is always volume-correct, and tile-aligned reshapes are untouched. Patch:
[supplemental/ttmlir_reshape_rowmajor_workaround.patch](/home/houjun/lessons/2026-06-03-ttxla-reshape-tilepadded-dim-flatten/supplemental/ttmlir_reshape_rowmajor_workaround.patch).

Three changes in tt-mlir (rebuilt `libTTMLIRCompiler.so`, surgical relink):
1. `TTNNOperandsWorkaroundsFactory::createReshapeOpOperandsWorkarounds`
   (`lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`): an `isTilePadded()` helper
   (`product(layout.getScalarShardShape()) != product(tensor.getShape())` for a
   tiled layout — this catches padding from the layout's *collapsed* memref, which
   a per-dim "logical last-2 % 32" check misses, e.g. the attention
   `dot_general reshapeRhs`); force `Layout::RowMajor` when **either** the input or
   output is tile-padded.
2. The factory signature gained the `outputType` (a `[1]`-target reshape whose
   *output* is the padded operand is also covered).
3. `TTNNOps.td` `ReshapeOp::getOperandsWorkarounds` passes `getResult().getType()`.

Verification (on HW, helper-confirmed): the RoPE `_take` grad reshape becomes
row-major (`#ttnn_layout` with no `tile<32x32>`, phys == logical); ~150 `row_major`
to_layouts inserted across the module; the trainer no longer FATALs here. The
**evolution of the detection heuristic mattered**: an initial input-only,
logical-last-2-`%32` check fixed the RoPE reshape but missed the output-tiled
`dot_general reshapeRhs`; switching to the both-operands physical-vs-logical-volume
check (above) covers all tile-padded reshapes in the module.

(Earlier alternative — have the gather/take VJP lowering avoid placing the
size-`n_head` dim in the tiled last-two-dims position — is unnecessary given the
operand workaround.)

## Minimal Reproducer

The bug is **layout/context-dependent and has resisted minimal reduction**. Three
standalone JAX attempts all ran **clean on hardware** (no FATAL) — tt-mlir picks an
*unpadded* layout for the reshape outside the exact trainer graph. **The only
confirmed reproducer is the full trainer** (`gpt/train/pretrain`, below). The
standalone scripts are kept only to document the layout-dependence; all are NEGATIVE
(HW-verified non-reproducing):

- `supplemental/repro_rope_reshape_NEGATIVE.py` — bare gather-grad on `(8,128,4,64)`;
  TT grad runs (≈1.5e-2 bf16), no FATAL, at n_head 4 and 32.
- `supplemental/repro_attn_rope_grad_NEGATIVE.py` — standalone causal-attention block (qkv →
  RoPE `rotate_half` take → attention → loss grad) at the triggering shapes (n_embd
  256, n_head 4, head_dim 64, seq 128, batch 8), f32. **HW-verified: TT grad OK at
  both n_head=4 and n_head=32 — does NOT reproduce** (its docstring's "expected
  FATAL" predates HW verification and is wrong).
- `supplemental/repro_attn_rope_grad_bf16_NEGATIVE.py` — the same block with bf16
  activations (matching the trainer's `activation: bfloat16`). **Also OK — does NOT
  reproduce.**

The padded `#ttnn_layout129` only arises in the real trainer's `train_eval.loss`
jvp, so the reproducer is the trainer command in Reproduction Steps.

## Reproduction Steps

From a venv that has the TT PJRT plugin installed (the trainer is the only confirmed
reproducer; the standalone scripts are diagnostic negatives):

```bash
cd /home/houjun/theseus && source .venv/bin/activate
mkdir -p /tmp/reshape_pin_out2/data/synthetic
python -c "import numpy as np; r=np.random.default_rng(0); \
  r.integers(0,100288,size=4_000_000,dtype=np.uint32).tofile('/tmp/reshape_pin_out2/data/synthetic/train.bin'); \
  r.integers(0,100288,size=500_000,dtype=np.uint32).tofile('/tmp/reshape_pin_out2/data/synthetic/val.bin')"
TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole \
  theseus run reshape-pin \
  /home/houjun/lessons/2026-06-03-ttxla-reshape-tilepadded-dim-flatten/supplemental/reshape_pin_cfg.yaml \
  /tmp/reshape_pin_out2
```

Add `TTXLA_LOGGER_LEVEL=DEBUG` to dump the TTNN IR and confirm the `#ttnn_layout129`
padded memref on the `64x8x128x4 -> 64x4096` reshape.

## Verification

```text
Failing op : ttnn.reshape 64x8x128x4xbf16 (#ttnn_layout129, padded) -> 64x4096 (#ttnn_layout130)
Layouts    : input memref<2048x1 x tile<32x32>> (size-4 dim padded to 32) ; output unpadded
Phys vols  : 2,097,152 (in) vs 262,144 (target) -> new_volume != old_volume -> FATAL
Source     : RoPE rotate_half gather (jnp.take axis=-1), in the training-loss jvp
```

## Notes

- The bare pure-JAX gather-grad on `(8,128,4,64)`
  (`repro_rope_reshape_NEGATIVE.py`) runs fine on TT (≈1.5e-2 bf16 error, no FATAL)
  for both `n_head=4` and `n_head=32`, because tt-mlir lowers it with an unpadded
  layout. The bug only appears with the attention/jvp context that produces the
  transposed `64x8x128x4` tile-padded layout — hence the attention-block reproducer.
- `n_head=4` is the trigger here; the invariant is "flatten a tile-padded
  (not %32) dim". Qwen2.5-0.5B (n_head 14, head_dim 64) would hit the same class if
  trained with this RoPE-grad path — 14 is also not %32.
- This is the gradient path enabled by the scatter-legalization fix
  ([2026-06-03-ttxla-scatter-not-legalized](/home/houjun/lessons/2026-06-03-ttxla-scatter-not-legalized/README.md));
  the reshape gap is the next layer down.
