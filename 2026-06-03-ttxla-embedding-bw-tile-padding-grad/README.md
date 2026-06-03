# TT Embedding-Backward Corrupts Gradients When vocab/embed-dim Are Not Multiples of the 32 Tile

## Summary

On Tenstorrent Blackhole, the gradient of an embedding lookup (`jnp.take` / gather
VJP, lowered to the embedding-backward scatter) is wrong when the weight table's
vocab or embedding dimension is **not a multiple of 32** (the tile height/width).
A spurious value (~+1.0 for a `sum` loss — i.e. the all-ones cotangent) leaks from
the padded tile region into the real weight gradient, in a structured pattern: odd
columns of even rows. When both dimensions are tile-aligned (multiples of 32), the
gradient is exact.

Duplicate-index accumulation itself is correct (a token appearing N times in the
batch gets N× its gradient). The bug is purely a tile-padding leak in the backward
op; the forward gather is unaffected. Practical impact: silent training-gradient
corruption for any model whose vocab or embedding dim is not 32-aligned. Qwen2.5-0.5B
is *not* affected — its dims are tile-aligned (vocab 151936 = 32×4748, hidden 896 =
32×28).

## Status

- Bug type: backend correctness — embedding-backward (gather-VJP scatter) reads/writes
  padded tile region, corrupting gradients for non-32-aligned shapes.
- Component: tt-xla / tt-mlir embedding-backward lowering + tt-metal embedding_bw
  kernel (exact layer not yet pinned to a file; characterized empirically on HW).
- Fixed locally: **No.** This is a characterization; root cause is pinned to tile
  padding (vocab/dim % 32 != 0), fix not yet implemented.
- Scope: affects training (backward) only; forward gather is correct (separate bf16
  embedding cast is documented in
  [2026-06-03-ttxla-fp32-embedding-bf16-cast](/home/houjun/lessons/2026-06-03-ttxla-fp32-embedding-bf16-cast/README.md)).
- Does NOT affect Qwen2.5-0.5B (tile-aligned dims); does affect any non-aligned vocab/dim.

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`,
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4` (dirty; patched plugin on tt-qb2).
- `tt-mlir` (submodule) — `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`,
  `412daacc440f10bb98ccc685c311b01f1fadab70`.
- `theseus` — `/home/houjun/theseus`, `feat/tenstorrent`,
  `f085ca67fa68ef08d63668cd7f866b2b8147839e` (repros are theseus-free).

## Host Environment

`tt-qb-ac-02` (tt-qb2.stanford.edu), 4× Blackhole p150b, jax/jaxlib 0.7.1,
ARCH_NAME=blackhole. Probe pinned to chip 1 (`TT_VISIBLE_DEVICES=1`), patched
plugin. Startup `TT_FATAL ... remote mmio` lines are benign single-chip warnings.

## User-Visible Failure

`loss = sum(W[idx])` with `idx = [2,2,2,5,5,9,0,0]`, W shape (vocab=16, dim=4).
The correct gradient row for token `t` is `count(t)` in every column. On TT the
even rows pick up a spurious +1.0 in the odd columns:

```text
row count  cpu_grad        tt_grad
 0   2   [2,2,2,2]      [2,3,2,3]    <- cols 1,3 +1
 2   3   [3,3,3,3]      [3,4,3,4]
 4   0   [0,0,0,0]      [0,1,0,1]    <- absent row still corrupted
 5   2   [2,2,2,2]      [2,2,2,2]    (odd row: clean)
 9   1   [1,1,1,1]      [1,1,1,1]
max|cpu-tt| grad = 1.0
```

Accumulation is correct (rows 0,2,5,9 carry their exact counts in the even columns).

## Root Cause

The corruption is governed by the 32×32 tile geometry, confirmed by two sweeps.

Column sweep (vocab=16), per-column mean of `(tt − cpu)` over rows:

```text
dim= 1 : [0]                         clean (no odd column)
dim= 2 : [0, .5]                     odd column corrupted (half the rows)
dim= 4 : [0,.5,0,.5]
dim= 8 : [0,.5,0,.5,0,.5,0,.5]
dim=16 : [0,1,0,1, ... ]             all rows now hit
dim=32 : [0,1,(cols 0-15) ...,0,0,(cols 16-31 clean)]
```

Row sweep (dim=2), rows carrying a spurious gradient:

```text
vocab= 4 : [0,2]        vocab= 8 : [0,2,4,6]     vocab=16 : [0,2,4,6]
vocab=32 : none (clean) vocab=33 : [32] only     vocab=64 : none (clean)
```

Reading: when vocab and dim are multiples of 32 the gradient is exact (vocab=32,64).
When they are not, the embedding-backward op leaks the all-ones cotangent from the
padded tile region into odd columns of even rows of the real gradient. vocab=33
leaks exactly at the overflow row 32 (first element of the second, mostly-padded
tile). The per-element spurious value is exactly the cotangent (1.0 for a `sum`
loss), and `dim=1` is clean because there is no odd column — both consistent with a
padding-layout leak, not numeric rounding (a bf16 effect would not be exactly 1.0,
nor parity-structured, nor vanish at tile-aligned sizes).

The exact kernel/lowering line was not pinned; the empirical signature (tile-aligned
⇒ correct, otherwise padding leak) localizes it to the embedding-backward tilize /
scatter handling of the padded region.

## Fix

Not implemented. The fix belongs in the embedding-backward path (tt-mlir lowering or
the tt-metal embedding_bw kernel): zero/ignore the padded tile region of the
gradient instead of letting it contribute. Until fixed, a workaround for affected
models is to pad vocab and embedding dim up to multiples of 32 (and slice the
gradient back), or avoid the on-device embedding-backward for non-aligned shapes.

## Minimal Reproducer

- [supplemental/repro_embed_grad_dupidx.py](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/supplemental/repro_embed_grad_dupidx.py)
  — `grad(sum(W[idx]))` with repeated indices, prints per-row CPU vs TT gradient and
  the spurious odd-column pattern (vocab=16, dim=4).
- [supplemental/repro_embed_grad_dimsweep.py](/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/supplemental/repro_embed_grad_dimsweep.py)
  — sweeps dim ∈ {1,2,4,8,16,32}, printing per-column mean(tt−cpu) to expose the
  odd-column / tile pattern.

Expected: correct (max 0) at tile-aligned vocab/dim (e.g. 32, 64); spurious +1.0 in
odd columns of even rows otherwise.

## Reproduction Steps

```bash
cd /home/houjun/theseus
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
python /home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/supplemental/repro_embed_grad_dupidx.py
python /home/houjun/lessons/2026-06-03-ttxla-embedding-bw-tile-padding-grad/supplemental/repro_embed_grad_dimsweep.py
```

## Verification

```text
vocab=32 dim=2  -> max|tt-cpu| = 0.00   (tile-aligned: correct)
vocab=64 dim=2  -> max|tt-cpu| = 0.00
vocab=16 dim=4  -> max|tt-cpu| = 1.00   (spurious +1 in odd cols of even rows)
vocab=33 dim=2  -> spurious only at row 32 (overflow into the padded tile)
```

The tile-aligned cases are exact; the non-aligned cases corrupt deterministically.

## Notes

- Qwen2.5-0.5B is tile-aligned (vocab 151936 = 32×4748, hidden 896 = 32×28), so its
  embedding gradients are correct despite this bug. An earlier `max|cpu-tt|=1.0`
  observation came from a small non-aligned probe, not the model.
- Duplicate-index accumulation is correct and is a *separate* property from this
  padding bug; do not conflate them.
- The spurious value equals the cotangent, so a non-`sum` loss would leak that loss's
  cotangent values into the same odd-column/even-row positions — the magnitude
  varies but the structure (tile padding) is the invariant.
