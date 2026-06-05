# tt-metal `embedding_backward` Drops the Last Partial Vocab Tile for Non-32-Aligned Vocab → `inf` Gradients

## Summary

Training a Theseus GPT on Blackhole with a **non-tile-aligned vocab** (`V=1000`) produced `inf`
for the tied token-embedding (`wte`) gradient on device, while every other parameter was finite and
matched CPU. The same model with `V=1024` (a multiple of 32) trained cleanly. Isolated to a minimal
`jax.grad(sum(take(wte, idx) * C))` — embedding-lookup VJP only, no LayerNorm/CE/matmul — the inf
reproduces with a **non-uniform cotangent** `C` at `V=1000` (and is masked by an all-ones cotangent,
which happens to stay finite).

The cause is integer truncation in the embedding-backward program factory:
`num_embeddings_tiles = num_embeddings / TILE_HEIGHT` floors `1000/32` to `31`, but the output tensor
is tile-padded to `1024` (32 tiles). The last partial vocab tile (tile 31, rows 992–1023, holding the
**valid** logical rows 992–999) is then left out of both the reader's zero-init loop and the compute
accumulation, so it keeps uninitialized DRAM garbage → `inf`. The fix is one line: `tt::div_up` (ceil)
instead of floor.

## Status

- **Bug type:** backend correctness — embedding-backward (gather-VJP scatter) drops the last partial
  vocab tile when vocab is not a multiple of 32 (`TILE_HEIGHT`).
- **Component:** tt-metal `embedding_backward_program_factory.cpp` (`num_embeddings_tiles`).
- **Fixed locally:** **yes** — `div_up` (ceil); `_ttnncpp.so` rebuilt + deployed to
  `install/lib/_ttnncpp.so` on the run box (tt-qb-ac-02). HW-verified.
- **HW-verified:** **yes.** Post-fix `V=1000` random-cotangent embed-bw grad flips `inf → 2.56e2`
  (==CPU); the full unfrozen GPT demo (`FREEZE_WTE=0`) trains `wte` on-device with falling loss
  (6.97 → 2.08 over 25 steps) and finite gnorm.
- **Corrects a prior lesson.** `2026-06-03-ttxla-embedding-bw-tile-padding-grad` characterized the
  same op's non-aligned behavior but tested **only the all-ones (`sum`) cotangent**, saw a small
  finite `+1.0` "leak", and concluded the trigger needs **compile-time-constant indices** with
  "blast radius ≈ none for dynamic-index training". That conclusion is **wrong for the inf**: with a
  real **non-uniform cotangent** (every real training step) the failure is `inf`, and it occurs with
  **dynamic (runtime-arg) indices** too. So **any** training with `vocab % 32 != 0` infs the embedding
  gradient on TT — not a constant-index-only edge case. (Qwen2.5-0.5B vocab `151936 = 32×4748` is
  aligned, so `qwen_parity` itself is unaffected; this is a general training-correctness bug.)
- **Distinct from** the fp32 `generate_zeros_cb` under-zero bug
  ([2026-06-04-ttmetal-embedding-backward-fp32-accumulator-underzero](/home/houjun/lessons/2026-06-04-ttmetal-embedding-backward-fp32-accumulator-underzero/README.md)):
  that zeroes too few bytes *within* a tile (dtype-specific); this drops a whole tile (vocab-alignment-
  specific). Both are needed for fp32 non-aligned-vocab embedding training.

## Repositories

- **tt-metal** (the fix), inside tt-mlir's third_party on the run box (tt-qb-ac-02):
  `…/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`, HEAD
  `90c914ef258b5cc92ad172f3604b784ec77253ca`, worktree dirty (this one-line fix + the prior
  `reader_embedding_backward.cpp` fp32 fix).
- **tt-mlir**: `…/src/tt-mlir`, `412daacc4…`.
- **tt-xla** (PJRT plugin host): `/home/houjun/tt-xla`, `03f29ed01…`.

## Host Environment

- Two boxes, separate `/home` (NOT shared; only `/nfs/tt2` is NFS): build/edit on `tt-qb-ac-01`,
  chip runs on `tt-qb-ac-02` (`ssh houjun@10.42.100.2`). The fix must be built+deployed on ac-02.
- Ubuntu 24.04, Linux 6.8.0-110, Python 3.12, Clang 20, jax/jaxlib 0.7.1.
- Device: Tenstorrent Blackhole p150b. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`,
  `CONVERT_SHLO_TO_SHARDY=1`, `TT_VISIBLE_DEVICES=1`.

## User-Visible Failure

A small GPT (`V=1000`) with the tied `wte` *unfrozen* gave `wte/.value: norm=inf` on TT (all other
leaves finite, ==CPU), and the training loss went `-inf` after step 1. The same model at `V=1024`
trained finitely. Minimal isolation (`supplemental/embed_bw_cot2.py`, dynamic indices):

```text
[cpu] V=1000 ones-cotangent=3.1231e+02  random-cotangent=2.5633e+02
[tt]  V=1000 ones-cotangent=3.1394e+02  random-cotangent=inf      <- non-uniform cotangent infs
[tt]  V=1024 random-cotangent=2.5605e+02                           <- aligned vocab: finite==CPU
```

## Root Cause

`embedding_backward_program_factory.cpp`:

```cpp
uint32_t num_embeddings_tiles = operation_attributes.num_embeddings / TILE_HEIGHT;
```

`num_embeddings` is the **logical** vocab (`weight.logical_shape()[-2]`, e.g. 1000). The output tensor
is created with that logical shape and TILE layout, so DRAM holds `ceil(1000/32) = 32` tiles (padded to
1024 rows). But `1000 / 32` truncates to **31**. The reader kernel uses this count both to zero-init the
output (`for i < num_embeddings` over the per-core tiles) and the compute kernel uses it for the
accumulation sweep. Tile 31 (rows 992–1023) — which contains the **valid** logical rows 992–999 — is
therefore never zeroed and never written, so it keeps whatever uninitialized garbage was in that DRAM
region. The host reads back logical rows 0–999, picking up the garbage in 992–999 → `inf`/NaN.

Why all-ones cotangent hid it: with a `sum` loss the cotangent is constant `1.0`; the leaked/garbage
contribution stayed finite (the prior lesson saw a structured `+1.0`). A real non-uniform cotangent
exposes the uninitialized region as `inf`. The bug needs only `vocab % 32 != 0`; indices can be dynamic.

## Fix

Use ceil so the partial last vocab tile is covered (its padding rows 1000–1023 are never indexed —
`idx < num_embeddings` — so they remain zero after the zero-init):

```cpp
uint32_t num_embeddings_tiles = tt::div_up(operation_attributes.num_embeddings, TILE_HEIGHT);
```

One file: `ttnn/cpp/ttnn/operations/embedding_backward/device/embedding_backward_program_factory.cpp`
(`tt::div_up` from the already-included `ttnn/operations/math.hpp`). This is **host** code compiled into
`_ttnncpp.so` (not a JIT kernel), so it requires a tt-metal library rebuild + relink, unlike the
reader/compute kernel edits. Patch:
`/home/houjun/lessons/2026-06-04-ttmetal-embedding-bw-nonaligned-vocab-droptile-inf/supplemental/embedding_bw_num_embeddings_div_up.patch`

## Minimal Reproducer

`supplemental/embed_bw_cot2.py` — `jax.grad(sum(take(wte, idx) * C))` with dynamic `idx`, comparing an
all-ones cotangent vs a random cotangent at `V=1000`. Pure JAX; no Theseus.

1. Builds `wte [V,128]`, dynamic `idx [8,64]`, two cotangents (ones, random).
2. Computes the embedding-VJP grad norm on CPU and TT.
3. Before fix: TT random-cotangent = `inf` at `V=1000` (ones finite). After fix: `≈2.56e2` == CPU.

## Reproduction Steps

```bash
# On the run box (tt-qb-ac-02):
cd /home/houjun/theseus && source .venv/bin/activate
TT_VISIBLE_DEVICES=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole CONVERT_SHLO_TO_SHARDY=1 \
  python -u /home/houjun/.agents/embed_bw_cot2.py

# Rebuild after editing the program factory (host code -> _ttnncpp.so):
cd …/third_party/tt-metal/src/tt-metal/build_Release && ninja _ttnncpp.so
cp ttnn/_ttnncpp.so ../../../../../install/lib/_ttnncpp.so   # the plugin's RUNPATH target
```

## Verification

```text
# embed_bw_cot2.py, V=1000, dynamic idx
                 before-fix            after-fix
[tt] ones        3.14e2 (finite)       3.1231e2 (==CPU)
[tt] random      inf                   2.5609e2 (==CPU 2.5633e2)

# full unfrozen GPT demo (FREEZE_WTE=0, V=1000) on TT, after fix:
[demo] step  1 | loss 6.9658 | gnorm 2.442
[demo] step 25 | loss 2.0813 | gnorm 1.178   <- wte trains on-device, finite, falling loss
```

## Notes

- Only the partial-last-tile case is affected; vocab that is already a multiple of 32 (e.g. 1024, or
  Qwen2.5's 151936) was always correct. Embedding dim is separately required to be a multiple of
  `TILE_WIDTH` (asserted in the device op).
- Two boxes with **separate `/home`**: a fix edited on ac-01 is invisible to ac-02. Apply + rebuild on
  the box that actually runs the chip (ac-02 here). The plugin loads `_ttnncpp.so` via RUNPATH from
  `…/install/lib/`, confirmed with `ldd pjrt_plugin_tt.so | grep ttnncpp`.
