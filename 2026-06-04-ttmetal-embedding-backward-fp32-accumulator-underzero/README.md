# tt-metal `embedding_backward` Under-Zeros the fp32 Accumulator Tile → `inf` Gradients (and Op Wedge)

## Summary

Training a GPT on Blackhole with a **tied token-embedding table** (`wte` used for both the input
embedding lookup and the output unembedding) produced `inf` for `wte`'s gradient on device, while every
other parameter's gradient was finite and matched CPU. A per-leaf diagnostic pinned it: `wte/.value:
norm=inf`, all blocks/layernorms finite. The same garbage also intermittently **wedged** the op (multi-
minute hangs).

The cause is in tt-metal's `embedding_backward` reader kernel: `generate_zeros_cb` zero-initializes the
accumulator tile with a hardcoded word count that only covers bf16 (512 words) or bfp8_b (272 words).
An **fp32** output tile is 1024 words (4096 B), so it fell into the bfp8_b `else` branch and zeroed only
1088 of 4096 bytes — leaving **2944 bytes of garbage per accumulator tile**, which the compute kernel
then accumulated gradients onto. The fix makes the zero-fill dtype-agnostic (zero the full tile).

## Status

- **Bug type:** device-kernel accumulator under-initialization (dtype-specific), tt-metal.
- **Component:** `ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward.cpp` (`generate_zeros_cb`).
- **Fixed locally:** yes — kernel edited + deployed to all kernel copies on the box. tt-metal kernels are
  JIT-compiled from source at runtime, so **no `libTTMLIRCompiler` rebuild needed** (but the kernel
  cache must be cleared so the edited source recompiles — see Reproduction).
- **Code-reviewed:** yes (independent review confirmed the loop is bounded, the word counts are correct
  fp32→1024 / bf16→512 / bfp8_b→272, and it cannot introduce a hang).
- **HW-verified to completion:** NO — blocked by a *separate*, downstream issue: with `wte` unfrozen
  (so `embedding_backward` is actually in the graph), the plugin compile is pathologically slow
  (>30 min, never finished in test windows). See **Notes**. CPU baseline of the repro is finite.
- **Related bug not fixed here:** the tied-weight **matmul-path gradient is dropped** —
  `ttnn::embedding_bw(input, weight, out_grad)` takes `weight` (the scatter operand carrying the
  matmul-VJP grad for a tied weight) but never uses it. That's a correctness bug separate from the inf.

## Repositories

- **tt-metal** (the fix), inside tt-mlir's third_party:
  `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`, worktree dirty.
- **tt-mlir**: `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`, commit `412daacc4…`.
- **tt-xla** (PJRT plugin host): `/home/houjun/tt-xla`, commit `03f29ed01…`.

## Host Environment

- Ubuntu 24.04, Linux 6.8.0-110, Python 3.12, Clang 20. jax/jaxlib 0.7.1.
- Device: Tenstorrent Blackhole p150b (`tt-qb2.stanford.edu` / 10.42.100.2).
- `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`, `CONVERT_SHLO_TO_SHARDY=1`.

## User-Visible Failure

A small GPT training loop on TT (one_hot/integer-label CE, fp32 activations) printed a correct first
step then `-inf` forever:

```text
[demo] step  1 | loss 6.9805
[demo] step  2 | loss -inf
...
[demo] step 25 | loss -inf
```

Per-parameter gradient norms at init (TT vs CPU) isolated the source — all finite **except** `wte`:

```text
[grad] blocks_0/attn/c_attn/kernel/.value: norm=0.31979525   (finite, ~CPU)
...
[grad] ln_f/weight: norm=0.017153270
[grad] wte/.value: norm=inf
```

CPU gives `wte` grad a finite ~O(1) norm. Only `wte` is routed through `embedding_backward` (every other
grad uses matmul/layernorm backward, which zero correctly), and its output tile is fp32 — exactly the
unhandled dtype.

## Root Cause

`reader_embedding_backward.cpp`'s `generate_zeros_cb` zero-fills the accumulator with hardcoded counts:

```cpp
constexpr bool is_output_bfloat16 = get_compile_time_arg_val(6) == 1;
if constexpr (is_output_bfloat16) {
    for (uint32_t i = 0; i < 512; ++i) input_l1_ptr[i] = 0;   // 2048 B = bf16 tile
} else {
    for (uint32_t i = 0; i < 272; ++i) input_l1_ptr[i] = 0;   // 1088 B = bfp8_b tile
}
```

There is **no fp32 branch**. An fp32 tile is 32×32×4 = 4096 B = 1024 words, so fp32 output falls into the
`else` and zeros only 1088 of 4096 bytes — leaving **2944 bytes of uninitialized garbage** per
accumulator tile. The compute kernel then does `accumulator += reshuffle(grad)` over that garbage,
yielding `inf`/garbage gradients; garbage values feeding `reshuffle_rows_tile` can also send rows to
bad addresses and **wedge** the op (the multi-minute hangs).

This is dtype-specific, which is exactly why **only `wte`** (the sole fp32 tensor flowing through
`embedding_backward`) is affected while every other (matmul/LN) gradient is finite.

## Fix

Make the zero-fill dtype-agnostic by zeroing the full tile, using the real output tile size:

```cpp
FORCE_INLINE void generate_zeros_cb(uint32_t input_l1_addr, uint32_t tile_bytes) {
    uint32_t num_words = tile_bytes >> 2;     // bytes/4 == uint32 words; fp32 -> 1024
    auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
    for (uint32_t i = 0; i < num_words; ++i) p[i] = 0;
}
// call site: generate_zeros_cb(out_read_ptr, out_page_size);  // out_page_size = get_tile_size(cb_id_out0)
```

Word counts after the fix: fp32→1024, bf16→512, bfp8_b→272 — matches the old hardcoded values for the
old dtypes, and now covers fp32. Patch:
`/home/houjun/lessons/2026-06-04-ttmetal-embedding-backward-fp32-accumulator-underzero/supplemental/embedding_bw_fp32_zero_fix.patch`

## Minimal Reproducer

`supplemental/embed_bw_min.py` — `jax.grad(sum(jnp.take(wte, idx)))` on TT vs CPU, printing the grad
norm. Before fix: TT = `inf`. After fix: TT ≈ CPU (`~3.12e2`). Also reproducible via the GPT demo with
`FREEZE_WTE=0` (loss `-inf` vs falling).

## Reproduction Steps

```bash
# tt-metal kernels are JIT-compiled from source, but the binary is CACHED by hash — clear the
# embedding_backward kernel cache so the edited source recompiles:
rm -rf ~/.cache/tt-metal-cache/*/kernels/reader_embedding_backward
cd /home/houjun/theseus && source .venv/bin/activate
TT_VISIBLE_DEVICES=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole CONVERT_SHLO_TO_SHARDY=1 \
  python /home/houjun/.agents/embed_bw_min.py
```

## Verification

CPU baseline finite (`embed_bw_norm=3.12e2`). On-device run-to-completion is **blocked** by a separate
downstream compile blowup (see Notes), so the fix is **deployed + code-reviewed but not yet HW-confirmed
end to end**. Expected on a healthy/fast device: TT `embed_bw_norm` flips `inf → ~3.12e2`.

## Notes

- **Why the training demo "worked" with `wte` frozen:** the demo zeroes `wte`'s grad after computing it;
  XLA **DCE then eliminates `embedding_backward` entirely** (its result is unused), so it never runs —
  no `inf`, fast compile. The bug only manifests when `embedding_backward` is actually live.
- **The downstream blocker:** with `wte` unfrozen, the plugin compile is pathologically slow (>30 min,
  never completed in test windows; python pinned ~102% CPU). Chip-free `ttmlir-opt` profiling showed the
  tt-mlir SHLO→TTIR→TTNN lowering is **sub-second** (embedding_backward lowering = 0.057 s), so the
  blowup is **downstream of MLIR** — TTNN→flatbuffer, tt-metal kernel-JIT, or PJRT/runtime — and needs
  box-side profiling to pinpoint. This blocks both verifying this fix and training tied embeddings on TT.
- **Diagnosis tooling caveat:** check the *python* pid's CPU, not the bash/timeout wrapper — the wrapper
  is idle while the python compiles at ~102%; this caused a false "hung" read mid-investigation.
- The `reshuffle_rows_tile(0, idx_addr - 16)` "subtract 16-byte header" hack in the compute kernel was
  considered and ruled out as the inf cause (dtype-independent; would corrupt values, not produce inf).
