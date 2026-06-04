# TT-XLA `embedding_bw` Returns a Rank-4 Tensor, Breaking a Downstream Rank-2 Reduce

## Summary

On-device training of a GPT model on Tenstorrent (Blackhole) aborted in the optimizer step with a tt-metal `new_volume == old_volume` reshape assert (and intermittent SIGSEGV). The trigger was `optax.clip_by_global_norm`: its `global_norm` sum-of-squares reduction over the word-embedding gradient `[100288, 256]` produced a "scalar" that at runtime still held the full `[100288, 256]` buffer. When the scalar was then used in a broadcasting binary op, tt-metal tried to reshape the big tensor to `[1]` and aborted.

Root cause: tt-metal's `ttnn::embedding_bw` returns a **rank-4** tensor (e.g. `[1,1,V,E]`), but the compiler declares the result as **rank-2** `[V,E]`. A downstream `ttnn.sum` lowered with `dim_arg=[0,1]` (correct for the rank-2 IR type) then reduces only the two leading unit dims of the rank-4 runtime tensor, leaving `[V,E]` unreduced. The fix reshapes the `embedding_bw` runtime output back to its IR-declared logical shape so its rank matches the IR.

## Status

- **Bug type:** runtime tensor/IR rank mismatch (silent wrong-result, then crash).
- **Component:** tt-mlir TTNN runtime, `ttnn::embedding_bw` op wrapper.
- **Fixed locally:** yes — `runtime/lib/ttnn/operations/embedding/embedding_backward.cpp` reshapes the output to the IR rank. Verified on hardware.
- **Not resolved by this fix:** the full theseus `gpt/train/pretrain` step now advances past the clip and global-norm and fails at the *next* gap — an eager `jax.lax.dynamic_slice` (`dynamic_slice_p`, JAX `slicing.py:1474`) → `XlaRuntimeError: INTERNAL: Error code 13`. That is a separate bug.
- **Inference unaffected:** `qwen_parity.py` does not exercise `embedding_bw`; output is byte-identical before/after (max diff 0.4292, top5 5).

## Repositories

- `/home/houjun/tt-xla` — PJRT plugin. Worktree dirty. The plugin dynamically links `third_party/tt-mlir/install/lib/libTTMLIRRuntime.so` (verified via `ldd`), so a runtime-only rebuild is picked up without rebuilding the plugin.
- `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir` — tt-mlir compiler + runtime. `TTMLIR_GIT_HASH=412daacc440f10bb98ccc685c311b01f1fadab70`. Worktree dirty (this fix + unrelated landed fixes).
- `/home/houjun/theseus` — training harness. The trainer config used: `configs/scratch/synthetic_pretrain.yaml`, `job=gpt/train/pretrain`. Not modified for this fix.

## Host Environment

- Box: `tt-qb2.stanford.edu` (`tt-qb-ac-02`), 4× Blackhole p150b.
- Linux x86-64; clang-20 toolchain at `/opt/ttmlir-toolchain`.
- jax/jaxlib 0.7.1, optax, flax. `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`.

## User-Visible Failure

Running the trainer (`gpt/train/pretrain`, clip present) aborted in the optimizer step:

```text
TT_FATAL: Invalid arguments to reshape (assert.hpp:104)
{TT_FATAL @ .../ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50: new_volume == old_volume
 --- ttnn::operations::data_movement::detail::infer_dims_for_reshape(...)
jaxlib._jax.XlaRuntimeError: INTERNAL: Error code: 13
```

Per-op runtime tracing showed the last ops before the abort were `clip_by_global_norm`'s `t / g_norm` div / `global_norm` adds, with a binary operand whose IR shape was scalar but whose runtime tensor had volume 25,673,728 (= `[100288,256]`):

```text
BINOP type=0(Add) lhs_rt_vol=1 rhs_rt_vol=25673728 lhs_ir_vol=1 rhs_ir_vol=1 ...
```

Tracing the reduction op pinned the producer — a full reduce that did not reduce:

```text
REDUCE in_vol=25673728 out_vol=25673728 out_ir_vol=1 keep_dim=0 dimArg=[0,1,] in_rank=4   # in-context: WRONG
REDUCE in_vol=25673728 out_vol=1        out_ir_vol=1 keep_dim=0 dimArg=[0,1,] in_rank=2   # clean host tensor: OK
```

Same op, same `dim_arg=[0,1]`, same TILE layout/dtype — the only difference is the input **rank** (4 vs 2).

## Root Cause

In MLIR the relevant subgraph is correct and rank-2:

```text
%17 = "ttnn.embedding_bw"(...) : (...) -> tensor<100288x256xbf16, ...>
%20 = "ttnn.sum"(%19) <{dim_arg = [0,1], keep_dim = false}>
        : (tensor<100288x256xf32, ...>) -> tensor<f32, ...>
```

But at runtime `ttnn::embedding_bw` (tt-metal) returns a **rank-4** tensor (`[1,1,100288,256]`). Nothing reshapes it back to the rank-2 shape the compiler declared. The runtime tensor then flows (square, reduce) keeping rank 4. `ttnn::sum` is called with `dim_arg=[0,1]` (the indices the compiler computed for the rank-2 logical type); on the rank-4 tensor that reduces the two leading **unit** dims and returns `[100288,256]` unchanged. That full tensor is inserted into the reduce's scalar-typed output slot, so `global_norm` and the per-leaf clip divide operate on a full `[100288,256]` tensor masquerading as a scalar. The first broadcasting binary op then triggers a reshape of the big tensor to `[1]` and aborts.

The `FlatbufferObjectCache` / serializer were ruled out directly: 180 serialized `ttnn.ReshapeOp`s, zero with a mis-shaped input ref; the failing `ttnn::reshape` is invoked *internally* by a binary op's broadcast, not by a serialized `ReshapeOp`.

## Fix

`runtime/lib/ttnn/operations/embedding/embedding_backward.cpp`: after `::ttnn::embedding_bw(...)`, if the output tensor's rank differs from the IR-declared result rank (`op->out()->desc()->shape()`), reshape it to the IR-declared logical shape before inserting into the tensor pool. One file. Patch: `/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/supplemental/embedding_bw_rank_fix.patch`.

## Minimal Reproducer

`/home/houjun/lessons/2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/supplemental/repro_reduce_embgrad.py` (pure JAX, theseus-free):

1. `emb = ones([100288,256])`; `idx = arange(32)`.
2. `fwd(e) = sum(e[idx]**2)` — gather (embedding fwd) then scalar loss.
3. `global_norm_of_grad(e) = sqrt(sum(grad**2))` where `grad = jax.grad(fwd)(e)` is `[100288,256]` produced via scatter / `embedding_bw`.
4. **Before fix:** the `jnp.sum(grad*grad)` returns the unreduced tensor → reshape abort / segfault.
5. **After fix:** returns a true scalar (`shape=()`), value ≈ 181.0.

`supplemental/repro_reduce_to_scalar.py` is a control: the identical all-dims reduce over a *fresh host* `[100288,256]` tensor reduces correctly both before and after — isolating the trigger to the embedding-bw gradient's rank, not the reduce itself.

## Reproduction Steps

```bash
ssh houjun@tt-qb2.stanford.edu
cd /home/houjun/theseus && source .venv/bin/activate
# reduced repro (fails before fix, passes after):
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python /home/houjun/lessons/2026-06-03-ttxla-embedding-bw-reduce-rank-mismatch/supplemental/repro_reduce_embgrad.py
# full trainer (clip present):
ARCH_NAME=blackhole JAX_PLATFORMS=tt,cpu TT_VISIBLE_DEVICES=0 \
  python -u -m theseus.cli run gpt/train/pretrain \
  configs/scratch/synthetic_pretrain.yaml ~/theseus training.tokens=8000
```

Surgical rebuild of the runtime only (no superbuild): recompile `embedding_backward.cpp` with the command from `build/compile_commands.json`, `ar r` the object into `build/runtime/lib/ttnn/operations/libTTRuntimeTTNNOps.a`, relink `libTTMLIRRuntime.so`, and copy it to `third_party/tt-mlir/install/lib/`.

## Verification

Reduced repro, before vs after:

```text
BEFORE: REDUCE in_vol=25673728 out_vol=25673728 ... in_rank=4   -> [FAIL]/segfault
AFTER : REDUCE in_vol=25673728 out_vol=1        ... in_rank=2   -> [PASS] norm=181.0193
```

Full trainer, before vs after (clip present, clean runtime — no instrumentation):

```text
BEFORE: TT_FATAL reshape_common.cpp:50 new_volume == old_volume   (in optimizer step)
AFTER : 0 reshape FATALs; advances past clip/global-norm to the next gap:
        jax.lax.dynamic_slice -> XlaRuntimeError INTERNAL Error code 13   (separate bug)
```

Inference regression check — `qwen_parity.py` default invocation, byte-identical:

```text
max diff: 0.4292325973510742   top5 overlap: 5
hf loss: 5.514760494232178     jax loss: 5.533915201822917
```

## Notes

- The reduce lowering itself is correct for the IR (`dim_arg` matches the rank-2 type); the bug is purely the runtime tensor's rank not matching the IR. Fixing it at the `embedding_bw` output (where the rank-4 originates) is the most localized correct fix; a more general guard would assert runtime-rank == IR-rank at tensor-pool insertion.
- Diagnosis used temporary flushed `fprintf` probes in `program_executor.cpp` (per-op trace), `binary.cpp` (operand volumes), `types.cpp` (pool insert volume mismatch), and `reduction.cpp` (reduce in/out vol, dims, rank). All were reverted; the shipped diff is only the `embedding_backward.cpp` reshape.
- This fix is for the *training* path; `embedding_bw` is backward-only, so inference (qwen parity) is unaffected.
- The next blocker (eager `dynamic_slice`, Error 13) is the frontier for continued trainer bringup.
