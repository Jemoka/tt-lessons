# TT Matmul Already Runs at TF32 Input Precision on Blackhole — the Residual Is the TF32 Ceiling, Not bf16

## Summary

The remaining Qwen-parity residual on Tenstorrent Blackhole was attributed to the
matmul FPU truncating fp32 operands to **bf16** (7 mantissa bits) at unpack. Direct
measurement shows that is not what the patched build does: a matmul's effective
input precision is **~9.96 mantissa bits at K=1 — i.e. TF32 (10 bits), not bf16**.
The compiled TTNN IR confirms why: the matmul carries all-fp32 operands/output and
`fp32_dest_acc_en=true`, which is exactly the condition tt-metal uses to auto-select
the TF32 unpack path for SrcA/SrcB.

Consequence: there is no "bf16 → tf32" fix left to make — it is already tf32, the
matrix engine's documented maximum. The per-matmul ~1e-3 relative error (tf32),
compounded across 24 layers, is the source of the ~0.93 plain-path residual. The
only way to go beyond it on the matrix engine is a multi-pass scheme such as a
3×bf16 (hi/lo) split (~14-16 effective bits), implemented as a compiler pass.

## Status

- Finding type: measurement + IR confirmation; corrects a prior "bf16-input" framing.
- Component: tt-metal matmul unpack (Blackhole) as driven by the tt-xla/tt-mlir
  compiled graph; no bug — tf32 is the intended HW ceiling.
- Actionable: the patched build already achieves tf32; the next precision lever is
  a 3×bf16 matmul-decomposition compiler pass (feasibility scouted separately:
  target `ttir.MatmulOp`, template `DecomposeMinReduction.cpp`, insert in
  `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` after the TTIR decomposition pass,
  and set `conservative_folding=true` on the generated typecasts so the canonicalizer
  does not fold the f32→bf16→f32 split back to identity).

## Repositories

- `tt-xla` — `/home/houjun/tt-xla`, branch `main`,
  `03f29ed01a2bca27f5d8eaace659534016c7d0c4` (worktree dirty, plus the primary's
  matmul compute-config patch deployed to tt-qb2).
- `tt-mlir` (submodule) — `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir`,
  `412daacc440f10bb98ccc685c311b01f1fadab70`.
- `tt-metal` (nested) — under `tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`.
- `theseus` — `/home/houjun/theseus`, `feat/tenstorrent`,
  `f085ca67fa68ef08d63668cd7f866b2b8147839e` (probes are theseus-free).

## Host Environment

`tt-qb-ac-02` (tt-qb2.stanford.edu), Linux 5.15.0-179 x86_64, 4× Blackhole p150b,
jax/jaxlib 0.7.1, ARCH_NAME=blackhole. Probe pinned to chip 1
(`TT_VISIBLE_DEVICES=1`), patched plugin. Startup `TT_FATAL ... remote mmio` lines
are benign single-chip warnings.

## User-Visible Failure

Plain-path (slow-safe OFF) qwen_parity on the patched plugin stays at max diff
~0.93, top5 3 — read by the team as "bf16-input matmul." The question this lesson
answers: is the matmul input really bf16, or already tf32?

## Root Cause

The matmul is already tf32; the residual is the tf32 ceiling, not a fixable bf16
truncation.

1. Effective mantissa bits of matmul inputs (CPU float64 reference vs TT, small K
   to isolate input rounding from accumulation):

```text
   K   rel_err(median)  eff_bits   (bf16~7-8, tf32~10, fp32~23)
   1     1.005e-03         9.96
   2     1.051e-03         9.89
   4     1.025e-03         9.93
   8     1.121e-03         9.80
  16     1.130e-03         9.79
  64     1.323e-03         9.56
```

   9.96 bits at K=1 is tf32. bf16 would be ~7-8 bits (rel ~4e-3 to 8e-3).

2. Compiled TTNN IR for a plain f32 matmul (`TTXLA_LOGGER_LEVEL=DEBUG`):

```text
%2 = "ttnn.matmul"(%0, %1) <{compute_config = #ttnn.device_compute_kernel_config<
       math_fidelity = hifi4, fp32_dest_acc_en = true, packer_l1_acc = true>,
       transpose_a = false, transpose_b = false}>
   : (tensor<32x896xf32, #ttnn_layout2>, tensor<896x896xf32, #ttnn_layout3>)
     -> tensor<32x896xf32, #ttnn_layout2>
```

   All operands and the output are `f32`, with `fp32_dest_acc_en = true`. Per
   tt-metal `tt_metal/jit_build/genfiles.cpp:285-289`, `fp32_dest_acc_en && is_all_fp32_formats`
   sets `unpack_conditional_dst_format = DataFormat::Tf32` — i.e. fp32 operands are
   converted to tf32 (not bf16) into SrcA/SrcB. The measurement (1) is that path
   working.

3. tf32 is the documented matrix-engine maximum
   (`docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:168`:
   "the matrix engine's maximum accuracy is TF32 (19 active bits) ... less than full
   32-bit precision"). So no compute-config change can exceed tf32 on the FPU.

## Fix

None for the FPU — tf32 is the hardware ceiling and is already in effect. To get
below ~1e-3 per matmul, decompose each matmul on the matrix engine:

- 3×bf16 (hi/lo) split: `A_hi=bf16(A), A_lo=A-A_hi, B_hi=bf16(B), B_lo=B-B_hi`,
  `result ≈ A_hi@B_hi + A_hi@B_lo + A_lo@B_hi` (~14-16 effective bits, rel ~1e-4).
  Implement as a tt-mlir TTIR pass over `ttir.MatmulOp`; the sub-matmuls run at bf16
  unpack (operands become bf16-typed, so the tf32 path does not fire on them — by
  design; the split is what recovers the bits).

Trade-off: 3× the matmul work for ~10× lower error vs tf32. The model-side chunked
"slow-safe" path is the alternative lever (bounds per-matmul K).

## Minimal Reproducer

- [supplemental/repro_matmul_bits.py](/home/houjun/lessons/2026-06-03-tt-matmul-already-tf32/supplemental/repro_matmul_bits.py)
  — measures effective mantissa bits of TT matmul inputs vs a float64 reference,
  across K. Expected on the patched build: ~9.8-10 bits (tf32).
- [supplemental/dump_matmul_ir.py](/home/houjun/lessons/2026-06-03-tt-matmul-already-tf32/supplemental/dump_matmul_ir.py)
  — compiles a plain f32 matmul with `TTXLA_LOGGER_LEVEL=DEBUG` and dumps the TTNN
  IR; grep for `ttnn.matmul` to read operand/output DataFormat + compute_config.

## Reproduction Steps

```bash
cd /home/houjun/theseus
source .venv/bin/activate
export TT_VISIBLE_DEVICES=1 CONVERT_SHLO_TO_SHARDY=1 JAX_PLATFORMS=tt,cpu ARCH_NAME=blackhole
python /home/houjun/lessons/2026-06-03-tt-matmul-already-tf32/supplemental/repro_matmul_bits.py
# IR (operand DataFormat + compute_config):
python /home/houjun/lessons/2026-06-03-tt-matmul-already-tf32/supplemental/dump_matmul_ir.py 2>dump.stderr
grep -n 'ttnn.matmul' dump.stderr
```

## Verification

```text
matmul input precision (patched build, chip 1):
  K=1  -> 9.96 effective mantissa bits (tf32), rel err 1.0e-3
IR: ttnn.matmul operands+output = f32, compute_config fp32_dest_acc_en=true
```

Together: the tf32 unpack path is active; bf16 is not the input format. A bf16-input
build would show ~7-8 bits.

## Notes

- This corrects the earlier "matmul truncates fp32 → bf16 at unpack" framing: the
  bf16-conditional in `data_format.cpp` is overridden to tf32 when all CBs are fp32
  and `fp32_dest_acc_en` is set, which the patched build satisfies. The unpacker
  dive was right about the mechanism (fidelity phases read the SrcReg; SrcReg width
  caps precision) but the realized format here is tf32, not bf16.
- K=64 dropping to 9.56 bits is mild accumulation droop on top of the tf32 input
  rounding; the input-precision signal is the small-K rows.
- Probe data are single-run medians; the conclusion (tf32 vs bf16) is a ~3-bit gap,
  far larger than run-to-run noise.
