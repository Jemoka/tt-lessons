# An Apparent tt-xla "Compile Blowup" Was Cold-Cache tt-metal Kernel JIT (How to Tell)

## Summary

A standalone `embedding_backward` gradient (`jax.grad(sum(jnp.take(wte, idx)))`) on Blackhole appeared to
have a pathological **>30–50 minute compile** — the Python process pinned one core at ~102% with the device
idle and no progress logged. It looked like a tt-mlir pass blowup, and earlier guesses (tile-padding,
Shardy, a multithreading bug) were all chased. They were all wrong.

Phase-by-phase, mostly-chip-free timing exonerated the entire compiler and located the cost in **tt-metal
cold-cache kernel JIT** — compiling `embedding_backward`'s kernels (especially the heavy
`reshuffle_rows_tile` SFPU compute kernel) C++→RISC-V on a cache miss. The "blowup" was **one-time and
self-inflicted**: the kernel cache was being cleared before every run (to pick up an edited kernel), so the
expensive cold JIT was paid every time, amplified by a degraded box. No tt-mlir/tt-xla source pathology.

## Status

- **Type:** misdiagnosis postmortem + reusable methodology. NOT a source bug.
- **Component:** tt-metal kernel JIT (`embedding_backward` reader/compute kernels); diagnosis spans tt-mlir
  pipeline, tt-xla PJRT `module_builder`, and the tt-metal runtime.
- **Resolution:** the slowness is cold-cache kernel JIT × box degradation — amortized by a warm cache or a
  healthy/rebooted box. Three independent agents concurred.
- **Related (distinct, real bugs in the same op):** the fp32 accumulator under-zero `inf`
  ([2026-06-04-ttmetal-embedding-backward-fp32-accumulator-underzero](/home/houjun/lessons/2026-06-04-ttmetal-embedding-backward-fp32-accumulator-underzero/README.md))
  and the tied-weight matmul-grad-drop (`ttnn::embedding_bw` ignores its `weight` operand). Those are real;
  *this* "blowup" is not.

## Host Environment

- Ubuntu 24.04, Python 3.12, jax/jaxlib 0.7.1; Tenstorrent Blackhole p150b.
- `ARCH_NAME=blackhole`, `JAX_PLATFORMS=tt,cpu`, `CONVERT_SHLO_TO_SHARDY=1`.
- Compiler: `tt-mlir` build at `/home/houjun/tt-xla/third_party/tt-mlir/src/tt-mlir/build` (`ttmlir-opt`,
  `ttmlir-translate`). Plugin compile entry: `tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc`.

## User-Visible Failure

`embed_bw_min.py` (and the GPT demo with `wte` unfrozen) never finished: CPU result printed, then the TT
path hung — process at ~102% CPU (one core), device pids empty, no `[tt]` result, repeated timeouts at
500s / 900s / 1500s / 3000s. The frozen GPT demo compiled in minutes; a *tiny* standalone
`embedding_backward` grad did not finish in 50 min — "smaller op slower than the whole model," the tell
that it wasn't graph size.

## Root Cause

Timing each phase in isolation (see Methodology) showed every compiler stage is fast and the cost is
elsewhere:

| Phase | Time | How measured |
|---|---|---|
| StableHLO→TTIR | 0.035 s | `ttmlir-opt --stablehlo-to-ttir-pipeline -mlir-timing` |
| TTIR→TTNN (`pm.run`) | 2.3 s | plugin IR-printing (`TTXLA_LOGGER_LEVEL=VERBOSE`); module printed at t=2.277s |
| TTNN→flatbuffer | 0.36 s | `ttmlir-translate --ttnn-to-flatbuffer` on the *real* extracted module |
| runtime invoke + program factory | trivial | source read (no host loop / config search) |
| **kernel JIT (cache miss)** | **>30–50 min** | residual: the silent 102%-CPU, device-idle phase |

So the silent multi-minute phase is tt-metal **JIT-compiling the kernels** (the `reshuffle_rows_tile` SFPU
compute kernel is heavy to compile) on a **cache miss**. It was paid every run because the kernel cache
(`~/.cache/tt-metal-cache/*/kernels/reader_embedding_backward`) was deliberately cleared each run to pick up
an edited kernel; the frozen demo reuses warm kernels (or XLA DCE removes `embedding_backward` entirely, so
it never builds). A ~4–5× box degradation (the frozen demo itself went from 2–3 min to >11 min over a long
session) pushed the cold JIT past every timeout.

## Methodology (the reusable part)

To decide whether a TT "compile" hang is MLIR lowering, flatbuffer, program-build, or kernel JIT — mostly
without a chip:

1. **MLIR lowering**: extract the StableHLO (`jax.jit(fn).lower(...).as_text()` on CPU) and run
   `ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=<sysdesc>"
   -mlir-timing`. Get a valid `system_desc` by copying the plugin's (`/tmp/tt_pjrt_system_descriptor`) — a
   stale/wrong one segfaults in `SystemDescAttr::getFromBuffer`. Hand-legalize stray `chlo` ops
   (`chlo.square` → `stablehlo.multiply x,x`) if the standalone pipeline rejects them.
2. **flatbuffer emit**: `ttmlir-translate --ttnn-to-flatbuffer <ttnn.mlir>`. To get the *real* module, set
   `TTXLA_LOGGER_LEVEL=VERBOSE` (triggers `ModuleBuilder::enableVerboseIRPrinting` →
   `pm.enableIRPrinting()`), and extract the printed `MLIR Module ttnn:` block from stderr.
3. **In-plugin pass progression**: `TTXLA_LOGGER_LEVEL=VERBOSE` prints "IR Dump Before/After <Pass>" per
   pass; the last "Before" with no "After" is a stuck pass. **Confound:** `enableVerboseIRPrinting` also
   calls `pm.getContext()->disableMultithreading()` — so VERBOSE measures MT-*off*; don't attribute a
   speedup to "disabling MT" without an MT-on control.
4. **kernel JIT vs program-build**: `TT_METAL_LOGGER_LEVEL=INFO` logs `BuildKernels`. Stuck with **zero**
   `BuildKernels` and device idle + host-CPU-bound ⇒ program-build or pre-JIT; `BuildKernels` streaming ⇒
   kernel JIT. Check cache: a warm `~/.cache/tt-metal-cache` (don't clear it) should make the phase vanish —
   if it does, it was cold JIT.

Tell-tale fingerprint of cold kernel JIT: host one-core ~100% CPU, device pids empty, no IR/pass output,
and it disappears with a warm cache.

## Verification

The phase timings above (compiler total <3s) plus the residual >30-min host-CPU-bound, device-idle,
`BuildKernels`-pending phase, plus the fact that a single cold JIT of the edited kernel timed out even at
3000s on the degraded box while the frozen (warm/ DCE'd) demo ran in minutes. Independently corroborated by
two other agents who saw the identical stall when they cleared `TT_METAL_CACHE` per run.

## Notes

- Don't mistake "tiny op compiles slower than the full model" for a compiler bug — under cold-cache JIT it
  just means the tiny op's *kernels* aren't cached while the model's are (or are DCE'd).
- The MT-disable confound (point 3) led to a premature "multithreading bug" claim that was retracted.
- Box degradation over a long session is real and silently inflates every timing — recheck against a known
  baseline (here: the frozen demo's own compile time) before concluding "pathological."
