# Lessons

This directory stores postmortem-style writeups of bugs, debugging sessions, and one-off investigations. Each lesson is self-contained: enough context for a future reader to understand what broke, why, and how to reproduce it without having to ask.

Commit as you go for each lesson into git@github.com:Jemoka/tt-lessons.git.

## Directory layout

One directory per lesson, named:

```
YYYY-MM-DD-short-kebab-slug
```

- date is when the lesson was written (not when the bug occurred)
- slug is short, lowercase, hyphen-separated, and names the bug or symptom (e.g. `ttxla-non-row-major-host-upload`)

Inside each lesson directory:

- `README.md` — the writeup (required)
- `supplemental/` — patches, repro scripts, log dumps, screenshots (optional, but use it whenever the README references an artifact)

Reference supplemental files from the README using absolute paths (`/home/houjun/lessons/<lesson-dir>/supplemental/<file>`), not relative ones — readers often open the file directly without `cd`-ing.

## README.md structure

Use these H2 sections in this order. Omit a section only if it truly does not apply; do not invent new top-level sections without reason.

1. **Title** (H1) — one descriptive line naming the bug, e.g. `TT-XLA Incorrectly Uploads Non-Row-Major Host Buffers`.
2. **Summary** — 1–2 short paragraphs. First paragraph: what broke and the concrete failure that exposed it. Second paragraph: the one-sentence shape of the fix.
3. **Status** — bullet list. Always include: bug type, component, whether fixed locally, and any remaining issues that this fix did *not* resolve.
4. **Repositories** — every repo touched, with absolute path, branch, commit SHA, and whether the worktree was dirty. Include external model/data snapshots (e.g. HF revision hashes) here too.
5. **Host Environment** — OS, kernel, language runtimes, key library versions, and device inventory. Include the exact output of any hardware-listing command used (e.g. `tt-smi -ls`).
6. **User-Visible Failure** — the symptom as the user saw it, with raw terminal output in fenced blocks. If a reduced reproducer also fails, show both the original symptom and the reduced one, and explain why the reduction matters.
7. **Root Cause** — what was actually wrong. Enumerate sub-problems if there are several. Be specific about functions and files.
8. **Fix** — what the patch does, the list of files it touches, and a link to the patch under `supplemental/`.
9. **Minimal Reproducer** — link to the repro script in `supplemental/`, followed by a short numbered list of what the script does and the expected before/after behavior.
10. **Reproduction Steps** — exact shell commands, including `source .venv/bin/activate` or equivalent, and any environment variables the run depends on.
11. **Verification** — before/after numbers in fenced output blocks. If the fix improves but does not fully resolve the symptom, say so plainly and note that the residual is a separate bug.
12. **Notes** — caveats, scope limits, anything a future reader should not over-interpret (e.g. "worktrees were dirty when diagnosed, so the patch is minimal").

## Writing style

- Terse and factual. No marketing, no hedging, no narrative filler.
- Concrete over abstract: name the function, file, flag, or commit rather than gesturing at "the upload path".
- Past tense for what happened, present tense for invariants and current behavior.
- Bullet lists are fine for enumerations; prose is fine for explanations. Don't bullet single items.
- Show, don't summarize, raw output. Paste the terminal block verbatim.

## Code-block conventions

- Use ` ```text ` for terminal output, log dumps, and numeric diffs.
- Use ` ```bash ` for shell commands a reader would run.
- Use the appropriate language tag (`python`, `cpp`, `diff`, etc.) for source snippets.
- Never paste a giant patch inline — put it in `supplemental/` and link to it.

## Supplemental artifacts

- Repro scripts should be runnable as-is from a documented working directory with a documented venv. Set environment variables inside the script via `os.environ.setdefault(...)` so the script is reproducible without extra shell setup.
- Patches should be minimal: only the hunks relevant to the documented bug. If the source worktree was dirty, say so in **Notes** and trim the patch accordingly.
- Keep filenames descriptive (`repro_fortran_upload_matmul.py`, `ttxla_fix.patch`), not generic (`script.py`, `fix.patch`).

## What a lesson is not

- Not a changelog entry — focus on diagnosis and reproduction, not on celebrating the fix.
- Not a design doc — if the work is forward-looking rather than a postmortem, it doesn't belong here.
- Not a dumping ground — if a section would be empty or speculative, leave it out.
