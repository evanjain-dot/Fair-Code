# CLAUDE.md - Standing Instructions

> Read this before making changes to this repo. It applies to AI agents and human contributors alike.

## Paper freeze - lifted

This repo was previously under a paper freeze, on the assumption that a research paper citing its
benchmark results was in peer review. That paper was never actually submitted this cycle - the real
paper, with a fresh run of results, is now planned for **next year**. Freezing analysis code and
results against a paper that doesn't exist yet was premature, so the freeze is lifted: `faircode/`
core, `results/`, `audit.yaml` manifests, dataset CSVs, `requirements-lock.txt`, and the reproducibility
parameters (`random_state`, split ratio, iteration counts, the fairness constraint, the six metrics)
are **all open to normal development again**. New audits merge to `main` like anything else. Dependabot's
version-bump PRs against `requirements-lock.txt` no longer need to be reverted on sight - use normal
judgment.

## `paper/results-frozen/` stays, as a reference

`paper/results-frozen/` is kept in the repo, but it is no longer under any special protection - it's
a snapshot of an earlier analysis pass, useful as a point of comparison while building toward next
year's real paper run, not the evidence backing a live citation. Nothing stops it from being read,
compared against, or eventually superseded; there's just no active enforcement requiring `results/`
to keep matching it, since `results/` is expected to move again as development resumes. The
`v1.0-paper` git tag is left in place as a historical marker of what that snapshot was.

If an explainer or piece of documentation quotes a Fair Code benchmark result, say which source it's
quoting - `results/` (current) or `paper/results-frozen/` (the earlier reference snapshot) - so a
reader isn't left guessing which numbers are live.

## Next freeze

When the real paper submission is actually ready (expected next year), expect a new freeze to be
established at that point, covering whatever fresh results back that submission - this file will be
updated again when that happens. Until then, treat this repo as under normal, fully open development.
