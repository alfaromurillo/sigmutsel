# This branch is superseded — do not merge it

Its single commit, `15eacb0 Add gene-level CV for passenger-gene R2`,
was the first attempt at gene-level cross-validation for
passenger-gene R². **That work now lives on `master`**, reimplemented
from scratch. This branch is kept only as a historical record of the
first attempt.

## What happened

The branch predates the entire consequence-split (`channel_dep`)
epic. By the time it was revisited, its diff against `master` had
stopped being an addition and become a partial *reversion*: merging
it would have **removed** channel-split fields that landed after it
was written. It could not be rebased into something safe to merge,
so it was reimplemented rather than salvaged.

The replacement is `gene-cv-passenger-r2-v2` (merged to `master`
2026-09-09), which:

- restores `train_genes`/`genes` on `estimate_cov_effects` and
  `estimate_passenger_genes_r2` with the same API, so callers needed
  no changes;
- **adds** `train_genes` support to
  `estimate_channel_rg_cov_effects`/`_channel_gene_statistics`, which
  this branch could not have had — the consequence-split model did
  not exist yet — and which gene-level CV against the production
  model requires;
- fixed a live regression on the compute machine along the way
  (`tcga_analysis/code/pca_nc_cv_sweep.py` was failing with
  `ModuleNotFoundError` precisely because this stale branch was what
  had been installed there);
- later gained per-gene residual output (`return_per_gene`), which
  is what made block-bootstrap and calibration diagnostics possible.

## Why this matters beyond the branch

The gene-CV machinery this work restored is what exposed the largest
methodological finding of the 2026-09 pass: every cohort's
`PCA_N_COMPONENTS` had been selected by a rule that compared
fold-sharing arms using *marginal* rather than *paired* standard
errors, leaving 7 of 13 cohorts at nc=5 and discarding held-out gains
as large as +0.0407 R² (DLBC) at paired t=13 (BLCA). See
`mutation_rates/TODO.md`, "Objective C, ninth pass".

## If you are here looking for gene-level CV

Use `master`. `sigmutsel.cross_validation` has both
`gene_cv_passenger_r2` and `channel_gene_cv_passenger_r2`, the latter
being the one that matches the production model.

*Kept, not deleted, deliberately — the first attempt is the record of
why the second one was written the way it was.*
