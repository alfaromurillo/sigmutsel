# This branch is superseded — do not merge it

Its gene-level cross-validation work **is on `master`**, merged
2026-09-09 (`e2a8d03 Return per-gene residuals from R2 and gene
CV`). `git log master..gene-cv-passenger-r2-v2` is empty: this
branch holds nothing `master` does not already have. It is kept as
a record of how the work was developed, not as anything to merge.

## What this branch was

It is the *second* attempt at gene-level CV for passenger-gene R².
The first, `gene-cv-passenger-r2`, predated the consequence-split
(`channel_dep`) epic, so by the time it was revisited its diff
against `master` had become a partial reversion — merging it would
have removed channel-split fields added after it was written. That
branch carries its own announcement; it was reimplemented here
rather than salvaged.

What this branch added over the first attempt:

- `train_genes` support on the channel entry points
  (`estimate_channel_rg_cov_effects`, `_channel_gene_statistics`),
  which the first attempt could not have had, and which gene-level
  CV against the production model requires;
- `return_per_gene` on `estimate_passenger_genes_r2` and
  `channel_gene_cv_passenger_r2`, returning gene-indexed observed
  and expected counts, which the per-sample allocation diagnostics
  are built on.

## A note on the remote tip

`origin/gene-cv-passenger-r2-v2` points at `8a31351`, a pre-rebase
version of what became `9bc9e0c` on `master`. The remote branch is
therefore *behind* this one and older than `master`. Do not merge
or fast-forward from it; if the remote branch is ever updated, it
should be force-pushed to match this tip, not merged.
