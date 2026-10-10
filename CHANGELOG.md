# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Indel hold-out for gamma: `MutationDataset.indel_db` (built from
  the dataset's MAFs by `build_indel_db`, via
  `load_maf_files.read_coding_indels`; manifest version 9) holds the
  non-silent coding insertions and deletions, and
  `estimate_gamma(hold_out_indel=True)` leaves a tumor with one in a
  gene out of that gene's and its variants' absent sets, recorded as
  `n_tumors_held_out_indel`. With the same-gene hold-out, a variant's
  absent set is then cancereffectsizeR's, except that tumors whose
  only hit in the gene is synonymous stay in.
- `presence_clip` (`estimate_gamma_from_mus`, `Model.estimate_gamma`):
  False scores the exact likelihood (`log_absence_probability`)
  instead of clipping P(present) at 1e-12, which caps an absent
  tumor's cost at 27.6 nats. With no dispersion, no mu cut and
  `absence_floor=0` the likelihood is then cancereffectsizeR's
  `sswm_lik`, and the MAP its maximum-likelihood estimate (tested in
  `tests/test_ces_parity.py`).
- Spectrum shrinkage (`spectrum_shrinkage`): each tumor's per-type
  spectrum is the signature reconstruction shrunk toward the
  cohort's passenger pool with a 10% share of SBS5 (`kappa_1`), then
  the tumor's own passenger counts shrunk toward that (`kappa_2`).
  Every type keeps a positive rate, and high-burden tumors follow
  their own spectrum. The kappas are per-cohort parameters, chosen on
  held-out mutations (`heldout_spectrum_scores`, `select_kappas`,
  with a nested cross-validation over tumors that reports the
  choice's optimism), stored with the dataset
  (`MutationDataset.set_spectrum_shrinkage`, manifest version 8) and
  applied by `Model.compute_mu_taus`
  (`Model.baseline_spectrum_shrinkage` records them).
- Genes with incomplete covariates get a rate instead of being
  dropped. With a PCA-reduced covariate matrix, such a gene is
  projected onto the complete genes' components from the covariates
  it has (`utils.project_onto_pca`; a missing standardized value is
  the column mean) and shifted by one fitted factor per covariate
  block it misses (`Model.estimate_fallback_shifts`,
  `estimate_rg.fit_fallback_shifts`), fitted after, and without
  changing, the complete genes' coefficients. `assign_cov_matrix`
  takes the column-to-block map as `missing_blocks`; the affected
  genes are `Model.covariate_fallback_genes`, and their `r_g`, the
  channel offset and the mu-posterior draws all apply. Gamma fits
  record `covariate_fallback` and `covariate_fallback_blocks` in
  `posterior.attrs`. Without PCA the fallback is the baseline rate.
- `MutationDataset.counts_by` cross-tabulates per-variant or
  per-gene tumor counts against a caller-supplied grouping of the
  tumors, in counts or in per-group prevalence.
- `estimate_gammas.presence_probability` exposes the link from
  gamma to P(present) as its own function, and
  `estimate_gamma_from_mus(presence_model=...)` replaces it.
- Compound variants: `compound_variants.define_compound_variants`
  groups variants by annotation columns and genomic proximity, and
  `Model.estimate_gamma_compound` fits one shared gamma over a
  group (rates summed, presence ORed).
- Gamma fits now record how many tumors they used --
  `n_tumors_with` / `_without` / `_included` / `_excluded` /
  `_held_out` in `posterior.attrs`, tabulated by
  `Model.gamma_sample_accounting()`.
- `signature_attribution.compute_signature_probabilities` returns
  the full per-mutation, per-signature attribution frame (the
  breakdown behind the existing gene-level and summed-mass
  functions).
- `signature_attribution.compute_signature_effect_shares` and
  `Model.signature_effect_shares` compute the share of a cohort's
  selection attributable to each signature -- attribution weighted
  by each unit's gamma, normalized per tumor and averaged over
  tumors -- against the signatures' plain source shares.
- Saved datasets and models now record their provenance: the
  `sigmutsel` version and git commit that wrote them (warned about
  on load if the running build differs), and a `run_history` of the
  calls that produced them, exposed as
  `MutationDataset.run_history` / `Model.run_history`. Manifest
  schema versions bumped accordingly (dataset 3 -> 4, model 1 -> 2);
  manifests written before this load unchanged.

### Changed
- `Model.aggregate_signatures` now raises `TypeError` (was
  `ValueError`) when `base_mus` isn't signature-separated -- it's a
  type check, not a value check. Callers catching `ValueError`
  specifically from this method should catch `TypeError` instead.

### Fixed
- Per-type rates for a gene without covariates (the basis of its
  variant rates) omitted the non-synonymous channel's `delta`
  offset.

## [0.1.1] - 2025-12-24

### Fixed
- Fixed signature class to variant type mapping in MAF file processing
- SBS, DBS, and RNA-SBS signature classes now correctly map to SNP, DBP variant types
- Resolved "'NoneType' object is not subscriptable" error when using signature_class="SBS"

## [0.1.0] - 2025-12-24

### Added
- Initial release
- Core functionality for mutation rate estimation
- Selection coefficient inference
- Multi-signature support
