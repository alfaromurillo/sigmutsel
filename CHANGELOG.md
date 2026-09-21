# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
