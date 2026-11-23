# Tutorial

The tutorial is in [`tutorials/`](tutorials/):

- **[Selection in head and neck cancer](tutorials/selection_in_hnsc.md)**
  (source: `tutorials/selection_in_hnsc.py`, a script that also opens
  as a notebook): from a public TCGA cohort to mutation rates per
  tumor and variant, selection coefficients for genes, channels and
  variants with their intervals, the selection call and its strength,
  and two research projects (frequency is not selection; selection in
  smokers and never-smokers). About 80 minutes on a desktop; every
  step's runtime and memory is stated.

The previous version of this page walked through the older
single-channel models (`compute_mu_gs`, `estimate_cov_effects`,
signature-specific covariate effects); the scripts in
[`examples/`](examples/) show them. The tutorial's channel model is
the recommended one: it is what the selection estimates' recommended
settings assume.

See also [README.md](README.md) (installation, the recommended
settings), [SETUP_GUIDE.md](SETUP_GUIDE.md) (reference data) and
[docs/api.md](docs/api.md) (API reference).
