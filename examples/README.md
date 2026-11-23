# Examples

Two short scripts of the **older, single-channel models**
(`compute_mu_gs`, `estimate_cov_effects`), kept for reference (they
are not run by the test suite):

- `basic_usage.py`: a no-covariate model on the 50 SKCM tumors of
  `example_gdc_manifest.txt` (downloaded with `gdc-client`);
- `with_covariates.py`: the same with GTEx skin expression as a
  covariate (`example_gtex_skin_covariates.parquet`).

For the recommended model (the consequence-split channel model, its
covariates from sigmutselcovs, and the selection estimates with their
recommended settings), follow the tutorial instead:
[`../tutorials/selection_in_hnsc.md`](../tutorials/selection_in_hnsc.md).
