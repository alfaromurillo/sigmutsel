# sigmutsel

**Signature-based mutation rate estimation and selection inference in cancer**

A mutation that is frequent in tumors is not necessarily one that
helps them grow: some sites mutate far faster than others. `sigmutsel`
estimates, for every tumor, the **neutral mutation rate** of every
gene and every variant (from the tumor's mutational signatures, each
gene's sequence opportunity and gene-level covariates such as
replication timing and chromatin), and then the **selection
coefficient** `gamma`: how many times more often a gene or variant is
found mutated than its rate predicts. `gamma = 1` is neutral.

It provides:

- a `MutationDataset` built from MAF files: call quality control, a
  germline mask, COSMIC signature decomposition, and each gene's
  opportunity by consequence (synonymous, missense, nonsense, splice);
- a `Model` of the neutral rates: per-tumor spectra, gene covariates,
  a per-gene rate correction learned from silent mutations, fitted by
  Hamiltonian Monte Carlo;
- the rate of any single-nucleotide variant in any tumor
  (`sigmutsel.site_rates.SiteRates`);
- selection coefficients for genes, consequence channels and variants,
  with intervals that carry the rate model's uncertainty; a selection
  call for every gene (a null test with Benjamini-Hochberg) and its
  strength under an empirical-Bayes prior.

## Start here: the tutorial

**[tutorials/selection_in_hnsc.md](tutorials/selection_in_hnsc.md)**
goes from a public TCGA cohort (head and neck cancer, downloaded from
the GDC) to rates, selection coefficients, the selection call, and two
small research projects: a frequent mutation that is mostly mutation
rate against a rarer one that is strongly selected, and comparing
selection between smokers and never-smokers. It states the runtime of
every step (about 80 minutes on a desktop, after an 18-minute setup;
one step needs 16-20 GB of memory) and the mistakes to avoid. Its
source, `tutorials/selection_in_hnsc.py`, runs as a script or as a
notebook.

Covariates come from the companion package
[sigmutselcovs](https://github.com/alfaromurillo/sigmutselcovs), which
publishes ready-made gene covariate matrices for every TCGA cohort on
Zenodo (its tutorial also shows how to use them in dNdScv).

## Installation

Python 3.12, Linux or macOS. Use a fresh virtual environment:

```bash
python3.12 -m venv sigmutsel_env
source sigmutsel_env/bin/activate
pip install git+https://github.com/alfaromurillo/sigmutsel.git
sigmutsel-setup        # reference data and the GRCh38 genome, once
sigmutsel-check-data   # every line should say "Found"
```

`sigmutsel-setup` downloads about 300 MB of reference files (Ensembl
CDS sequences, GENCODE annotations, HGNC, the exome target, a liftover
chain, the germline-mask table, RepeatMasker regions) and installs
SigProfilerMatrixGenerator's GRCh38 reference genome (about 3 GB,
stored inside that package's own directory). **The genome is needed**:
without it nothing fails, but genome-dependent contexts fall back and
the rates differ (you see `RuntimeWarning: No installed GRCh38
genome`). `--skip-genome` leaves it out, for machines that only read
saved results.

To keep the reference files somewhere else than the package
directory:

```bash
sigmutsel-setup --data-dir ~/sigmutsel_data
export SIGMUTSEL_DATA_DIR=~/sigmutsel_data   # in every session that uses it
```

The versions sigmutsel is tested with are pinned in
`ci/constraints-pinned.txt`; to install exactly those,
`pip install -c ci/constraints-pinned.txt .` from a clone. For
development, `pip install -e ".[dev]"` (see
[CONTRIBUTING.md](CONTRIBUTING.md)). [SETUP_GUIDE.md](SETUP_GUIDE.md)
has the reference files in detail.

## Quick start

The whole recommended pipeline, without the explanations (the
tutorial has them):

```python
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pandas as pd
import sigmutselcovs
from sigmutsel import Model, MutationDataset
from sigmutsel.download_tcga_data import download_project_maf_files
from sigmutsel.qc import load_repeat_intervals
from sigmutsel.setup import download_repeatmasker_bed
from sigmutsel.tcga_sample_selection import select_tcga_maf_files

# 1. Calls: download, then one treatment-naive primary tumor per patient
download_project_maf_files("HNSC", "hnsc/raw")
chosen = select_tcga_maf_files("hnsc/raw", duplicate_policy="newest",
                               exclude_prior_treatment=True, project="HNSC")
# ... link the chosen files into "hnsc/maf_dir" (tutorial, step 1)

# 2. Dataset
dataset = MutationDataset("hnsc/maf_dir", signature_class="SBS")
dataset.generate_mutation_db(
    qc_mode=True,
    qc_kwargs={"repeat_intervals": load_repeat_intervals(
        download_repeatmasker_bed("hg38"))},
    germline_mask_af=1.5e-4)
dataset.run_two_pass_signature_decomposition(
    exclude_signature_subgroups="HNSC", treatment_naive=True,
    genome_build="GRCh38", exome=True, cosmic_version=3.6,
    treatment_load_threshold=0.2, prevalence_override_min_fraction=0.15)
dataset.build_full_dataset()  # keeps the cleaned calls
# ... sample QC, spectrum shrinkage, site weights (tutorial, step 3)

# 3. Rate model, on the published covariate components
pcs = sigmutselcovs.fetch_covariate_matrix("HNSC")
files = sigmutselcovs.fetch_covariate_artifacts("HNSC", artifacts=("pca_indicators",))
design = pd.concat([pcs.iloc[:, :300],
                    pd.read_parquet(files["pca_indicators"])], axis=1)
model = Model(dataset, None, cov_effects_per_sigma=False,
              prob_g_tau_tau_independent=False)
model.assign_cov_matrix(design)
model.compute_channel_base_mus()
model.estimate_channel_rg_cov_effects(excluded_samples=excluded,
                                      include_drivers=True, sample="full")

# 4. Selection
model.estimate_gene_tumor_dispersion(excluded_samples=excluded)
settings = dict(excluded_samples=excluded, use_mu_posterior=True,
                mu_uncertainty="draws", r_g_variant="evaluation",
                gene_tumor_dispersion="fitted")
tp53 = model.estimate_gamma("TP53", level="gene", **settings)
e545k = model.estimate_gamma("PIK3CA p.E545K", level="variant", **settings)
table = model.gene_selection_table(excluded_samples=excluded, fdr=0.1)
```

### The defaults are not the recommended settings

Several defaults are kept so that older analyses reproduce. Pass
these:

| call | argument | default | recommended | why |
|---|---|---|---|---|
| `select_tcga_maf_files` | `duplicate_policy` | `"keep_all"` | `"newest"` | one aliquot per patient |
| `select_tcga_maf_files` | `exclude_prior_treatment` | `False` | `True` | treated tumors carry therapy mutations |
| `MutationDataset.generate_mutation_db` | `qc_mode`, `germline_mask_af` | off | `True`, `1.5e-4` | call QC and the germline mask |
| `MutationDataset.run_two_pass_signature_decomposition` | `treatment_load_threshold`, `prevalence_override_min_fraction` | off | `0.2`, `0.15` | drop treatment-exposed tumors; keep signatures that recur |
| `Model` | `prob_g_tau_tau_independent` | `None` (unset) | `False` | a tumor's spectrum shapes where its mutations fall; `True` is an older model |
| `Model.estimate_gamma` | `use_mu_posterior` | `False` | `True` | the rates' uncertainty in gamma's interval |
| `Model.estimate_gamma` | `mu_uncertainty` | `None` (per-tumor approximation) | `"draws"` | the exact cut; the approximation's intervals are too narrow |
| `Model.estimate_gamma` | `r_g_variant` | `"none"` | `"evaluation"` | the gene's rate correction from silent mutations only |
| `Model.estimate_gamma` | `gene_tumor_dispersion` | `None` | `"fitted"` (after `estimate_gene_tumor_dispersion`) | mutations cluster in some tumors |

### Reading the results

- `gamma` is the **strength of selection** relative to the variant's
  or gene's own neutral rate, not a probability of being a driver.
- Decide *whether* a gene is selected with the selection table's
  `called` (`q < 0.1`); compare *how strongly* across genes with its
  empirical-Bayes estimates (`eb_median`, `eb_lower`, `eb_upper`).
  One flat-prior interval above 1 is not a call.
- For oncogenes read variant (or missense-channel) gammas; for tumor
  suppressors the `plof` channel (nonsense and essential splice).
- `Model.mu_gs` does not include the gene's rate correction `r_g`;
  multiply by `compute_r_g_for_evaluation()`.

## Downloading TCGA data

`sigmutsel.download_tcga_data.download_project_maf_files(project,
destination)` downloads a TCGA project's open-access MAFs from the GDC
over HTTPS (through [gdcfetch](https://github.com/alfaromurillo/gdcfetch)),
decompressed into one flat directory: no account and no other program.
A rerun fetches only what is missing. Select the files with
`select_tcga_maf_files` before building a dataset (see the table above).

For a manifest made by hand on the [GDC Data Portal](https://portal.gdc.cancer.gov/)
(Files: Simple Nucleotide Variation, MAF, "Aliquot Ensemble Somatic
Variant Merging and Masking"), `download_tcga_maf_files(manifest,
destination)` does the same through NCI's `gdc-client`, which must be
on `PATH` ([GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)).
`examples/example_gdc_manifest.txt` is such a manifest (50 SKCM
samples).

## API reference

[docs/api.md](docs/api.md), generated from the code (signatures and
defaults are the code's): the functions of the tutorial's workflow in
full, then a one-line summary of every other public name.

## Package structure

| module | what it does |
|---|---|
| `models` | `MutationDataset` (data) and `Model` (rates and selection) |
| `download_tcga_data`, `tcga_sample_selection` | getting and selecting TCGA MAFs |
| `qc`, `sample_qc`, `germline_mask`, `repeatmasker` | call and sample quality control |
| `signature_decomposition`, `spectrum_shrinkage` | signatures and per-tumor spectra |
| `channel_universe`, `site_weights`, `capture` | transcripts, opportunities, detection |
| `estimate_mus`, `estimate_rg`, `estimate_covariates_effect` | the rate model |
| `site_rates` | the rate of any variant in any tumor |
| `estimate_gammas`, `selection`, `gene_tumor_dispersion` | selection coefficients, the call and the EB prior |
| `cross_validation` | held-out comparison of rate models |
| `setup`, `locations` | reference data |

## Related packages

- [sigmutselcovs](https://github.com/alfaromurillo/sigmutselcovs):
  gene covariate matrices for every TCGA cohort, built from public
  tracks and published on Zenodo.
- [gdcfetch](https://github.com/alfaromurillo/gdcfetch): search and
  download any GDC data.

## License

MIT License, see [LICENSE](LICENSE).

## Citation

If you use this software in your research, please cite:

```
[Citation to be added]
```

## Contributing and support

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
For questions and issues, use the
[GitHub issue tracker](https://github.com/alfaromurillo/sigmutsel/issues).
