# sigmutsel

## Project overview

Python package for estimating somatic mutation rates (μ) and
selection coefficients (γ) from TCGA tumor sequencing data.

Pipeline: MAF files → mutational signature decomposition (COSMIC
SBS) → per-context baseline μ estimation → covariate modeling →
γ inference.

Companion: `sigmutselcovs` builds covariate matrices (gene
expression, replication timing, chromatin) consumed by this package.
Analysis projects (`luad_analysis`, `coad_analysis`) consume both.

## Setup

```bash
pip install -e ".[dev]"   # installs with test dependencies
python -m sigmutsel setup  # downloads HGNC, GENCODE, CDS FASTA
pytest tests/              # smoke tests
```

## Key modules

| Module | Role |
|--------|------|
| `models.py` | `MutationDataset` and `Model` — the main API |
| `estimate_mus.py` | Core μ estimation (mu_tau, mu_g, mu_m) |
| `estimate_gammas.py` | Bayesian γ inference (PyMC) |
| `estimate_covariates_effect.py` | Covariate log-linear regression |
| `signature_decomposition.py` | COSMIC SBS decomposition wrapper |
| `signature_attribution.py` | P(σ\|τ,j) per gene |
| `compute_mutation_burden.py` | Synonymous burden, ℓ̂ estimation |
| `compute_alphas.py` | Per-sample signature exposure α |
| `contexts_by_gene.py` | Trinucleotide context counts from CDS |
| `load_maf_files.py` | MAF validation and compact DB loading |
| `constants.py` | Central parameters (SBS96 types, chr list) |
| `locations.py` | Data file paths; respects `SIGMUTSEL_DATA_DIR` |
| `figures.py` | γ posterior scatter plots; **no titles** |
| `utils.py` | `run_pca_on_covariates` and other helpers |

## Non-obvious rules

- **`model.gammas` keys**: gene results stored by ENSG ID
  (e.g., `"ENSG00000141510"`), variant results by display string with
  spaces (e.g., `"KRAS p.G12D"`). To map gene keys back to symbols:
  `mutation_db[["gene","ensembl_gene_id"]].drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["gene"]`
- **Optional numeric parameters**: use `if value is not None` not
  `if value` — 0 is a valid axis-limit value and is falsy
- **No titles in matplotlib figures** — titles go in captions
- Results cached as `.npy`/`.parquet`/`.nc`; use
  `force_produce_results=True` or `force_generation=True` to recompute
- `cancer_epistasis` uses μ values from this pipeline — do not
  change `mu_gs` column names or array shapes
- The `prob_g_tau_tau_independent` flag selects between two
  mathematical paths in `compute_mu_g_per_tumor`; the COAD analysis
  uses `False` (type-dependent), the baseline model uses `True`
- `cov_effects_per_sigma=True` enables per-signature covariate
  effects; this is experimental and requires `signature_selection`

## Testing

```bash
pytest tests/                     # all tests
pytest tests/test_smoke_imports.py  # import sanity only
```
