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
| `utils.py` | `run_pca_on_covariates`, `run_riemannian_stats_on_covariates` |

## Non-obvious rules

- `assign_cov_matrix` accepts `dr_method='pca'|'riemannian_stats'` and
  `dr_kwargs={}`; `run_pca=True` still works as a backwards-compat alias
- Riemannian STATS requires `pip install riemannian-stats` (optional dep,
  `sigmutsel[riemannian]`); import is lazy so the package is not required
  at load time
- The `riemannian-stats` PyPI package has an O(n²·p) memory bottleneck;
  `run_riemannian_stats_on_covariates` reimplements the same algorithm
  without that tensor — safe at genome scale (n~18k, p~223)
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
- `constants.random_seed` controls UMAP reproducibility in
  `run_riemannian_stats_on_covariates`; default is `None` (stochastic).
  Callers override at runtime via `import sigmutsel.constants;
  sigmutsel.constants.random_seed = 777` — do NOT use
  `from sigmutsel.constants import random_seed; random_seed = 777`
  (that only rebinds a local name and has no effect)
- Riemannian components are nested when seed is fixed: RC1..RC_k from
  an nc=k run are bit-identical to the first k columns of an nc=N (N>k)
  run. Without a seed, separate calls produce different UMAP graphs and
  thus different components
- UMAP `n_neighbors` has negligible effect on passenger-gene R² across
  the range [5,50]; `n_components` is the lever that matters (R² rises
  from 0.648 at nc=5 to 0.693 at nc=30, knee at nc≈20)
- Explained inertia of `cov_matrix_full` has a sharp elbow: RC1=52%,
  RC2=22%, RC3=3% — but predictive R² keeps improving well past nc=3,
  so inertia and R² tell different stories

## Testing

```bash
pytest tests/                     # all tests
pytest tests/test_smoke_imports.py  # import sanity only
```
