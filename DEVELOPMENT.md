# Developing sigmutsel

Internals reference for anyone modifying this codebase. For setup and
the contribution workflow, see `CONTRIBUTING.md` and `SETUP_GUIDE.md`.

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
| `download_tcga_data.py` | `gdc-client`-based MAF download/unpack |
| `tcga_sample_selection.py` | Which downloaded MAF files to use: sample-type filter + per-case duplicate policy (see below) |
| `constants.py` | Central parameters (SBS96 types, chr list) |
| `locations.py` | Data file paths; respects `SIGMUTSEL_DATA_DIR` |
| `figures.py` | γ posterior scatter plots; **no titles** |
| `utils.py` | `run_pca_on_covariates`, `run_riemannian_stats_on_covariates` |

## Testing

```bash
pytest tests/                       # all tests
pytest tests/test_smoke_imports.py  # import sanity only — no deep
                                     # coverage of estimate_mus.py /
                                     # models.py math, so verify
                                     # correctness-critical changes
                                     # with your own synthetic checks
```

## `Model` internals

- `Model` is `@dataclass(repr=False, init=False)` with a fully
  hand-written `__init__` — dataclass `field(default_factory=...)`
  defaults do **not** auto-populate, since `init=False` disables the
  generated `__init__`. Any new instance attribute must be explicitly
  set in `__init__` (both `MutationDataset`/`Model`'s own
  construction and `load_model`'s `model = cls(...)` call go through
  it), or it raises `AttributeError` on constructed/loaded models.
- `model.gammas` keys: gene results are stored by ENSG ID (e.g.
  `"ENSG00000141510"`), variant results by display string with
  spaces (e.g. `"KRAS p.G12D"`). To map gene keys back to symbols:
  `mutation_db[["gene","ensembl_gene_id"]].drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["gene"]`
- `save_model` writes `gammas/*.nc` keyed by object identity — it
  only skips rewriting a file when the in-memory result is literally
  the object `load_model` read from that same path (tracked via
  `_gammas_loaded_from_disk`), not merely because a file already
  exists there. If you delete a `model.gammas[key]` entry to force a
  recompute, the new result correctly overwrites the old file; don't
  assume a stale file needs manual cleanup, but do verify a
  save/reload round-trip if you're ever unsure a recompute "took".
- `assign_cov_matrix` accepts `dr_method='pca'|'riemannian_stats'` and
  `dr_kwargs={}`; `run_pca=True` still works as a backwards-compat
  alias.
- Riemannian STATS requires `pip install riemannian-stats` (optional
  dep, `sigmutsel[riemannian]`); import is lazy so the package is not
  required at load time. The `riemannian-stats` PyPI package has an
  O(n²·p) memory bottleneck; `run_riemannian_stats_on_covariates`
  reimplements the same algorithm without that tensor — safe at
  genome scale (n~18k, p~223).
- Riemannian components are nested when `constants.random_seed` is
  fixed: RC1..RC_k from an `nc=k` run are bit-identical to the first
  k columns of an `nc=N` (N>k) run. Without a seed, separate calls
  produce different UMAP graphs and thus different components.
- `cov_effects_per_sigma=True` enables per-signature covariate
  effects; this is experimental and requires `signature_selection`.
- `prob_g_tau_tau_independent` selects between two mathematical
  paths in `compute_mu_g_per_tumor`: `True` treats gene probability
  as type-independent (simpler, uses total gene opportunities);
  `False` (default) computes type-specific p(g|τ) from per-context
  opportunities — see the function's own docstring for the exact
  formulas.

## Gamma estimation (`estimate_gammas.py`)

- A bounded prior (`pm.Uniform(0, upper_bound_prior)` in
  `estimate_gamma_from_mus`) can silently truncate γ's posterior
  with clean R-hat/ESS diagnostics — sampling up to a hard edge
  isn't a numerical problem, so nothing flags it on its own.
  `estimate_gamma_from_mus` auto-expands the bound when the
  posterior mean exceeds `saturation_ratio` (default 20%) of the
  current bound, refitting up to `max_bound_expansions` times. Watch
  for the same failure mode in any other bounded-prior estimation
  added later, and don't trust a capped-looking γ just because MCMC
  diagnostics look fine.
- `constants.random_seed` controls reproducibility; default is
  `None` (stochastic). Callers override at runtime via
  `import sigmutsel.constants; sigmutsel.constants.random_seed = 777`
  — do **not** use `from sigmutsel.constants import random_seed;
  random_seed = 777` (that only rebinds a local name and has no
  effect, since the estimation functions read `constants.random_seed`
  as a module attribute at call time).

## TCGA sample selection (`tcga_sample_selection.py`)

`download_tcga_data.py` fetches one MAF file per sequenced
*aliquot*, with no sample-type filtering or per-case dedup — a case
(patient) with multiple sequenced aliquots becomes multiple
independent files (e.g. a re-plated portion, or both a primary and a
metastatic tumor).

`select_tcga_maf_files(maf_dir, ...)` is the entry point: sample-type
filter (default keeps only 01/03 — Primary Solid Tumor and Primary
Blood Derived Cancer — matching cancereffectsizeR's
`get_TCGA_project_MAF()`), then a per-case duplicate policy
(`keep_all`/`oldest`/`newest`/`random`), then optional
`exclude_prior_treatment`. It does **not** run automatically inside
`main.py`-style pipelines or `MutationDataset` construction — call it
on an already-downloaded `all_maf_files/`-style directory and feed
the returned subset forward yourself if you want filtering applied.

Barcode parsing (`TcgaBarcodeInfo`, `parse_tcga_barcode`,
`SAMPLE_TYPE_CODES`) and the GDC cases-API lookup
(`fetch_case_metadata`, aliased here as `fetch_gdc_case_metadata`)
live in `gdcfetch.tcga_barcode`, a dependency — this module
re-exports those names so callers don't need a second import.

## MAF preprocessing QC (`qc.py`)

`load_maf_files.py`'s default `validate_full` silently drops
invalid rows via boolean masking, with only a `logger.warning` as a
record of what was dropped. `qc.py` offers a structured alternative
and three additional checks, all opt-in (conventions adapted from
cancereffectsizeR, an R package addressing the same estimation
problem — see https://github.com/Townsend-Lab-Yale/cancereffectsizeR):

- `apply_qc(df, ...)` runs the full sequence and returns every row
  with a `"problem"` column (`None` for a clean row, else a reason
  string) instead of dropping anything — enable via
  `qc_mode=True` on `process_single_maf`/
  `load_validate_compact_all_maf_files_parallel`/
  `MutationDataset.generate_mutation_db`/`.build_full_dataset`
  (forwarded through `**kwargs` at every layer). Rows still tagged
  `None` after `apply_qc` are what actually gets kept when
  `qc_mode=True` is used through that chain -- tagged rows are
  dropped before `compact_data`, same as the default path, but with
  a per-file problem-count summary logged first.
- `flag_exact_duplicates`: same sample+chromosome+position+alleles
  appearing twice (no equivalent check existed before).
- `flag_germline_variants`: tags a row whose `gnomAD_non_cancer_MAX_AF_adj`
  exceeds a threshold (default 0.1%) as likely germline, not
  somatic. That column is already present in raw GDC MAFs (see
  `constants.maf_column_descriptions`) but was previously dropped
  before it could be used.
- `detect_mnv_dbs`: same-sample SNVs within 2bp of each other are
  very likely one real multi-nucleotide event an upstream caller
  split into separate single-base calls (classic case: UV-signature
  CC>TT), not two independent substitutions. Tags the whole cluster
  for exclusion, since this package has no DBS/MNV modeling
  downstream to route them to instead.
- `flag_repetitive_regions` (needs `repeat_intervals`, built by
  `load_repeat_intervals` from a file downloaded via
  `setup.download_repeatmasker_bed`): tags calls overlapping a
  RepeatMasker-annotated region, where sequencing/mapping artifacts
  cluster. Not part of `apply_qc`'s default checks (needs an
  external download the others don't) — pass `repeat_intervals`
  explicitly to enable it.
- `check_sample_overlap(mutation_db)` is a separate, cohort-level
  diagnostic (not part of `apply_qc`, and not wired into the
  pipeline anywhere) that flags sample *pairs* with suspiciously
  high shared-mutation counts — catches contamination or a sample
  processed twice under different barcodes, which barcode-based
  dedup in `tcga_sample_selection.py` can't see (that only catches
  *known* same-patient duplicates by barcode). Call it directly on
  a built `MutationDataset.mutation_db` when investigating a cohort.

## General conventions

- **No titles in matplotlib figures** — titles go in captions.
- **Optional numeric parameters**: use `if value is not None`, not
  `if value` — `0` is a valid axis-limit value and is falsy.
- Results cached as `.npy`/`.parquet`/`.nc`; use
  `force_produce_results=True` or `force_generation=True` to
  recompute.
