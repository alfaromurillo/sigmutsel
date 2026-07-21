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

## General conventions

- **No titles in matplotlib figures** — titles go in captions.
- **Optional numeric parameters**: use `if value is not None`, not
  `if value` — `0` is a valid axis-limit value and is falsy.
- Results cached as `.npy`/`.parquet`/`.nc`; use
  `force_produce_results=True` or `force_generation=True` to
  recompute.
