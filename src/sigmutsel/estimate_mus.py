"""Estimate mutation rates at multiple levels.

This module provides functions to estimate baseline mutation rates
(μ) at different granularities:
- Per trinucleotide type per tumor (μ_τ,j)
- Per gene per tumor (μ_g,j)
- Per variant per tumor (μ_m,j)

The estimation accounts for mutational signatures, genomic contexts,
and optionally covariate effects to provide accurate baseline
mutation rate estimates used in selection inference.
"""

import logging
import pandas as pd
import numpy as np

from collections.abc import Sequence

from .estimate_presence import filter_passenger_genes_ensembl


logger = logging.getLogger(__name__)


def compute_mu_tau_per_tumor(
    db,
    location_signature_matrix,
    assignments,
    L_low=None,
    L_high=None,
    cut_at_L_low=False,
    separate_per_sigma=False,
):
    r"""Compute per-tumor per-type baseline mutation rates.

    Without considering covariates, the baseline mutation rate
    for a mutation of type τ in tumor j is:

    .. math::

        μ^{(j)}_{τ} = \hat{ℓ}^{(j)} \sum_{σ} α^{(j)}_{σ} s^{σ}_{τ}

    where σ is the signature index, α are exposures, s is the
    signature matrix, and \hat{ℓ} is the mutation burden.

    This function estimates mutation burden (\hat{ℓ}), refines
    signature exposures (α), and combines them with the
    signature matrix (s) to produce baseline mutation rates.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation database with one row per mutation record.
        Must contain columns: Tumor_Sample_Barcode, variant,
        type, etc.
    location_signature_matrix : str or Path
        Path to the normalized signature matrix. Should be a
        CSV/TSV file with rows = mutation types (τ) and
        columns = signatures (σ). Values must be normalized
        so each column sums to 1.
    assignments : pd.DataFrame
        Initial signature exposures with rows = tumor samples
        and columns = signatures. Will be refined by
        estimate_alphas() during computation.
    L_low : float or None, default None
        Lower burden threshold for correcting low-burden
        samples. If None, no correction is applied. Used to
        handle samples with very few mutations.
    L_high : float or None, default None
        Upper burden threshold for intermediate-burden
        correction. If None, no correction is applied.
        Typically used with L_low to define correction regions.
    cut_at_L_low : bool, default False
        How to handle low-burden samples when L_low is set:
        - False: Apply smooth correction to gradually adjust
          burden estimates
        - True: Hard clip all burden estimates below L_low to
          exactly L_low
    separate_per_sigma : bool, default False
        Whether to return signature-separated mutation rates:
        - False: Return single DataFrame summed across all
          signatures
        - True: Return dict mapping each signature to its
          contribution

    Returns
    -------
    pd.DataFrame or dict[int or str, pd.DataFrame]
        When separate_per_sigma=False (default):
            DataFrame with:
            - index: tumor sample barcodes
            - columns: mutation types τ
            - values: total mutation rate μ^{(j)}_{τ} summed
              across all signatures

        When separate_per_sigma=True:
            Dictionary mapping signature identifiers to
            DataFrames. Each DataFrame has:
            - index: tumor sample barcodes
            - columns: mutation types τ
            - values: signature-specific contribution
              \hat{ℓ}^{(j)} α^{(j)}_{σ} s^{σ}_{τ}

    Notes
    -----
    The function internally calls:
    - estimate_alphas(): Refines signature exposures α
    - estimate_ell_hats(): Computes mutation burden \hat{ℓ}
    - load_signature_matrix(): Loads signature matrix s

    The signature matrix will be filtered to only include
    signatures present in the assignments DataFrame.

    Examples
    --------
    >>> # Simple usage without burden correction
    >>> mu_tau = compute_mu_tau_per_tumor(
    ...     db, "signatures.csv", initial_exposures)

    >>> # With burden correction for low-count samples
    >>> mu_tau = compute_mu_tau_per_tumor(
    ...     db, "signatures.csv", initial_exposures,
    ...     L_low=50, L_high=200, cut_at_L_low=False)

    >>> # Get signature-separated rates
    >>> mu_by_sig = compute_mu_tau_per_tumor(
    ...     db, "signatures.csv", initial_exposures,
    ...     separate_per_sigma=True)
    >>> mu_sig1 = mu_by_sig['Signature_1']
    """
    from .compute_alphas import estimate_alphas
    from .compute_mutation_burden import estimate_ell_hats
    from .load_signature_matrix import load_signature_matrix

    alphas = estimate_alphas(db, assignments, L_low, L_high)

    ell_hats = estimate_ell_hats(
        db, L_low, L_high, cut_at_L_low=cut_at_L_low
    )

    sig_matrix = load_signature_matrix(location_signature_matrix)

    # Filter sig_matrix to only include signatures in assignments
    # (in case assignments was filtered to exclude signatures)
    common_sigs = sig_matrix.columns.intersection(assignments.columns)
    sig_matrix = sig_matrix[common_sigs]
    alphas = alphas[common_sigs]

    if not separate_per_sigma:
        mus = alphas.dot(sig_matrix.T).multiply(ell_hats, axis=0)
        return mus
    else:
        # Return dictionary: signature -> DataFrame(samples × types)
        # Compute: ell_hats * alpha_sigma * sig_sigma^T for each σ
        # Broadcasting: (samples,) * (samples,) -> (samples,)
        # then outer with (types,) -> (samples × types)
        mus_per_sigma = {
            sigma: pd.DataFrame(
                (ell_hats * alphas[sigma]).values[:, None]
                @ sig_matrix[sigma].values[None, :],
                index=alphas.index,
                columns=sig_matrix.index,
            )
            for sigma in sig_matrix.columns
        }

        return mus_per_sigma


def compute_mu_g_per_tumor(
    mu_taus: pd.DataFrame | dict[int | str, pd.DataFrame],
    contexts_by_gene,
    prob_g_tau_tau_independent=False,
) -> pd.DataFrame | dict[int | str, pd.DataFrame]:
    """Compute baseline per-gene expected mutation rate per tumor.

    Compute a Genes × Tumors matrix of expected mutation rates by
    mixing per-tumor, per-type rates with gene-level trinucleotide
    opportunities, without taking into account covariates.

    TODO: Make this function work with other other family of types:

        - DBS modify contexts_by_gene to count by the 10 possible source
          doublets (the DBS contexts):

          (From COSMIC:) "there are 16 possible source doublet bases
          (4 x 4). Of these, AT, TA, CG, and GC are their own reverse
          complement. The remaining 12 can be represented as 6
          possible strand-agnostic doublets. Thus, there are 4+6=10
          source doublet bases. Because they are their own reverse
          complements, AT, TA, CG, and GC can each be substituted by
          only 6 doublets. For the remaining doublets, there are 9
          possible DBS mutation types (3 x 3). Therefore, in total
          there are 4 x 6 + 6 x 9 = 78 strand-agnostic DBS mutation
          types."

        - ID: prob_g_tau tau_independent

        - CN: start prob_g_tau tau_independent leave for other to
          think more about it

        - SV: same as CN

    Parameters
    ----------
    mu_taus : pandas.DataFrame | dict[int | str, pandas.DataFrame]
        **Single DataFrame mode:**
            Tumors × 96 SBS types. Expected counts (or rates) per
            tumor *before* division by per-type opportunity totals.
            Index: tumor barcodes (e.g., 'TCGA-5M-A8F6-...').
            Columns: canonical SBS types (e.g., 'A[C>T]A', ...),
            length 96.

        **Dictionary mode:**
            Mapping from signature identifiers to DataFrames, each
            shaped (tumors × 96 types). As returned by
            :func:`compute_mu_tau_per_tumor` with
            ``separate_per_sigma=True``. When a dictionary is
            provided, the function processes each signature
            separately and returns a dictionary of per-gene rates.

    contexts_by_gene : pandas.DataFrame
        Gene opportunities indexed by ``ensembl_gene_id``. Columns
        must be contexts. For example for SNV (only option right now):
        trinucleotides *without* the middle change, i.e., the
        collapsed triplet formed from an SBS label as ``x[0] + x[2] +
        x[-1]`` (e.g., 'ACA', 'ACC', ...). Each entry is the count (or
        opportunity) of that context in the gene region used for
        modeling. As returned by
        :func:`contexts_by_gene.load_or_generate_contexts_by_gene`

    prob_g_tau_tau_independent : bool, default False
        If True, assume gene probability is independent of type:
        compute ``p(g)`` from total opportunities per gene and use the
        tumor total rate ``sum_tau mu_{j,tau}``.
        If False, compute type-specific ``p(g | tau)`` by normalizing
        context counts *per column* and mix with ``mu_{j,tau}``.

    Returns
    -------
    pandas.DataFrame | dict[int | str, pandas.DataFrame]
        **When mu_taus is a DataFrame:**
            Single DataFrame with Genes × Tumors. Index =
            ``ensembl_gene_id``, columns = tumor barcodes. Each cell
            is the expected mutation rate for that gene in that tumor.

        **When mu_taus is a dict:**
            Dictionary mapping signature identifiers to DataFrames,
            each shaped (genes × tumors), containing per-gene
            mutation rates attributable to that signature alone.

    Notes
    -----
    When ``prob_g_tau_tau_independent`` is True:
        ``out[g, j] = p(g) * sum_tau mu_{j,tau}``,
        where ``p(g) = opp_g / sum_g' opp_{g'}``.

    When False:
        ``out[g, j] = sum_tau mu_{j,tau} * p(g | tau)``,
        where ``p(g | tau)`` is the gene's share of opportunities for
        the trinucleotide underlying ``tau``.

    The function expects ``mu_taus`` columns (or dictionary values'
    columns) to match ``constants.canonical_types_order`` exactly.
    The ``contexts_by_gene`` columns must contain all contexts.

    See Also
    --------
    compute_mu_m_per_tumor : Per-variant expected rate per tumor.

    """
    # Check if mu_taus is a dictionary (signature-separated mode)
    if isinstance(mu_taus, dict):
        return {
            sigma: compute_mu_g_per_tumor(
                mu_taus=mu_tau_sigma,
                contexts_by_gene=contexts_by_gene,
                prob_g_tau_tau_independent=prob_g_tau_tau_independent,
            )
            for sigma, mu_tau_sigma in mu_taus.items()
        }

    # Original single-DataFrame logic
    if prob_g_tau_tau_independent:
        probs_g = contexts_by_gene.sum(axis=1) / np.sum(
            contexts_by_gene.values
        )

        mu_tumor = mu_taus.sum(axis=1)

        out = probs_g.to_frame(0).dot(mu_tumor.to_frame(0).T)

    else:
        from .constants import canonical_types_order
        from .constants import extract_context

        probs_g_context = contexts_by_gene / contexts_by_gene.sum(
            axis=0
        )

        probs_g_tau = probs_g_context[
            [extract_context(x) for x in canonical_types_order]
        ]
        probs_g_tau.columns = canonical_types_order

        out = probs_g_tau.dot(mu_taus[canonical_types_order].T)

    out.index.name = "ensembl_gene_id"

    return out


def compute_n_taus(contexts_by_gene_or_db):
    """Compute counts per mutation type from contexts or a MAF.

    Compute total counts for each mutation *type* either from a wide,
    per-context table (columns are canonical contexts, rows possibly
    genes) or from a long, MAF-like table (rows are mutations with a
    'type' column) as returned by
    :func:`load_maf_files.load_or_generate_compact_db`. Detection is
    automatic based on the input columns.

    Parameters
    ----------
    contexts_by_gene_or_db : pandas.DataFrame
        One of:
        (i) A wide table whose columns equal
            ``constants.canonical_contexts_order``. Rows represent
            genes (or groups), and values are counts per context.
        (ii) A MAF-like table with a column named ``'type'`` giving
             the mutation type for each row.

    Returns
    -------
    pandas.Series
        Counts per mutation type. For the wide form, the index is
        ``constants.canonical_types_order`` and respects that
        order. For the MAF-like form, the index contains the
        observed type labels.

    Notes
    -----
    The function distinguishes inputs by testing whether the set
    of columns matches
    `constants.canonical_contexts_order`. In the wide form it
    first sums counts across rows and then re-indexes/selects
    columns so the result aligns with
    `:const:constants.canonical_types_order`.

    """
    from .constants import canonical_contexts_order

    if set(canonical_contexts_order) == set(
        contexts_by_gene_or_db.columns
    ):
        # case where it is contexts_by_gene
        from .constants import canonical_types_order

        repeated_contexts = [
            f"{context[0]}{context[2]}{context[-1]}"
            for context in canonical_types_order
        ]

        counts_per_context = contexts_by_gene_or_db.sum(axis=0)

        counts_per_type = counts_per_context.loc[repeated_contexts]

        counts_per_type.index = canonical_types_order

    else:
        # if not, then it should come from a MAF with mutations
        counts_per_type = contexts_by_gene_or_db.groupby(
            "type"
        ).size()

    return counts_per_type


def compute_mus_per_gene_per_sample(
    db,
    base_mus: pd.DataFrame | dict[int | str, pd.DataFrame],
    cov_effect: dict | np.ndarray | Sequence[float] | None,
    cov_matrix: pd.DataFrame | None = None,
    restrict_to_passenger: bool = False,
    separate_mus_per_model: bool = False,
) -> pd.DataFrame | dict[tuple[str, ...], pd.DataFrame]:
    """Return per-gene, per-sample mutation rates.

    Scale the baseline `base_mus` by covariate effects when provided,
    otherwise return baseline rates restricted to the gene set of
    interest.

    Usage patterns
    --------------
    • No covariates:
        Set `cov_effect=None` to obtain the baseline rates filtered by
        passenger genes (if requested).

    • Signature independent, single model:
        Provide a 1D array ``[intercept, beta1, ...]`` or ``{'c': array}``.
        The order of coefficients after the intercept must match the
        column order of `cov_matrix`.

    • Signature independent, multiple models:
        Supply a dict mapping tuples of covariate names to coefficient
        vectors, e.g.: ``{('loglog1p_gtex',): [c0, c_gtex], ...}``.

    • Multi-signature:
        When `base_mus` is a dict, `cov_effect` should be:
          * 2D array ``(n_signatures, n_coeffs)``
          * ``{'c': 2D array}`` from :func:`estimate_covariates_effect`
          * ``{('cov',): 2D array}`` from
            :func:`estimate_all_cov_effects`

    Parameters
    ----------
    db : Any
        Handle passed to `filter_passenger_genes_ensembl(db)` if
        restricting.
    base_mus : pd.DataFrame | dict[int | str, pd.DataFrame]
        **Signature independent mode:**
            Genes × tumors baseline components. Index = Ensembl gene IDs.
        **Multi-signature mode:**
            Dict mapping signature identifiers to DataFrames (genes ×
            tumors).
    cov_effect : dict | numpy.ndarray | Sequence[float] | None
        Covariate effect(s). When *None*, no scaling is applied.
        **Formats**:
          * 1D array or ``{'c': 1D array}`` for signature independent
          * 2D array ``(n_sigs, n_coeffs)`` for multi-signature
          * ``{'c': 2D array}`` from :func:`estimate_covariates_effect`
          * ``{('cov',): 2D array, ...}`` from
            :func:`estimate_all_cov_effects`
          * Dict of tuple keys for multiple signature independent models
    cov_matrix : pd.DataFrame | None
        Genes × covariates values. Index must be Ensembl gene IDs. Must
        be provided when `cov_effect` is not *None*.
    restrict_to_passenger : bool
        If True, restrict to passenger genes via
        `filter_passenger_genes_ensembl(db)`.
    separate_mus_per_model : bool, default False
        If True and `cov_effect` is a dict of models (signature
        independent mode only), return a dictionary keyed by the
        model's covariate tuple, where each value is a genes×tumors
        DataFrame of scaled `mus` computed on the largest set of genes
        that have non-missing values for **all** covariates in that
        model. Models with no eligible genes are omitted.
        If False, then for each gene, the function selects the largest
        model for which all required covariates are present
        (non-NaN). Genes with no applicable model keep their
        `base_mus`.

    Returns
    -------
    pd.DataFrame or dict[tuple[str, ...], pd.DataFrame]
        - When `cov_effect` is a single model (array or {'c': array}),
          or when `separate_mus_per_model` is False: a single
          genes×tumors DataFrame with per-gene scaling chosen by the
          largest applicable model per gene.
        - When `cov_effect` is a dict **and** `separate_mus_per_model`
          is True: a dict mapping each model's covariate tuple to a
          genes×tumors DataFrame of scaled `mus` for the eligible
          genes of that model.
        - When `base_mus` is a dict (multi-signature): a single
          genes×tumors DataFrame with summed signature contributions.

    Notes
    -----
    Scaling is multiplicative: mus(g, t) = base_mus(g, t) * exp(eta(g)).
    For signature independent usage, ensure `cov_matrix` column order
    matches the coefficient order (after the intercept).
    In multi-signature mode, row i of `cov_effect` corresponds to
    signature i in `base_mus.keys()` order.

    """
    # ──────── Multi-signature mode ────────
    if isinstance(base_mus, dict):
        signatures = list(base_mus.keys())

        if cov_effect is None:
            # No covariates: just sum baselines and filter
            base_mus_summed = sum(base_mus.values())
            if restrict_to_passenger:
                ids_pass = filter_passenger_genes_ensembl(db)
                return base_mus_summed.loc[
                    base_mus_summed.index.intersection(ids_pass)
                ]
            return base_mus_summed

        # Check if cov_effect is from estimate_all_cov_effects with multiple models
        if isinstance(cov_effect, dict):
            tuple_keys = [
                k for k in cov_effect.keys() if isinstance(k, tuple)
            ]

            # Multiple models case: separate_mus_per_model must be True
            if len(tuple_keys) > 1 or (
                len(tuple_keys) == 1 and separate_mus_per_model
            ):
                if not separate_mus_per_model:
                    raise ValueError(
                        "Multi-signature mode with multiple models in cov_effect "
                        "requires separate_mus_per_model=True"
                    )

                # Process each model separately
                results = {}
                for covs_tuple in tuple_keys:
                    c_model = np.asarray(
                        cov_effect[covs_tuple], dtype=np.float32
                    )

                    # Validate shape
                    if c_model.ndim != 2:
                        raise ValueError(
                            f"Model {covs_tuple}: expected 2D array, got shape {c_model.shape}"
                        )
                    if c_model.shape[0] != len(signatures):
                        raise ValueError(
                            f"Model {covs_tuple}: has {c_model.shape[0]} rows but "
                            f"base_mus has {len(signatures)} signatures"
                        )

                    # Get gene set for this model
                    first_df = next(iter(base_mus.values()))
                    if restrict_to_passenger:
                        ids_pass = filter_passenger_genes_ensembl(db)
                    else:
                        ids_pass = pd.Index(first_df.index)

                    # Filter to genes with non-missing covariates
                    cov_subset = cov_matrix.loc[:, list(covs_tuple)]
                    ids = first_df.index.intersection(
                        ids_pass
                    ).intersection(cov_subset.index)
                    ids = ids[cov_subset.loc[ids].notna().all(axis=1)]

                    # Ensure all signatures have same genes
                    for sigma, df in base_mus.items():
                        ids = ids.intersection(df.index)

                    if len(ids) == 0:
                        continue  # Skip models with no eligible genes

                    # Build design matrix for this model
                    cov_df = cov_subset.loc[ids]
                    X_cov = cov_df.to_numpy(dtype=np.float32)
                    ones = np.ones(
                        (X_cov.shape[0], 1), dtype=np.float32
                    )
                    X = np.concatenate([ones, X_cov], axis=1)

                    if c_model.shape[1] != X.shape[1]:
                        raise ValueError(
                            f"Model {covs_tuple}: has {c_model.shape[1]} coefficients "
                            f"but needs {X.shape[1]} (including intercept)"
                        )

                    # Compute signature-specific scaling and sum
                    eta = X @ c_model.T  # (n_genes, n_signatures)
                    scale = np.exp(eta).astype(np.float32)

                    mus_full = None
                    for s_idx, sigma in enumerate(signatures):
                        mus_sigma = base_mus[sigma].loc[ids]
                        mus_scaled = mus_sigma.mul(
                            scale[:, s_idx], axis=0
                        )

                        if mus_full is None:
                            mus_full = mus_scaled
                        else:
                            mus_full = mus_full + mus_scaled

                    results[covs_tuple] = mus_full

                return results

            # Single model from dict: extract coefficient array
            if "c" in cov_effect:
                c = np.asarray(cov_effect["c"], dtype=np.float32)
            elif tuple_keys:
                c = np.asarray(
                    cov_effect[tuple_keys[0]], dtype=np.float32
                )
            else:
                raise ValueError(
                    "Multi-signature mode with dict cov_effect "
                    "requires either 'c' key or tuple keys"
                )
        else:
            c = np.asarray(cov_effect, dtype=np.float32)

        # Validate it's 2D
        if c.ndim != 2:
            raise ValueError(
                "In multi-signature mode, cov_effect must be a 2D array "
                f"with shape (n_signatures, n_coeffs), got shape {c.shape}"
            )

        if c.shape[0] != len(signatures):
            raise ValueError(
                f"cov_effect has {c.shape[0]} rows but base_mus has "
                f"{len(signatures)} signatures"
            )

        if cov_matrix is None:
            raise ValueError(
                "cov_matrix must be provided when cov_effect is not None"
            )

        # Get gene set
        first_df = next(iter(base_mus.values()))
        if restrict_to_passenger:
            ids_pass = filter_passenger_genes_ensembl(db)
        else:
            ids_pass = pd.Index(first_df.index)

        ids = first_df.index.intersection(ids_pass).intersection(
            cov_matrix.index
        )

        # Ensure all signatures have same genes
        for sigma, df in base_mus.items():
            ids = ids.intersection(df.index)

        if len(ids) == 0:
            raise ValueError(
                "No overlapping genes between base_mus signatures, "
                "cov_matrix, and passenger set"
            )

        # Build design matrix
        cov_df = cov_matrix.loc[ids]
        X_cov = cov_df.to_numpy(dtype=np.float32)
        ones = np.ones((X_cov.shape[0], 1), dtype=np.float32)
        X = np.concatenate(
            [ones, X_cov], axis=1
        )  # (n_genes, n_coeffs)

        if c.shape[1] != X.shape[1]:
            raise ValueError(
                f"cov_effect has {c.shape[1]} coefficients but "
                f"cov_matrix has {X.shape[1]} (including intercept)"
            )

        # Compute signature-specific scaling and sum
        # eta[s, g] = X[g, :] @ c[s, :]
        eta = X @ c.T  # (n_genes, n_signatures)
        scale = np.exp(eta).astype(
            np.float32
        )  # (n_genes, n_signatures)

        # Sum: mus_full[g, t] = sum_s(base_mus[s][g, t] * scale[g, s])
        mus_full = None
        for s_idx, sigma in enumerate(signatures):
            mus_sigma = base_mus[sigma].loc[ids]
            # scale[:, s_idx] is (n_genes,), broadcast over tumors
            mus_scaled = mus_sigma.mul(scale[:, s_idx], axis=0)

            if mus_full is None:
                mus_full = mus_scaled
            else:
                mus_full = mus_full + mus_scaled

        return mus_full

    # ──────── Signature independent mode (original logic) ────────
    # --- choose gene set ---
    if restrict_to_passenger:
        ids_pass = filter_passenger_genes_ensembl(db)
    else:
        ids_pass = pd.Index(base_mus.index)

    ids = base_mus.index.intersection(ids_pass)

    if cov_effect is None:
        if len(ids) == 0:
            raise ValueError(
                "No overlapping genes between base_mus "
                "and passenger set."
            )
        return base_mus.loc[ids]

    if cov_matrix is None:
        raise ValueError(
            "cov_matrix must be provided when cov_effect is not None."
        )

    ids = ids.intersection(cov_matrix.index)
    if len(ids) == 0:
        raise ValueError(
            "No overlapping genes between "
            "base_mus, cov_matrix, and passenger set."
        )

    mus = base_mus.loc[ids]
    cov_df = cov_matrix.loc[ids]

    # ---------- single-model path ----------
    if not isinstance(cov_effect, dict) or (
        isinstance(cov_effect, dict)
        and set(cov_effect.keys()) == {"c"}
    ):
        if isinstance(cov_effect, dict):
            c = np.asarray(cov_effect["c"], dtype=np.float32)
        else:
            c = np.asarray(cov_effect, dtype=np.float32)

        X_cov = cov_df.to_numpy(dtype=np.float32)
        ones = np.ones((X_cov.shape[0], 1), dtype=np.float32)
        X = np.concatenate([ones, X_cov], axis=1)

        if c.ndim != 1 or c.shape[0] != X.shape[1]:
            raise ValueError(
                "Length of cov_effect does not match cov_matrix "
                f"(got {c.shape[0]} vs {X.shape[1]})."
            )

        eta = X @ c
        scale = np.exp(eta).astype(np.float32)
        mus_full = mus.mul(scale, axis=0)
        return mus_full

    # ---------- multi-model path ----------
    # Normalize models and validate coefficients
    models: list[tuple[tuple[str, ...], np.ndarray]] = []
    for key, coef in cov_effect.items():
        if key == "c":
            continue
        covs = (key,) if isinstance(key, str) else tuple(key)
        coef = np.asarray(coef, dtype=np.float32)
        if coef.ndim != 1 or coef.shape[0] != 1 + len(covs):
            raise ValueError(
                "Coefficient length mismatch for model "
                f"{covs}: expected {1 + len(covs)}, "
                f"got {coef.shape[0]}."
            )
        if any(cov not in cov_df.columns for cov in covs):
            # skip models referring to covariates not present at all
            continue
        models.append((covs, coef))

    if not models:
        # No usable models: return baseline mus unchanged
        return mus.copy()

    # Prefer larger models; tie-break lexicographically for determinism
    models.sort(key=lambda x: (-len(x[0]), tuple(x[0])))

    if separate_mus_per_model:
        out: dict[tuple[str, ...], pd.DataFrame] = {}
        for covs, coef in models:
            need = list(covs)
            elig = cov_df[need].notna().all(axis=1)
            if not elig.any():
                continue
            X_cov = cov_df.loc[elig, need].to_numpy(dtype=np.float32)
            ones = np.ones((X_cov.shape[0], 1), dtype=np.float32)
            X = np.concatenate([ones, X_cov], axis=1)
            eta = X @ coef
            scale = np.exp(eta).astype(np.float32)
            idx = mus.index[elig]
            scale_s = pd.Series(scale, index=idx)
            out[covs] = mus.loc[idx].mul(scale_s, axis=0)
        return out
    else:
        mus_full = mus.copy()
        assigned = pd.Series(False, index=mus.index)

        for covs, coef in models:
            need = list(covs)
            elig = cov_df[need].notna().all(axis=1) & (~assigned)
            if not elig.any():
                continue

            X_cov = cov_df.loc[elig, need].to_numpy(dtype=np.float32)
            ones = np.ones((X_cov.shape[0], 1), dtype=np.float32)
            X = np.concatenate([ones, X_cov], axis=1)
            eta = X @ coef
            scale = np.exp(eta).astype(np.float32)

            mus_full.loc[elig] = mus_full.loc[elig].mul(scale, axis=0)
            assigned.loc[elig] = True

        # Unassigned genes keep baseline mus
        return mus_full


def compute_mu_m_per_tumor(
    variants_df: pd.DataFrame,
    mu_g_j: pd.DataFrame | dict[int | str, pd.DataFrame],
    contexts_by_gene: pd.DataFrame,
    prob_g_tau_tau_independent=False,
) -> pd.DataFrame:
    """Compute per-variant expected mutation rate per tumor.

    Transform gene-level mutation rates into variant-level rates by
    accounting for the number of mutational opportunities (contexts)
    each variant has within its gene.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        Variant table with columns:
        - 'mut_types': mutation type(s) in COSMIC format (e.g.,
          'G[C>T]G' or list of types for multi-type variants)
        - 'ensembl_gene_id': Ensembl gene identifier
        - 'gene': gene symbol
        Index should be variant identifiers (e.g., 'KRAS p.G12D').

    mu_g_j : pd.DataFrame or dict[int | str, pd.DataFrame]
        Genes × tumors matrix of mutation rates, or dictionary of
        such matrices (one per signature in multi-signature mode).
        When dict: mutation rates are summed across all signatures.
        Index: Ensembl gene IDs (e.g., 'ENSG00000133703').
        Columns: tumor barcodes (e.g., 'TCGA-5M-A8F6-...').
        Values: expected mutation rate for each gene in each tumor.

    contexts_by_gene : pandas.DataFrame
        Gene-level trinucleotide context counts.
        Index: Ensembl gene IDs.
        Columns: 32 trinucleotide contexts (e.g., 'ACA', 'GCG').
        Values: count of each context in the gene's coding sequence.

    prob_g_tau_tau_independent : bool, default False
        If True, assume context type distribution is independent of
        gene identity: reconstructs context counts by redistributing
        each gene's total contexts according to genome-wide context
        proportions. Setting to True is useful when we believe that
        the data of gene-specific context counts may not reflect the
        mutation data calling. If False (default), uses gene-specific
        context counts for each mutation type. This setting should
        reflect that one used to compute the mu_g_j, when using
        :func:`compute_mu_g_per_tumor`

    Returns
    -------
    pandas.DataFrame
        Variants × tumors matrix.
        Index: variant identifiers from ``variants_df.index``.
        Columns: tumor barcodes (same as ``mu_g_j.columns``).
        Values: expected mutation rate for each variant in each tumor.

    Notes
    -----
    The calculation proceeds in three steps:

    1. Extract trinucleotide contexts from mutation types:
       'G[C>T]G' → 'GCG'

    2. Count context opportunities per variant:

       **When prob_g_tau_tau_independent=True:**
           First, reconstruct ``contexts_by_gene`` by redistributing
           each gene's total contexts according to genome-wide
           proportions:

           ``contexts_by_gene'[g, τ] = (Σ_τ contexts[g, τ]) × (Σ_g contexts[g, τ] / Σ_g,τ contexts[g, τ])``

           Then compute ``n_contexts[m]`` from the reconstructed
           matrix using the variant's specific context type(s). This
           assumes all genes have the same relative distribution of
           context types.

       **When prob_g_tau_tau_independent=False (default):**
           For variant m in gene g with context(s) τ:
           ``n_contexts[m] = sum_τ contexts_by_gene[g, τ]``
           Uses gene-specific context counts matching the mutation
           type(s).

    3. Normalize gene rates to variant rates:
       ``μ_{m,j} = μ_{g,j} / n_contexts[m]``

    For variants with multiple mutation types (rare), context counts
    are summed across all relevant types.

    See Also
    --------
    compute_mu_g_per_tumor : Per-gene expected rate per tumor.

    """
    logger.info("Computing per-variant mutation rates per tumor...")

    # Handle multi-signature mode: sum across all signatures
    if isinstance(mu_g_j, dict):
        logger.info(
            "Multi-signature mode detected: summing mutation rates "
            f"across {len(mu_g_j)} signatures..."
        )
        # Sum all signature-specific DataFrames
        mu_g_j = sum(mu_g_j.values())

    from .constants import extract_context

    variants = variants_df.copy()

    def extract_contexts(mut_types):
        """Extract context(s) from mut_types (string or list)."""
        if isinstance(mut_types, str):
            return extract_context(mut_types)
        elif isinstance(mut_types, list):
            return [extract_context(t) for t in mut_types]
        else:
            return None

    variants["contexts"] = variants["mut_types"].apply(
        extract_contexts
    )

    # When prob_g_tau_tau_independent=True, reconstruct contexts_by_gene
    # by redistributing each gene's total contexts according to
    # genome-wide proportions. This assumes the distribution of context
    # types is the same across all genes, useful when gene-specific
    # context counts may not reflect what mutation calling scanned.
    if prob_g_tau_tau_independent:
        contexts_by_gene = (
            pd.DataFrame(contexts_by_gene.sum(axis=1))
            @ pd.DataFrame(
                contexts_by_gene.sum(axis=0)
                / contexts_by_gene.values.sum()
            ).T
        )

    # Pre-compute sets for O(1) membership checks
    valid_genes = set(contexts_by_gene.index)
    valid_contexts = set(contexts_by_gene.columns)

    # Separate single-context from multi-context variants
    is_string = variants["contexts"].apply(
        lambda x: isinstance(x, str)
    )
    is_list = variants["contexts"].apply(
        lambda x: isinstance(x, list)
    )

    # Initialize all with floating zeros to allow fractional counts
    variants["n_contexts"] = 0.0

    # Handle single-context variants (most common case)
    single_mask = is_string
    if single_mask.any():
        single_vars = variants[single_mask].copy()

        # Use pandas get() method for safe lookups
        def lookup_single(row):
            gene_id = row["ensembl_gene_id"]
            ctx = row["contexts"]
            if gene_id in valid_genes and ctx in valid_contexts:
                return contexts_by_gene.at[gene_id, ctx]
            return 0

        variants.loc[single_mask, "n_contexts"] = single_vars.apply(
            lookup_single, axis=1
        )

    # Handle multi-context variants (rare)
    if is_list.any():

        def sum_multi_contexts(row):
            gene_id = row["ensembl_gene_id"]
            if gene_id not in valid_genes:
                return 0

            contexts = row["contexts"]
            return sum(
                contexts_by_gene.at[gene_id, ctx]
                for ctx in contexts
                if ctx in valid_contexts
            )

        variants.loc[is_list, "n_contexts"] = variants[is_list].apply(
            sum_multi_contexts, axis=1
        )

    # Build variants × tumors matrix using vectorized operations
    # Reindex mu_g_j by the gene_ids from variants to align rows
    gene_ids = variants["ensembl_gene_id"]
    mu_genes_aligned = mu_g_j.reindex(gene_ids)
    mu_genes_aligned.index = variants.index

    # Divide by n_contexts (broadcasting across columns)
    n_contexts = variants["n_contexts"].replace(0, np.nan)
    mu_m_j = mu_genes_aligned.div(n_contexts, axis=0)

    # Fill NaN with 0 (for missing genes or zero contexts)
    mu_m_j = mu_m_j.fillna(0.0)

    logger.info("... done with per-variant mutation rates per tumor.")
    return mu_m_j
