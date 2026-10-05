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
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .estimate_presence import filter_passenger_genes_ensembl

logger = logging.getLogger(__name__)


def covariate_complete_genes(cov_matrix):
    """Genes whose covariate row is complete (no NaN).

    Only these can be covariate-scaled. Everything else in a baseline
    falls back -- see :func:`compute_mus_per_gene_per_sample`'s
    ``fallback_log_scale``.
    """
    return cov_matrix.index[cov_matrix.notna().all(axis=1)]


def _with_fallback(scaled, baseline, ids_all, fallback_log_scale):
    """Add the genes of ``ids_all`` missing from ``scaled`` at their
    fallback rate, ``baseline * exp(fallback_log_scale)``, and return
    the rows in ``ids_all``'s order. ``fallback_log_scale`` is one
    number, or a per-gene ``pd.Series`` (a gene it lacks gets 0).
    """
    missing = ids_all.difference(scaled.index)
    if len(missing):
        rows = baseline.loc[missing]
        if isinstance(fallback_log_scale, pd.Series):
            factor = np.exp(
                fallback_log_scale.reindex(missing).fillna(0.0)
            ).astype(np.asarray(rows.values).dtype)
            rows = rows.mul(factor, axis=0)
        else:
            # A Python float, so a float32 baseline stays float32.
            rows = rows * float(np.exp(fallback_log_scale))
        scaled = pd.concat([scaled, rows])
    return scaled.loc[ids_all]


def _floor_alphas(
    alphas,
    sig_matrix,
    full_matrix,
    assignments,
    signature,
    kappa,
    scope,
):
    """Mix ``kappa`` pseudo-mutations of ``signature`` into exposures.

    See `compute_mu_tau_per_tumor`'s ``floor_signature``. Returns the
    floored exposures and the signature matrix extended with
    ``signature`` if the fit did not use it.
    """
    if scope not in ("all", "zero_types"):
        raise ValueError(
            f"floor_scope must be 'all' or 'zero_types', not {scope!r}"
        )
    if signature not in full_matrix.columns:
        raise ValueError(
            f"floor signature {signature!r} is not in the signature matrix"
        )
    floor = full_matrix[signature]
    if (floor <= 0).any():
        raise ValueError(
            f"floor signature {signature!r} has types with probability 0; "
            "it cannot guarantee every type a positive rate"
        )
    if signature not in sig_matrix.columns:
        sig_matrix = sig_matrix.join(floor)
        alphas = alphas.assign(**{signature: 0.0})
    n = (
        assignments.reindex(index=alphas.index)
        .sum(axis=1)
        .fillna(0.0)
        .astype(float)
    )
    rows = pd.Series(True, index=alphas.index)
    if scope == "zero_types":
        spectrum = alphas.to_numpy() @ sig_matrix.to_numpy().T
        rows = pd.Series(
            (spectrum <= 0).any(axis=1), index=alphas.index
        )
    e = pd.Series(0.0, index=alphas.columns)
    e[signature] = 1.0
    floored = (alphas.mul(n, axis=0) + kappa * e).div(
        n + kappa, axis=0
    )
    alphas = alphas.where(~rows, floored, axis=0)
    logger.info(
        "Exposure floor: %g pseudo-mutation(s) of %s in %d of %d tumors "
        "(scope %s); median weight %.3g",
        kappa,
        signature,
        int(rows.sum()),
        len(rows),
        scope,
        (
            float((kappa / (n[rows] + kappa)).median())
            if rows.any()
            else 0.0
        ),
    )
    return alphas, sig_matrix


def _census_genes():
    """Cancer Gene Census symbols (the drivers left out of the pool)."""
    from .locations import location_cancer_gene_census

    census = pd.read_csv(location_cancer_gene_census, sep="\t")
    return set(census["Gene Symbol"].dropna())


def _floored_spectra(
    db,
    alphas,
    sig_matrix,
    full_matrix,
    assignments,
    ell_hats,
    signature,
    kappa,
    scope,
    pooled_share,
):
    """Per-type rates with the spectrum floor of a pooled share.

    See `compute_mu_tau_per_tumor`'s ``floor_pooled_share``.
    """
    from .channel_universe import model_calls

    if scope not in ("all", "zero_types"):
        raise ValueError(
            f"floor_scope must be 'all' or 'zero_types', not {scope!r}"
        )
    if signature not in full_matrix.columns:
        raise ValueError(
            f"floor signature {signature!r} is not in the signature matrix"
        )
    types = sig_matrix.index
    floor = full_matrix[signature].reindex(types)
    if (floor <= 0).any():
        raise ValueError(
            f"floor signature {signature!r} has types with probability 0; "
            "it cannot guarantee every type a positive rate"
        )
    # Passenger genes only: a recurrent driver's calls would otherwise
    # inflate their own type in the pool -- thyroid BRAF V600E is 5.9%
    # of its cohort's calls, all G[T>A]G -- and the floor would raise
    # that driver's rate in every tumor, letting selection into the
    # rate model.
    calls = model_calls(db)
    calls = calls[~calls["gene"].isin(_census_genes())]
    counts = calls["type"].value_counts().reindex(types, fill_value=0)
    pooled = counts / counts.sum() if counts.sum() else floor * 0
    e = pooled_share * pooled + (1 - pooled_share) * floor
    p = alphas.dot(sig_matrix.T)
    n = (
        assignments.reindex(index=alphas.index)
        .sum(axis=1)
        .fillna(0.0)
        .astype(float)
    )
    rows = pd.Series(True, index=p.index)
    if scope == "zero_types":
        rows = (p <= 0).any(axis=1)
    floored = (
        p.mul(n, axis=0).add(kappa * e, axis=1).div(n + kappa, axis=0)
    )
    p = p.where(~rows, floored, axis=0)
    logger.info(
        "Spectrum floor: %g pseudo-mutation(s), %.0f%% the cohort's "
        "pooled spectrum and %.0f%% %s, in %d of %d tumors (scope %s)",
        kappa,
        100 * pooled_share,
        100 * (1 - pooled_share),
        signature,
        int(rows.sum()),
        len(rows),
        scope,
    )
    return p.multiply(ell_hats, axis=0)


def compute_mu_tau_per_tumor(
    db,
    location_signature_matrix,
    assignments,
    L_low=None,
    L_high=None,
    cut_at_L_low=False,
    separate_per_sigma=False,
    floor_signature=None,
    floor_pseudocount=0.0,
    floor_scope="all",
    floor_pooled_share=0.0,
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
    floor_signature : str or None, default None
        A signature mixed into every tumor's exposures as
        ``floor_pseudocount`` pseudo-mutations:
        ``alpha' = (n alpha + kappa e) / (n + kappa)``, with ``n`` the
        tumor's fitted mutation count (its assignment total) and ``e``
        the unit vector of ``floor_signature``. A fit on a few
        mutations can land on signatures that give some type
        probability 0 -- e.g. SBS84 alone, which cannot emit any
        T>A -- and then a mutation the tumor carries has rate 0. A
        signature with no zero types (SBS5) guarantees every type a
        positive rate; its weight ``kappa / (n + kappa)`` vanishes
        with burden. None (default) applies no floor.
    floor_pseudocount : float, default 0.0
        ``kappa`` above. 0 applies no floor.
    floor_scope : {"all", "zero_types"}, default "all"
        Which tumors get the floor: every tumor, or only those whose
        unfloored spectrum gives some type probability 0.
    floor_pooled_share : float, default 0.0
        Share of the cohort's pooled spectrum (its observed type
        frequencies over every tumor in ``db``, passenger genes only:
        Cancer Gene Census genes are left out, or a recurrent driver
        would raise its own rate) in what the
        pseudo-mutations are drawn from; the rest is
        ``floor_signature``. The floor then acts on the spectrum, ``p'
        = (n p + kappa e) / (n + kappa)``, ``e = share * pooled + (1 -
        share) * floor_signature`` -- with share 0 the same as the
        signature floor above. Must be below 1, so ``e`` keeps every
        type positive. Held out, a pooled spectrum with a 10% SBS5
        share predicted tumors' unseen mutations best; it shrinks
        every tumor's spectrum toward its cohort's. Not available
        with ``separate_per_sigma``.
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

    if (
        floor_signature is not None
        and floor_pseudocount > 0
        and floor_pooled_share > 0
    ):
        if separate_per_sigma:
            raise ValueError(
                "floor_pooled_share acts on the spectrum, not per "
                "signature; it cannot be combined with separate_per_sigma"
            )
        if not 0 < floor_pooled_share < 1:
            raise ValueError("floor_pooled_share must be in [0, 1)")
        return _floored_spectra(
            db,
            alphas,
            sig_matrix,
            load_signature_matrix(location_signature_matrix),
            assignments,
            ell_hats,
            floor_signature,
            floor_pseudocount,
            floor_scope,
            floor_pooled_share,
        )

    if floor_signature is not None and floor_pseudocount > 0:
        alphas, sig_matrix = _floor_alphas(
            alphas,
            sig_matrix,
            load_signature_matrix(location_signature_matrix),
            assignments,
            floor_signature,
            floor_pseudocount,
            floor_scope,
        )

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
    separate_per_tau=False,
    type_opportunity=None,
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
    separate_per_tau : bool | Sequence[str], default False
        If True, do not sum over mutation types τ: return the
        per-type gene rate ``mu_{g,tau}^{j}`` needed by
        :func:`compute_mu_m_per_tumor` to correctly derive
        variant-level rates. See Returns.
        Materializing all 96 types at once is O(genes × tumors ×
        96) (× n_signatures in dict mode), which can exhaust memory
        for large gene/tumor counts. Pass a subset of
        ``constants.canonical_types_order`` (e.g. a single τ) to
        compute just those types and bound memory --- the intended
        use is to loop over types one (or a few) at a time.
    type_opportunity : pandas.DataFrame or None, default None
        Genes x 96 canonical types: every opportunity of each type,
        all channels together (the sum of the channel tables). Needed
        once a germline mask has removed single alternate bases, when
        a type's opportunity is no longer its context's position
        count; it then replaces ``contexts_by_gene`` in both the
        numerator and the denominator of ``p(g | tau)``. None keeps
        the context-based form, which is identical when nothing is
        masked.

    Returns
    -------
    pandas.DataFrame | dict[int | str, pandas.DataFrame]
        **When mu_taus is a DataFrame and separate_per_tau=False:**
            Single DataFrame with Genes × Tumors. Index =
            ``ensembl_gene_id``, columns = tumor barcodes. Each entry
            is the expected mutation rate for that gene in that tumor.

        **When mu_taus is a dict and separate_per_tau=False:**
            Dictionary mapping signature identifiers to DataFrames,
            each shaped (genes × tumors), containing per-gene
            mutation rates attributable to that signature alone.

        **When mu_taus is a DataFrame and separate_per_tau=True:**
            Dictionary mapping each mutation type τ (from
            ``constants.canonical_types_order``) to a Genes × Tumors
            DataFrame of ``mu_{g,tau}^{j}``, i.e. the type-τ share of
            gene g's rate --- *not* summed over τ.

        **When mu_taus is a dict and separate_per_tau=True:**
            Nested dictionary: signature -> {τ: Genes × Tumors
            DataFrame}, i.e. the per-signature version of the above.

    Notes
    -----
    When ``prob_g_tau_tau_independent`` is True:
        ``out[g, j] = p(g) * sum_tau mu_{j,tau}``,
        where ``p(g) = opp_g / sum_g' opp_{g'}``.

    When False:
        ``out[g, j] = sum_tau mu_{j,tau} * p(g | tau)``,
        where ``p(g | tau)`` is the gene's share of opportunities for
        the trinucleotide underlying ``tau``.

    With ``separate_per_tau=True``, the summand ``mu_{j,tau} * p(g |
    tau)`` (or ``mu_{j,tau} * p(g)`` in the τ-independent case) is
    returned per τ instead of summed --- this is ``mu_{g,tau}^{j}``,
    the quantity :func:`compute_mu_m_per_tumor` divides by
    ``n_{g,c(tau)}`` to get the per-variant rate. Summing the
    per-τ dict over τ recovers the ``separate_per_tau=False`` result.

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
                separate_per_tau=separate_per_tau,
                type_opportunity=type_opportunity,
            )
            for sigma, mu_tau_sigma in mu_taus.items()
        }

    from .constants import canonical_types_order, extract_context

    tau_list = (
        canonical_types_order
        if separate_per_tau is True
        else list(separate_per_tau) if separate_per_tau else None
    )

    if type_opportunity is not None:
        type_opportunity = type_opportunity.reindex(
            index=contexts_by_gene.index,
            columns=canonical_types_order,
        ).fillna(0.0)

    # Original single-DataFrame logic
    if prob_g_tau_tau_independent:
        if type_opportunity is not None:
            probs_g = type_opportunity.sum(axis=1) / np.sum(
                type_opportunity.values
            )
        else:
            probs_g = contexts_by_gene.sum(axis=1) / np.sum(
                contexts_by_gene.values
            )

        if tau_list is not None:
            out = {
                tau: probs_g.to_frame(0).dot(
                    mu_taus[tau].to_frame(0).T
                )
                for tau in tau_list
            }
        else:
            mu_tumor = mu_taus.sum(axis=1)
            out = probs_g.to_frame(0).dot(mu_tumor.to_frame(0).T)

    else:
        if type_opportunity is not None:
            probs_g_tau = type_opportunity / type_opportunity.sum(
                axis=0
            )
        else:
            probs_g_context = contexts_by_gene / contexts_by_gene.sum(
                axis=0
            )
            probs_g_tau = probs_g_context[
                [extract_context(x) for x in canonical_types_order]
            ]
            probs_g_tau.columns = canonical_types_order

        if tau_list is not None:
            out = {
                tau: probs_g_tau[tau]
                .to_frame(0)
                .dot(mu_taus[tau].to_frame(0).T)
                for tau in tau_list
            }
        else:
            out = probs_g_tau.dot(mu_taus[canonical_types_order].T)

    if tau_list is not None:
        for tau_out in out.values():
            tau_out.index.name = "ensembl_gene_id"
    else:
        out.index.name = "ensembl_gene_id"

    return out


def type_denominators(contexts_by_gene, type_opportunity=None):
    """``D_tau``: the per-type opportunity over the gene universe.

    The denominator of ``p_{g tau}`` in the tau-dependent channel rates
    (:func:`compute_mu_g_channel_per_tumor`): the column sums of
    ``type_opportunity`` (genes x 96, all channels, after any germline
    mask and site weights) over ``contexts_by_gene``'s genes, or, when
    it is None, each type's context position count.

    Returns
    -------
    pandas.Series
        Indexed by ``constants.canonical_types_order``.
    """
    from .constants import canonical_types_order, extract_context

    if type_opportunity is not None:
        return (
            type_opportunity.reindex(
                index=contexts_by_gene.index,
                columns=canonical_types_order,
            )
            .fillna(0.0)
            .sum(axis=0)
        )
    out = contexts_by_gene.sum(axis=0)[
        [extract_context(x) for x in canonical_types_order]
    ]
    out.index = canonical_types_order
    return out


def compute_mu_g_channel_per_tumor(
    mu_taus: pd.DataFrame | dict[int | str, pd.DataFrame],
    channel_contexts_by_gene: pd.DataFrame,
    contexts_by_gene: pd.DataFrame,
    prob_g_tau_tau_independent: bool = False,
    separate_per_tau: bool = False,
    type_opportunity: pd.DataFrame | None = None,
) -> pd.DataFrame | dict[int | str, pd.DataFrame]:
    """Per-gene rate for one consequence channel (syn or non-syn).

    The consequence-channel counterpart of
    :func:`compute_mu_g_per_tumor`: identical in every respect except
    that the *numerator* of ``p_gτ`` comes from a consequence-split
    opportunity table (genes × 96 SBS types, from
    :func:`consequence_contexts_by_gene.compute_consequence_contexts_by_gene`)
    while the **denominator stays the full τ-site count** taken from
    ``contexts_by_gene``.

    That denominator is the whole point, and getting it wrong is the
    easiest way to break this: ``p_gτ^(syn)`` has to match what it
    multiplies (``μ̄_τ^j``, the *total* type-τ rate), so it does not
    sum to 1 over genes -- it sums to the genome-wide synonymous
    fraction (≈0.23). Normalising by the synonymous column sums
    instead would overcount by ≈4×.

    Because the two channels' numerators add up to
    ``contexts_by_gene`` exactly (guaranteed by construction, see
    :mod:`consequence_contexts_by_gene`) and the denominator is
    shared, this function satisfies::

        compute_mu_g_channel_per_tumor(mu_taus, syn, contexts, ...)
        + compute_mu_g_channel_per_tumor(mu_taus, nonsyn, contexts, ...)
        == compute_mu_g_per_tumor(mu_taus, contexts, ...)

    for both ``prob_g_tau_tau_independent`` settings, which is the
    ``μ_g = μ_g^(syn) + μ_g^(nonsyn)`` identity the channel-split
    model rests on. Tested directly in
    ``tests/test_channel_cov_effects.py``.

    Parameters
    ----------
    mu_taus : pandas.DataFrame | dict[int | str, pandas.DataFrame]
        Tumors × 96 SBS types, or a per-signature dict of such
        frames. Same as :func:`compute_mu_g_per_tumor`.
    channel_contexts_by_gene : pandas.DataFrame
        Genes × 96 canonical SBS types: this channel's share of the
        opportunities. Either output of
        :func:`consequence_contexts_by_gene.compute_consequence_contexts_by_gene`.
    contexts_by_gene : pandas.DataFrame
        Genes × 32 contexts, the *full* opportunity table -- used
        only for the denominator, never the numerator. Must cover
        exactly the same genes as ``channel_contexts_by_gene``.
    prob_g_tau_tau_independent : bool, default False
        As in :func:`compute_mu_g_per_tumor`. **Leave this at
        False for channel work.** ``True`` averages the synonymous
        fraction over τ, which makes
        ``μ̄_g^(syn)/μ̄_g^(nonsyn)`` a per-gene constant that no
        longer responds to a sample's mutational spectrum -- see
        this module's note in ``DEVELOPMENT.md``. It stays
        available so the two channels can still be built to match
        a τ-independent merged baseline.
    separate_per_tau : bool | Sequence[str], default False
        As in :func:`compute_mu_g_per_tumor`.
    type_opportunity : pandas.DataFrame or None, default None
        As in :func:`compute_mu_g_per_tumor`: when given, the
        denominator of ``p_gtau`` is its column sum, the masked
        opportunity of each type, instead of the position count.

    Returns
    -------
    pandas.DataFrame | dict
        Same shapes and modes as :func:`compute_mu_g_per_tumor`.

    Notes
    -----
    The two paths normalise differently, and the asymmetry is real
    rather than an oversight:

    * τ-**dependent**: for a *given* type τ, a position offers exactly
      one opportunity, so the per-type denominator
      ``Σ_g' contexts[g', c(τ)]`` is already in opportunity units --
      it is used unchanged.
    * τ-**independent**: the aggregate collapses over types, and each
      position offers **3** opportunities (one per alternate base),
      while ``contexts_by_gene`` counts positions. The denominator is
      therefore ``3 × contexts_by_gene.values.sum()``, which equals
      the full table's total opportunity count exactly.
    """
    if isinstance(mu_taus, dict):
        return {
            sigma: compute_mu_g_channel_per_tumor(
                mu_taus=mu_tau_sigma,
                channel_contexts_by_gene=channel_contexts_by_gene,
                contexts_by_gene=contexts_by_gene,
                prob_g_tau_tau_independent=prob_g_tau_tau_independent,
                separate_per_tau=separate_per_tau,
                type_opportunity=type_opportunity,
            )
            for sigma, mu_tau_sigma in mu_taus.items()
        }

    from .constants import canonical_types_order

    if set(channel_contexts_by_gene.index) != set(
        contexts_by_gene.index
    ):
        raise ValueError(
            "channel_contexts_by_gene and contexts_by_gene must "
            "cover the same genes -- the denominator is a sum over "
            "genes, so a mismatched gene universe silently rescales "
            "every rate."
        )

    channel = channel_contexts_by_gene.loc[contexts_by_gene.index]

    tau_list = (
        canonical_types_order
        if separate_per_tau is True
        else list(separate_per_tau) if separate_per_tau else None
    )

    if type_opportunity is not None:
        type_opportunity = type_opportunity.reindex(
            index=contexts_by_gene.index,
            columns=canonical_types_order,
        ).fillna(0.0)

    if prob_g_tau_tau_independent:
        # 3 opportunities per position; contexts_by_gene counts
        # positions, channel counts opportunities. Under a germline
        # mask the total is the masked opportunity itself.
        total = (
            np.sum(type_opportunity.values)
            if type_opportunity is not None
            else 3 * np.sum(contexts_by_gene.values)
        )
        probs_g = channel.sum(axis=1) / total

        if tau_list is not None:
            out = {
                tau: probs_g.to_frame(0).dot(
                    mu_taus[tau].to_frame(0).T
                )
                for tau in tau_list
            }
        else:
            mu_tumor = mu_taus.sum(axis=1)
            out = probs_g.to_frame(0).dot(mu_tumor.to_frame(0).T)

    else:
        denominators = type_denominators(
            contexts_by_gene, type_opportunity
        )

        probs_g_tau = channel[canonical_types_order] / denominators

        if tau_list is not None:
            out = {
                tau: probs_g_tau[tau]
                .to_frame(0)
                .dot(mu_taus[tau].to_frame(0).T)
                for tau in tau_list
            }
        else:
            out = probs_g_tau.dot(mu_taus[canonical_types_order].T)

    if tau_list is not None:
        for tau_out in out.values():
            tau_out.index.name = "ensembl_gene_id"
    else:
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
    fallback_log_scale: float | pd.Series | None = None,
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
    fallback_log_scale : float, pd.Series or None, default None
        What a gene without a complete covariate row gets. ``None``
        drops it from the output (the historical behaviour, kept for
        callers that want the covariate-scaled genes only). A number,
        or a per-gene Series of log-scales, keeps it, at
        ``base_mus * exp(fallback_log_scale)``, so that
        every gene of `base_mus` (after `restrict_to_passenger`) has a
        rate. Applies to the single-model paths and to the combined
        multi-model path (genes no model covers); a
        ``separate_mus_per_model`` dict is per model by definition
        and ignores it.

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
                k for k in cov_effect if isinstance(k, tuple)
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

        ids_all = first_df.index.intersection(ids_pass)
        ids = ids_all.intersection(cov_matrix.index)
        if fallback_log_scale is not None:
            ids = ids.intersection(
                covariate_complete_genes(cov_matrix)
            )

        # Ensure all signatures have same genes
        for sigma, df in base_mus.items():
            ids = ids.intersection(df.index)
            ids_all = ids_all.intersection(df.index)

        if len(ids) == 0 and fallback_log_scale is None:
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

        if fallback_log_scale is not None:
            mus_full = _with_fallback(
                mus_full,
                sum(df.loc[ids_all] for df in base_mus.values()),
                ids_all,
                fallback_log_scale,
            )
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

    ids_all = ids
    ids = ids.intersection(cov_matrix.index)
    if len(ids) == 0 and fallback_log_scale is None:
        raise ValueError(
            "No overlapping genes between "
            "base_mus, cov_matrix, and passenger set."
        )

    def _finish(scaled):
        if fallback_log_scale is None:
            return scaled
        return _with_fallback(
            scaled, base_mus, ids_all, fallback_log_scale
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

        if fallback_log_scale is not None:
            # An incomplete row cannot be scaled (its eta would be
            # NaN); it falls back instead.
            complete = cov_df.notna().all(axis=1)
            mus, cov_df = mus.loc[complete], cov_df.loc[complete]

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
        return _finish(mus_full)

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
        # No usable models: return baseline mus unchanged -- or, when
        # a fallback is given, every gene takes it.
        if fallback_log_scale is not None:
            return _finish(mus.iloc[:0])
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

        # Unassigned genes keep baseline mus -- or, when a fallback is
        # given, take it like a gene absent from cov_matrix does.
        if fallback_log_scale is not None and (~assigned).any():
            mus_full = mus_full.loc[assigned]
        return _finish(mus_full)


def variant_site_denominators(
    contexts_by_gene: pd.DataFrame,
    opportunity_by_type: pd.DataFrame | None = None,
    prob_g_tau_tau_independent: bool = False,
) -> pd.DataFrame:
    """Per-(gene, type) site counts that turn a gene rate into a site rate.

    A variant's rate is its gene's type-tau rate spread evenly over
    the sites that rate was built from, so the divisor has to count
    exactly the opportunities in that rate's numerator:

    * **Merged rate** (``opportunity_by_type`` is None): every
      position with tau's source context offers one type-tau
      opportunity, so the divisor is ``n_{g,c(tau)}``.
    * **One consequence channel** (``opportunity_by_type`` is that
      channel's genes x 96 table, e.g. the non-synonymous one): only
      that channel's opportunities are in the numerator, so the
      divisor is ``n^{channel}_{g,tau}``. Dividing a channel rate by
      the full ``n_{g,c(tau)}`` would scale every site by the gene's
      channel fraction for tau -- a site's rate must not depend on how
      many *other* sites in its gene happen to be synonymous.

    Under ``prob_g_tau_tau_independent`` each gene's total is spread
    over types by the exome-wide context proportions, mirroring how
    the tau-independent gene rates are built: positions
    (``contexts_by_gene.sum(axis=1)``) for the merged rate, and
    channel opportunities / 3 for a channel (3 opportunities per
    position), which reduces to the merged form when the channel is
    everything.

    Parameters
    ----------
    contexts_by_gene : pandas.DataFrame
        Genes x 32 contexts (positions).
    opportunity_by_type : pandas.DataFrame or None
        Genes x 96 canonical types, one consequence channel's
        opportunities, or None for the merged rate.
    prob_g_tau_tau_independent : bool
        Must match the setting the gene rates were built with.

    Returns
    -------
    pandas.DataFrame
        Genes x 96 canonical types (``contexts_by_gene``'s genes).
    """
    from .constants import canonical_types_order, extract_context

    type_contexts = [
        extract_context(t) for t in canonical_types_order
    ]

    if prob_g_tau_tau_independent:
        context_share = (
            contexts_by_gene.sum(axis=0)
            / contexts_by_gene.values.sum()
        )
        if opportunity_by_type is None:
            per_gene = contexts_by_gene.sum(axis=1)
        else:
            per_gene = (
                opportunity_by_type.reindex(contexts_by_gene.index)
                .fillna(0)
                .sum(axis=1)
                / 3.0
            )
        out = pd.DataFrame(
            np.outer(
                per_gene.to_numpy(),
                context_share[type_contexts].to_numpy(),
            ),
            index=contexts_by_gene.index,
            columns=canonical_types_order,
        )
    elif opportunity_by_type is None:
        out = contexts_by_gene[type_contexts].copy()
        out.columns = canonical_types_order
    else:
        out = (
            opportunity_by_type.reindex(
                index=contexts_by_gene.index,
                columns=canonical_types_order,
            )
            .fillna(0)
            .astype(float)
        )
    return out


def route_weights_of(types, weights):
    """One site weight per route of a variant, aligned with its types.

    ``types`` is a variant's ``mut_types`` (one type or a list, one
    entry per route) and ``weights`` its ``route_weights`` entry
    (:func:`site_weights.variant_route_weights`). Anything that is not
    a list of the same length -- no site weights, a splice variant --
    gives weight 1 to every route. Every variant rate multiplies its
    route terms by these, so that the rate is the one its sites carry
    in the opportunity.
    """
    n = 1 if isinstance(types, str) else len(types)
    if (
        isinstance(weights, (list, tuple, np.ndarray))
        and len(weights) == n
    ):
        return [float(w) for w in weights]
    return [1.0] * n


def compute_mu_m_per_tumor(
    variants_df: pd.DataFrame,
    mu_g_tau_j: dict[str, pd.DataFrame],
    contexts_by_gene: pd.DataFrame,
    prob_g_tau_tau_independent=False,
    opportunity_by_type: pd.DataFrame | None = None,
) -> pd.DataFrame:
    r"""Compute per-variant expected mutation rate per tumor.

    Transform gene-level, per-type mutation rates into variant-level
    rates: for variant m of type τ in gene g,
    ``μ_{m}^{j} = μ_{g,tau}^{j} / n_{g,c(tau)}``, i.e. the type-τ
    share of gene g's rate, split evenly across the ``n_{g,c(tau)}``
    occurrences of τ's source context in gene g.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        Variant table with columns:
        - 'mut_types': mutation type(s) in COSMIC format (e.g.,
          'G[C>T]G' or list of types for multi-type variants)
        - 'ensembl_gene_id': Ensembl gene identifier
        - 'gene': gene symbol
        Index should be variant identifiers (e.g., 'KRAS p.G12D').

    mu_g_tau_j : dict[str, pandas.DataFrame]
        Mapping from mutation type τ (as in
        ``constants.canonical_types_order``) to a Genes × Tumors
        DataFrame of ``μ_{g,tau}^{j}`` --- the type-τ share of each
        gene's rate, *not* summed over τ. As returned by
        :func:`compute_mu_g_per_tumor` with ``separate_per_tau=True``
        (dict-of-signature input there must already be summed over
        signature per τ before being passed here).
        Index: Ensembl gene IDs (e.g., 'ENSG00000133703').
        Columns: tumor barcodes (e.g., 'TCGA-5M-A8F6-...').

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
        match the one used to compute ``mu_g_tau_j``, via
        :func:`compute_mu_g_per_tumor`.

    opportunity_by_type : pandas.DataFrame or None, default None
        Genes x 96 canonical types: the opportunities of the channel
        ``mu_g_tau_j`` was built from, e.g. the non-synonymous table
        when ``mu_g_tau_j`` comes from
        :func:`compute_mu_g_channel_per_tumor`. The divisor is then
        ``n^{channel}_{g,tau}`` instead of ``n_{g,c(tau)}``; see
        :func:`variant_site_denominators`. None (the default) keeps
        the merged-rate divisor.

    Returns
    -------
    pandas.DataFrame
        Variants × tumors matrix.
        Index: variant identifiers from ``variants_df.index``.
        Columns: tumor barcodes (same as the ``mu_g_tau_j`` values').
        Values: expected mutation rate for each variant in each tumor.

    Notes
    -----
    The calculation proceeds in three steps:

    1. Extract the trinucleotide context from each variant's
       mutation type(s): 'G[C>T]G' → 'GCG'.

    2. Look up ``n_{g,c(tau)}``, the number of occurrences of that
       context in the variant's gene (from ``contexts_by_gene``, or
       from a genome-wide-redistributed version of it when
       ``prob_g_tau_tau_independent=True``).

    3. Divide the τ-specific gene rate by that count:
       ``μ_{m}^{j} = μ_{g,tau}^{j} / n_{g,c(tau)}``.

    For a variant with several types, the per-type terms are summed:
    ``μ_m^j = Σ_tau μ_{g,tau}^{j} / n_{g,c(tau)}``. With the channel
    universe ``mut_types`` lists every single-nucleotide route to the
    protein change (repeats kept, one per site), so this is the rate
    of acquiring the change by any route -- common for synonymous
    changes and for about one missense variant in nine.

    See Also
    --------
    compute_mu_g_per_tumor : Per-gene, per-type expected rate per
        tumor.

    """
    logger.info("Computing per-variant mutation rates per tumor...")

    from .constants import extract_context

    variants = variants_df.copy()

    # One divisor per (gene, type), matching the opportunities in the
    # numerator's rate: n_{g,c(tau)} for a merged rate, the channel's
    # own n_{g,tau} for a consequence channel. Includes the
    # tau-independent genome-wide redistribution.
    denominators = variant_site_denominators(
        contexts_by_gene,
        opportunity_by_type=opportunity_by_type,
        prob_g_tau_tau_independent=prob_g_tau_tau_independent,
    )

    valid_genes = set(contexts_by_gene.index)
    valid_contexts = set(contexts_by_gene.columns)

    tumors = next(iter(mu_g_tau_j.values())).columns
    mu_m_j = np.zeros((len(variants), len(tumors)))
    # Track which variants got at least one valid (gene, tau) term,
    # so the rest can be filled with 0, matching the
    # zero-contexts/missing-gene fallback of the previous
    # implementation.
    any_term = np.zeros(len(variants), dtype=bool)

    # One row per (variant, route type). A list of types is summed
    # term by term, repeats included: with every route enumerated
    # (channel universe), two sites of one type are two routes, each
    # at the site rate. Vectorised per type, so a cohort where most
    # variants have several routes costs no more than one where
    # almost none do.
    # Site weights (site_weights.py): one weight per route, aligned
    # with mut_types; 1 where there are none.
    has_w = "route_weights" in variants.columns

    routes = pd.DataFrame(
        {
            "row": np.arange(len(variants)),
            "mut_types": variants["mut_types"].to_numpy(),
            "ensembl_gene_id": variants["ensembl_gene_id"].to_numpy(),
        }
    )
    routes = routes[
        routes["mut_types"].map(lambda x: isinstance(x, (str, list)))
    ]
    routes["mut_types"] = routes["mut_types"].map(
        lambda x: [x] if isinstance(x, str) else list(x)
    )
    wcol = (
        variants["route_weights"].to_numpy()[routes["row"].to_numpy()]
        if has_w
        else [None] * len(routes)
    )
    routes["route_weight"] = [
        route_weights_of(t, w)
        for t, w in zip(routes["mut_types"], wcol)
    ]
    routes = routes.explode(["mut_types", "route_weight"])
    routes["route_weight"] = routes["route_weight"].astype(float)

    for tau, group in routes.groupby("mut_types"):
        if tau not in mu_g_tau_j:
            continue
        context = extract_context(tau)
        if context not in valid_contexts:
            continue

        gene_ids = group["ensembl_gene_id"]
        known = gene_ids.isin(valid_genes).to_numpy()
        n_sites = np.full(len(group), np.nan)
        n_sites[known] = denominators.loc[
            gene_ids[known], tau
        ].to_numpy(dtype=float)
        valid = np.isfinite(n_sites) & (n_sites != 0)
        if not valid.any():
            continue

        rates = (
            mu_g_tau_j[tau]
            .reindex(index=gene_ids[valid], columns=tumors)
            .to_numpy(dtype=float)
        )
        term = rates / n_sites[valid][:, None]
        term = term * group["route_weight"].to_numpy()[valid][:, None]
        rows = group["row"].to_numpy()[valid]
        np.add.at(mu_m_j, rows, np.nan_to_num(term))
        any_term[rows] = True

    mu_m_j = pd.DataFrame(
        mu_m_j, index=variants.index, columns=tumors
    )

    # Fill variants with no valid (gene, tau) term with 0 (missing
    # genes or zero contexts), matching the previous implementation.
    mu_m_j.loc[~any_term] = 0.0
    mu_m_j = mu_m_j.fillna(0.0)

    logger.info("... done with per-variant mutation rates per tumor.")
    return mu_m_j
