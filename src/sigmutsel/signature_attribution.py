"""Signature attribution and probability computations.

This module provides functions to attribute observed mutations to
specific mutational signatures by computing posterior probabilities
P(σ | τ, j) for each mutation, where σ is a signature, τ is a
mutation type, and j is a sample.

The main workflow involves:
1. Estimating signature weights (alphas) per sample using
   `compute_alphas.estimate_alphas`
2. Computing attribution probabilities using Bayes' rule:
   P(σ | τ, j) = P(τ | σ) × P(σ | j) / P(τ | j)
3. Aggregating attributions by gene to quantify signature
   contributions at the gene level

These attribution probabilities are useful for:
- Understanding which signatures contribute to mutations in specific
  genes
- Analyzing correlations between signature activity and genomic
  features
- Building signature-specific covariate models

Notes
-----
Memory efficiency is achieved through chunked processing when
handling large mutation databases with hundreds of thousands of
variants.

See Also
--------
compute_alphas : Estimate signature weights per sample
signature_decomposition : Decompose mutation counts into signatures
"""

from collections import namedtuple

import numpy as np
import pandas as pd

#: Everything :func:`_attribution_chunks` needs, resolved once.
_Prepared = namedtuple(
    "_Prepared",
    "alphas signatures sig_matrix sample_idx type_idx",
)


def _prepare_attribution(
    db,
    assignments,
    location_sig_matrix_norm,
    L_low,
    L_high,
    alphas=None,
):
    """Resolve alphas, the signature matrix and db's row indices."""
    from .compute_alphas import estimate_alphas

    if alphas is None:
        alphas = estimate_alphas(db, assignments, L_low, L_high)

    if isinstance(location_sig_matrix_norm, pd.DataFrame):
        sig_matrix = location_sig_matrix_norm
        if "MutationType" in sig_matrix.columns:
            sig_matrix = sig_matrix.set_index("MutationType")
    else:
        sig_matrix = pd.read_csv(
            location_sig_matrix_norm, sep="\t"
        ).set_index("MutationType")

    common_sigs = sig_matrix.columns.intersection(alphas.columns)
    sig_matrix = sig_matrix[common_sigs]
    alphas = alphas[common_sigs]

    sample_to_idx = {
        sample: i for i, sample in enumerate(alphas.index)
    }
    type_to_idx = {
        mut_type: i for i, mut_type in enumerate(sig_matrix.index)
    }

    return _Prepared(
        alphas=alphas,
        signatures=common_sigs,
        sig_matrix=sig_matrix.values,
        sample_idx=[
            sample_to_idx[s] for s in db["Tumor_Sample_Barcode"]
        ],
        type_idx=[type_to_idx[t] for t in db["type"]],
    )


def _attribution_chunks(prepared, n_mutations, chunk_size):
    """Yield (start, end, probs) over `n_mutations` in batches.

    ``probs`` is the Bayes-rule attribution
    ``P(sigma|tau,j) = alpha_sigma(j) s_sigma(tau) / sum_sigma' ...``
    for the chunk's mutations, shape ``(end - start, n_signatures)``.
    """
    alphas_arr = prepared.alphas.values  # (n_samples, n_sigs)

    for start_idx in range(0, n_mutations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_mutations)

        numerator = (
            prepared.sig_matrix[prepared.type_idx[start_idx:end_idx]]
            * alphas_arr[prepared.sample_idx[start_idx:end_idx]]
        )
        denominator = numerator.sum(axis=1, keepdims=True)
        # A sample with all-zero alphas (zero total burden) gives an
        # all-zero numerator/denominator here; treat as 0 probability
        # mass rather than propagating a 0/0 NaN.
        with np.errstate(invalid="ignore", divide="ignore"):
            yield start_idx, end_idx, np.where(
                denominator > 0, numerator / denominator, 0.0
            )


def assign_signatures_per_gene_id(
    db,
    assignments,
    location_sig_matrix_norm,
    L_low,
    L_high,
    chunk_size=50000,
):
    """Assign signature probabilities per gene ID.

    Memory-efficient chunked version that processes mutations
    in batches to avoid creating large intermediate arrays.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation database with columns 'Tumor_Sample_Barcode',
        'type', and 'ensembl_gene_id'.
    assignments : pd.DataFrame
        Signature assignments per sample, output from
        signature_decomposition.
    location_sig_matrix_norm : str or Path
        Path to normalized signature matrix file.
    L_low : float
        Lower bound for mutation burden filter.
    L_high : float
        Upper bound for mutation burden filter.
    chunk_size : int, optional
        Number of mutations to process at once. Default
        50000.

    Returns
    -------
    pd.DataFrame
        Signature probabilities summed per gene, indexed by
        ensembl_gene_id with signature columns.
    """
    from .compute_alphas import estimate_alphas

    alphas = estimate_alphas(db, assignments, L_low, L_high)

    sig_matrix = pd.read_csv(
        location_sig_matrix_norm, sep="\t"
    ).set_index("MutationType")

    # Filter sig_matrix to only include signatures in assignments
    # (in case assignments was filtered to exclude signatures)
    common_sigs = sig_matrix.columns.intersection(alphas.columns)
    sig_matrix = sig_matrix[common_sigs]
    alphas = alphas[common_sigs]

    alphas_arr = alphas.values  # (n_samples, n_sigs)
    sig_matrix_arr = sig_matrix.values  # (n_types, n_sigs)
    n_sigs = sig_matrix_arr.shape[1]

    # Create lookup dictionaries
    sample_to_idx = {
        sample: i for i, sample in enumerate(alphas.index)
    }
    type_to_idx = {
        mut_type: i for i, mut_type in enumerate(sig_matrix.index)
    }

    # Get gene IDs and create mapping
    gene_ids_sorted = sorted(db["ensembl_gene_id"].unique())
    gene_to_idx = {gene: i for i, gene in enumerate(gene_ids_sorted)}
    n_genes = len(gene_ids_sorted)

    # Initialize result array
    result = np.zeros((n_genes, n_sigs), dtype=np.float64)

    # Extract columns once to avoid repeated DataFrame access
    sample_col = db["Tumor_Sample_Barcode"].values
    type_col = db["type"].values
    gene_col = db["ensembl_gene_id"].values

    # Process in chunks to limit memory usage
    n_mutations = len(db)
    for start_idx in range(0, n_mutations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_mutations)

        # Get indices for this chunk
        sample_chunk = [
            sample_to_idx[s] for s in sample_col[start_idx:end_idx]
        ]
        type_chunk = [
            type_to_idx[t] for t in type_col[start_idx:end_idx]
        ]
        gene_chunk = [
            gene_to_idx[g] for g in gene_col[start_idx:end_idx]
        ]

        # Compute probabilities for this chunk
        # P(τ | σ) × P(σ | j)
        numerator = (
            sig_matrix_arr[type_chunk] * alphas_arr[sample_chunk]
        )
        # P(τ | j) = Σ_σ numerator
        denominator = numerator.sum(axis=1, keepdims=True)
        # P(σ | τ, j)
        probs_chunk = numerator / denominator

        # Accumulate into result
        np.add.at(result, gene_chunk, probs_chunk)

    # Return as DataFrame
    return pd.DataFrame(
        result, columns=sig_matrix.columns, index=gene_ids_sorted
    )


def compute_signature_probability_mass(
    db,
    assignments,
    location_sig_matrix_norm,
    target_signatures,
    L_low=None,
    L_high=None,
    chunk_size=50000,
):
    """Per-mutation probability mass on a set of signatures.

    Computes the same per-mutation P(σ|τ,j) Bayes-rule attribution as
    :func:`assign_signatures_per_gene_id`, but returns it per mutation
    (aligned with `db`'s row order) rather than aggregated to gene
    level, and only for the summed probability mass on
    `target_signatures` rather than the full per-signature breakdown --
    e.g. for identifying mutations dominantly attributed to a
    technical-artifact signature (see `sigmutsel.qc
    .flag_artifact_signature_mutations`), without materializing a full
    mutations × all-signatures probability matrix.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation database with columns 'Tumor_Sample_Barcode' and
        'type'.
    assignments : pd.DataFrame
        Signature assignments per sample, output from
        signature_decomposition.
    location_sig_matrix_norm : str, Path, or pd.DataFrame
        Path to normalized signature matrix file, or an already-loaded
        DataFrame (index or a "MutationType" column giving mutation
        types, columns giving signature names) -- accepting a
        DataFrame avoids a redundant re-read when the caller already
        has it loaded, e.g. `MutationDataset._signature_matrix` right
        after a fit.
    target_signatures : Iterable[str]
        Signature names whose probability mass to sum per mutation
        (e.g. `constants.ARTIFACT_SIGNATURES`).
    L_low, L_high : float, optional
        Forwarded to `compute_alphas.estimate_alphas`. Default None
        (no low-burden blending correction -- matches this project's
        current default, see `Model.compute_mu_taus`).
    chunk_size : int, optional
        Mutations processed per batch. Default 50000.

    Returns
    -------
    np.ndarray
        1-D array of length len(db), same row order as db: each
        mutation's summed probability mass on `target_signatures`.
        A mutation whose sample has zero total burden (all-zero
        alphas) gets 0.0, not NaN.
    """
    prepared = _prepare_attribution(
        db, assignments, location_sig_matrix_norm, L_low, L_high
    )

    target_signatures = set(target_signatures) & set(
        prepared.signatures
    )
    target_mask = np.array(
        [sig in target_signatures for sig in prepared.signatures]
    )

    result = np.zeros(len(db), dtype=np.float64)
    for start_idx, end_idx, probs_chunk in _attribution_chunks(
        prepared, len(db), chunk_size
    ):
        result[start_idx:end_idx] = probs_chunk[:, target_mask].sum(
            axis=1
        )

    return result


def compute_signature_probabilities(
    db,
    assignments,
    location_sig_matrix_norm,
    L_low=None,
    L_high=None,
    chunk_size=50000,
    alphas=None,
):
    """Per-mutation P(sigma|tau,j) for every signature.

    The full per-signature breakdown that
    :func:`compute_signature_probability_mass` sums over a chosen
    subset, and :func:`assign_signatures_per_gene_id` aggregates to
    genes: one row per mutation of `db`, in `db`'s row order, one
    column per signature.

    It therefore materializes an ``n_mutations x n_signatures``
    frame, which the other two exist to avoid. Pass a `db` already
    restricted to the mutations of interest (a cohort's full
    mutation database against ~30 signatures is tens of megabytes).

    Parameters
    ----------
    db : pd.DataFrame
        Mutation database with columns 'Tumor_Sample_Barcode' and
        'type'.
    assignments : pd.DataFrame
        Signature assignments per sample, output from
        signature_decomposition.
    location_sig_matrix_norm : str, Path, or pd.DataFrame
        Normalized signature matrix, as a path or already loaded.
    L_low, L_high : float, optional
        Forwarded to `compute_alphas.estimate_alphas`.
    chunk_size : int, optional
        Mutations processed per batch. Default 50000.
    alphas : pd.DataFrame, optional
        Per-sample signature proportions, if already computed.
        Worth passing when `db` is a *subset* of a cohort: the
        attribution itself is unchanged either way (a per-sample
        rescaling of alphas cancels in the Bayes ratio), but alphas
        fitted on the subset are proportions of the subset, so
        anything else that reads them -- a source-share comparison,
        say -- would be reading the wrong denominator.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(db), n_signatures)``, indexed like `db`. Rows
        for a sample with zero total burden are all zeros, not NaN.
    """
    prepared = _prepare_attribution(
        db,
        assignments,
        location_sig_matrix_norm,
        L_low,
        L_high,
        alphas=alphas,
    )

    result = np.zeros(
        (len(db), len(prepared.signatures)), dtype=np.float64
    )
    for start_idx, end_idx, probs_chunk in _attribution_chunks(
        prepared, len(db), chunk_size
    ):
        result[start_idx:end_idx] = probs_chunk

    return pd.DataFrame(
        result, index=db.index, columns=prepared.signatures
    )


def compute_signature_effect_shares(
    probabilities,
    samples,
    units,
    gammas,
    source_shares=None,
    gamma_draws=None,
):
    """Share of a cohort's selection attributable to each signature.

    Attribution (P(sigma|tau,j), from
    :func:`compute_signature_probabilities`) answers "which process
    made this mutation". Weighting each mutation by the selection
    intensity gamma of the gene or variant it hit, and normalizing
    *within a tumor*, answers a different question: which process
    made the mutations that **mattered**. A signature whose effect
    share exceeds its source share contributes to oncogenesis out of
    proportion to the mutations it causes (Cannataro et al. 2022,
    Molecular Biology and Evolution, doi:10.1093/molbev/msac084).

    Per-tumor normalization *before* averaging over tumors is the
    point, not an implementation detail: it weights every tumor
    equally, so a hypermutator with a hundred times the mutations
    does not set the cohort's answer by itself.

    Parameters
    ----------
    probabilities : pd.DataFrame
        Per-mutation attribution, ``(n_mutations, n_signatures)``,
        as returned by :func:`compute_signature_probabilities`.
    samples : array-like
        Tumor of each row of `probabilities`.
    units : array-like
        The gene or variant each row's mutation belongs to -- the
        key its selection intensity is stored under.
    gammas : dict
        Selection intensity per unit. Rows whose unit is absent are
        dropped: a mutation with no fitted gamma contributes no
        effect, which is not the same as contributing zero effect,
        so the count of what was kept is reported.
    source_shares : pd.DataFrame, optional
        Per-sample signature proportions (``alphas``), for the
        source-share comparison. Restricted to the tumors that carry
        effects, so both averages describe the same population --
        cancereffectsizeR instead averages source shares over *all*
        samples, which compares two different denominators.
    gamma_draws : dict, optional
        Posterior draws per unit, each an array of the same length
        `n_draws`, aligned across units (draw ``d`` of every unit
        comes from the same posterior sample). Produces a
        distribution of cohort effect shares -- the uncertainty
        cancereffectsizeR's point estimates cannot carry.

    Returns
    -------
    dict
        ``by_mutation`` (attribution plus sample/unit columns),
        ``average_by_unit`` (mean attribution per unit),
        ``by_sample`` (per-tumor normalized effect shares),
        ``average_effect_shares``, ``average_source_shares`` (None
        if `source_shares` was not given), ``effect_share_draws``
        (``(n_draws, n_signatures)``, None without `gamma_draws`),
        and the counts ``n_mutations``, ``n_units``, ``n_samples``.
    """
    samples = np.asarray(samples)
    units = np.asarray(units)
    if not (len(samples) == len(units) == len(probabilities)):
        raise ValueError(
            "probabilities, samples and units must describe the "
            f"same rows: got {len(probabilities)}, {len(samples)} "
            f"and {len(units)}."
        )

    keep = np.array([unit in gammas for unit in units])
    if not keep.any():
        raise ValueError(
            "None of the mutations' units have a selection "
            "intensity in gammas."
        )

    probabilities = probabilities.loc[keep]
    samples = samples[keep]
    units = units[keep]
    signatures = probabilities.columns

    by_mutation = probabilities.copy()
    by_mutation.insert(0, "unit", units)
    by_mutation.insert(0, "Tumor_Sample_Barcode", samples)

    average_by_unit = probabilities.groupby(units, sort=True).mean()
    average_by_unit.index.name = "unit"

    weights = np.array(
        [float(gammas[unit]) for unit in units], dtype=np.float64
    )
    by_sample = _normalized_shares_by_sample(
        probabilities.values, weights, samples, signatures
    )

    if source_shares is not None:
        source_shares = source_shares.loc[by_sample.index]
        totals = source_shares.sum(axis=1).replace(0.0, np.nan)
        average_source_shares = source_shares.div(
            totals, axis=0
        ).mean()
    else:
        average_source_shares = None

    effect_share_draws = None
    if gamma_draws is not None:
        draws = np.column_stack(
            [
                np.asarray(gamma_draws[unit], dtype=np.float64)
                for unit in units
            ]
        )  # (n_draws, n_kept)
        effect_share_draws = pd.DataFrame(
            [
                _normalized_shares_by_sample(
                    probabilities.values, draw, samples, signatures
                ).mean()
                for draw in draws
            ],
            columns=signatures,
        )

    return {
        "by_mutation": by_mutation,
        "average_by_unit": average_by_unit,
        "by_sample": by_sample,
        "average_effect_shares": by_sample.mean(),
        "average_source_shares": average_source_shares,
        "effect_share_draws": effect_share_draws,
        "n_mutations": int(keep.sum()),
        "n_units": int(average_by_unit.shape[0]),
        "n_samples": int(by_sample.shape[0]),
    }


def _normalized_shares_by_sample(
    probabilities, weights, samples, signatures
):
    """Sum gamma-weighted attribution per tumor, normalized to 1."""
    weighted = pd.DataFrame(
        probabilities * weights[:, None],
        columns=signatures,
    )
    by_sample = weighted.groupby(samples, sort=True).sum()
    by_sample.index.name = "Tumor_Sample_Barcode"

    totals = by_sample.sum(axis=1)
    # A tumor whose every fitted unit has gamma 0 (or whose sample
    # had zero burden, so no attribution) has no effect to share
    # out; it is dropped rather than turned into NaN rows that
    # would poison the cohort average.
    by_sample = by_sample.loc[totals > 0]
    return by_sample.div(by_sample.sum(axis=1), axis=0)
