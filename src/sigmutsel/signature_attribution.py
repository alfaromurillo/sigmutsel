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

import numpy as np
import pandas as pd


def assign_signatures_per_gene_id(
        db,
        assignments,
        location_sig_matrix_norm,
        L_low,
        L_high,
        chunk_size=50000):
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
    from compute_alphas import estimate_alphas
    alphas = estimate_alphas(db, assignments, L_low, L_high)

    sig_matrix = pd.read_csv(
        location_sig_matrix_norm,
        sep="\t").set_index('MutationType')

    # Filter sig_matrix to only include signatures in assignments
    # (in case assignments was filtered to exclude signatures)
    common_sigs = sig_matrix.columns.intersection(
        alphas.columns)
    sig_matrix = sig_matrix[common_sigs]
    alphas = alphas[common_sigs]

    alphas_arr = alphas.values  # (n_samples, n_sigs)
    sig_matrix_arr = sig_matrix.values  # (n_types, n_sigs)
    n_sigs = sig_matrix_arr.shape[1]

    # Create lookup dictionaries
    sample_to_idx = {
        sample: i for i, sample in enumerate(alphas.index)}
    type_to_idx = {
        mut_type: i
        for i, mut_type in enumerate(sig_matrix.index)}

    # Get gene IDs and create mapping
    gene_ids_sorted = sorted(db['ensembl_gene_id'].unique())
    gene_to_idx = {
        gene: i for i, gene in enumerate(gene_ids_sorted)}
    n_genes = len(gene_ids_sorted)

    # Initialize result array
    result = np.zeros((n_genes, n_sigs), dtype=np.float64)

    # Extract columns once to avoid repeated DataFrame access
    sample_col = db['Tumor_Sample_Barcode'].values
    type_col = db['type'].values
    gene_col = db['ensembl_gene_id'].values

    # Process in chunks to limit memory usage
    n_mutations = len(db)
    for start_idx in range(0, n_mutations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_mutations)

        # Get indices for this chunk
        sample_chunk = [
            sample_to_idx[s]
            for s in sample_col[start_idx:end_idx]]
        type_chunk = [
            type_to_idx[t] for t in type_col[start_idx:end_idx]]
        gene_chunk = [
            gene_to_idx[g] for g in gene_col[start_idx:end_idx]]

        # Compute probabilities for this chunk
        # P(τ | σ) × P(σ | j)
        numerator = (sig_matrix_arr[type_chunk] *
                     alphas_arr[sample_chunk])
        # P(τ | j) = Σ_σ numerator
        denominator = numerator.sum(axis=1, keepdims=True)
        # P(σ | τ, j)
        probs_chunk = numerator / denominator

        # Accumulate into result
        np.add.at(result, gene_chunk, probs_chunk)

    # Return as DataFrame
    return pd.DataFrame(
        result,
        columns=sig_matrix.columns,
        index=gene_ids_sorted)
