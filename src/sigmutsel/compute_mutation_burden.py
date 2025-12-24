"""Mutation burden.

Estimation of mutation burden, expected mutation burden and mutation
burden corrected for low counts (ell_hats) per tumor.

"""

import logging
import pandas as pd


logger = logging.getLogger(__name__)


def count_mutation_burden(db):
    """Count total and synonymous mutations per sample.

    This function groups a mutation DataFrame by tumor sample barcode
    and computes:

    - The total number of mutations per sample.
    - The number of synonymous ("Silent") mutations per sample.

    It returns a DataFrame with one row per sample and two columns:
    'total_mutations' and 'synonymous_mutations'.


    Parameters
    ----------
    db : pd.DataFrame
        A DataFrame containing mutation data with at least the columns
        'Tumor_Sample_Barcode' and 'Variant_Classification'.


    Returns
    -------
    mutation_counts : pd.DataFrame
        A DataFrame indexed by 'Tumor_Sample_Barcode' with two columns:
        - total_mutations : int
            Total number of mutations in each sample.
        - synonymous_mutations : int
            Number of synonymous (silent) mutations in each sample.

    """
    total_counts = db.groupby(
        'Tumor_Sample_Barcode').size().rename('total_mutations')

    silent_db = db[db['Variant_Classification'] == 'Silent']

    silent_counts = silent_db.groupby(
        'Tumor_Sample_Barcode').size().rename('synonymous_mutations')

    mutation_counts = pd.concat([total_counts,
                                 silent_counts], axis=1).fillna(0).astype(int)

    return mutation_counts


def estimate_ell_hats(df, L_low, L_high, *,
                      cut_at_L_low=False):
    """Estimate adjusted mutation burdens (ell_hat) per sample.

    This function uses the total or expected mutation burden per
    sample and applies a correction for low-burden samples based on a
    minimum threshold `L_low`.

    If `cut_at_L_low` is True, samples with fewer than `L_low`
    mutations are clipped to a fixed value. Otherwise, a smoothed
    adjustment is applied:
        ell_hat = ell + (1 - ell / L_low) * ell

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing mutation burden per sample, as returned
        by :func:`count_mutation_burden`. Must include columns:
        'total_mutations', and 'synonymous_mutations'. If those
        columns are not present it assumes that `df` is a mutation
        DataFrame and runs :func:`count_mutation_burden` first.

    L_low : float
        Threshold below which burdens are corrected.

    L_high : float
        Upper threshold for computing expected burden at `L_low`.

    cut_at_L_low : bool, default=True
        If True, low-burden samples are assigned a fixed value.
        If False, use a smoothed burden adjustment.

    Returns
    -------
    ell_hats : pd.Series
        Estimated burdens, indexed by Tumor_Sample_Barcode.

    """
    if all(col in df.columns for col in ['total_mutations',
                                         'synonymous_mutations']):
        mb = df.copy()

    else:
        mb = count_mutation_burden(df)

    ells = mb['total_mutations']
    L_low_star = L_low

    samples_low = mb['total_mutations'] < L_low
    samples_not_low = ~samples_low

    if cut_at_L_low:
        ell_hats = pd.Series(L_low_star,
                             index=mb.index[samples_low])
    else:
        ells_low = mb['total_mutations'][samples_low]
        ell_hats = ((ells_low / L_low) * L_low_star
                    + (1 - ells_low / L_low) * ells[samples_low])

    ell_hats = pd.concat([ell_hats, ells[samples_not_low]])
    ell_hats = ell_hats.reindex(mb.index)
    ell_hats = ell_hats.rename('ell_hats')

    return ell_hats
