"""Probabilities mutation is attributed to signature (alphas).

This module contains tools for estimating the probabilities per tumor
that a mutation is attributed to a signature (alphas).

The main function `estimate_alphas` uses observed signature
assignments and adjusts proportions for low-burden samples using a
convex combination with an average profile (`alpha_bar`) computed
from intermediate-burden samples.

Typical usage requires a mutation burden summary from
:func:`mutation_burden.count_mutation_burden` and a signature
assignment matrix from a signature decomposition step.

"""

from typing import Optional
import logging
import pandas as pd
import numpy as np

from .compute_mutation_burden import count_mutation_burden


logger = logging.getLogger(__name__)


def estimate_alphas(
        df: pd.DataFrame,
        assignments: pd.DataFrame,
        L_low: Optional[float] = None,
        L_high: Optional[float] = None
        ) -> pd.DataFrame:
    """Return per-sample signature probabilities (`alphas`).

    This computes the probability that a mutation in a sample belongs
    to each signature: raw per-signature counts divided by that
    sample's total mutations. Optionally, it corrects low-burden
    samples via interpolation of signature assignments with an average
    profile from intermediate-burden samples.

    Behavior of thresholds
    ----------------------
    - If `L_low` is None: no correction; return raw proportions.
    - If `L_high` is None: treated as +inf (i.e., all with burden
      ≥ `L_low` are eligible as "intermediate").

    Parameters
    ----------
    df : pd.DataFrame
        Mutation burden summary. If it lacks a 'total_mutations'
        column, `count_mutation_burden(df)` is called.
    assignments : pd.DataFrame
        Raw signature counts (rows: samples, cols: signatures).
        Must be index-aligned (same sample IDs) with `df` or a
        superset; the intersection is used.
    L_low : float or None, default None
        Lower threshold for defining "low-burden" samples.
    L_high : float or None, default None
        Upper threshold for selecting "intermediate-burden" samples
        used to compute the average profile. It has to be higher than
        L_low. No effect if L_low is None.

    Returns
    -------
    alphas : pd.DataFrame
        Per-sample signature proportions (rows: samples,
        cols: signatures). Index matches the intersection of
        `df` and `assignments` sample indices.

    Notes
    -----
    - Samples with zero total burden yield all-zeros raw proportions.
    - If no samples fall in the intermediate range, the correction
      is skipped and raw proportions are returned.

    """
    # Ensure we have totals; align to assignments by sample index
    if 'total_mutations' in df.columns:
        mb = df.copy()
    else:
        mb = count_mutation_burden(df)

    # Align on common samples only
    idx = assignments.index.intersection(mb.index)
    if len(idx) == 0:
        raise ValueError("No overlapping samples between df and "
                         "assignments.")
    mb = mb.loc[idx]
    assignments = assignments.loc[idx]

    totals = mb['total_mutations'].astype(float)

    # Raw proportions; avoid divide-by-zero by masking totals==0.
    denom = totals.replace(0.0, np.nan)
    pre_alphas = assignments.div(denom, axis=0).fillna(0.0)

    # If no correction requested, return early.
    if L_low is None:
        logger.warning("No low-burden correction requested.")
        return pre_alphas

    # Normalize threshold defaults.
    L_low = 0.0 if L_low is None else float(L_low)
    L_high = np.inf if L_high is None else float(L_high)

    if not (L_low < L_high or np.isinf(L_high)):
        raise ValueError("Require L_low < L_high (or L_high=inf).")

    # Define sets.
    samples_normal = totals.between(L_low, L_high, inclusive='both')
    samples_low = totals < L_low

    # If there are no low-burden samples, nothing to blend.
    if not samples_low.any():
        logger.warning("No samples below L_low; returning raw proportions.")
        return pre_alphas

    # Compute alpha_bar over intermediate-burden samples.
    if samples_normal.any():
        alpha_bar = pre_alphas.loc[samples_normal].mean(axis=0)
    else:
        logger.warning("No samples in [L_low, L_high]; skipping "
                       "correction and returning raw proportions.")
        return pre_alphas

    # Blend for low-burden samples:
    # weight = totals / L_low (clipped to [0,1]); avoid div by zero.
    if L_low <= 0.0:
        logger.warning("L_low==0 implies no meaningful blending; "
                       "returning raw proportions.")
        return pre_alphas

    w = (totals.loc[samples_low] / L_low).clip(0.0, 1.0)

    # Broadcast alpha_bar to low sample index.
    alpha_bars = pd.DataFrame(
        np.repeat(alpha_bar.values[None, :], w.shape[0], axis=0),
        index=w.index, columns=alpha_bar.index)

    alphas_low = (pre_alphas.loc[samples_low].mul(w, axis=0) +
                  alpha_bars.mul(1.0 - w, axis=0))

    # Merge with non-low samples.
    alphas = pd.concat([alphas_low, pre_alphas.loc[~samples_low]])
    alphas = alphas.loc[idx]  # preserve original order

    return alphas
