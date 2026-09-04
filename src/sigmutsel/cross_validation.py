"""Gene-level cross-validation for passenger-gene R² metrics.

Both :meth:`models.Model.estimate_passenger_genes_r2` targets are
in-sample metrics: they fit covariate effects (or ``r_g``/``theta``)
on every passenger gene and score against that same set, so the R²
reported is mathematically guaranteed non-decreasing as the covariate
matrix grows richer (e.g. as PCA ``n_components`` increases) and
cannot by itself detect overfitting. The covariates here are
gene-level, not tumor-level, so the natural cross-validation unit is
a gene-level train/test split: fit on a subset of passenger genes,
then score on the held-out remainder.

This module adds no new fitting logic of its own -- it only
orchestrates parameters that already exist on :class:`models.Model`
for exactly this purpose:

- :func:`gene_cv_passenger_r2` for the single-channel presence/absence
  model (:meth:`models.Model.estimate_cov_effects`'s ``train_genes``,
  :meth:`models.Model.estimate_passenger_genes_r2`'s ``genes``).
- :func:`channel_gene_cv_passenger_r2` for the consequence-split,
  shared-``r_g`` model
  (:meth:`models.Model.estimate_channel_rg_cov_effects`'s
  ``train_genes``, the same ``estimate_passenger_genes_r2`` ``genes``
  on the ``"non_silent_counts"`` target).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .estimate_presence import filter_passenger_genes_ensembl

logger = logging.getLogger(__name__)


def gene_cv_passenger_r2(
    model,
    n_splits=5,
    *,
    random_state=None,
    excluded_samples=None,
    cov_effects_kwargs=None,
):
    """Gene-level k-fold cross-validated passenger-gene R² (single channel).

    For each fold, covariate effects are fit
    (:meth:`models.Model.estimate_cov_effects`) on the training
    genes only, then scored
    (:meth:`models.Model.estimate_passenger_genes_r2`) on the
    held-out test genes. ``model.cov_matrix`` (and any PCA/DR
    transform already applied to it via :meth:`models.Model.
    assign_cov_matrix`) is reused unchanged across folds -- PCA is
    unsupervised and never sees the presence/absence outcome being
    predicted, so refitting it per fold would add cost without
    changing what's being validated. Only the supervised
    covariate-effects fit is cross-validated.

    Mutates ``model`` in place (``model.cov_effects``,
    ``model.passenger_genes_r2``, etc. reflect whichever fold ran
    last) -- call this on a model you don't otherwise need the
    in-sample fit/score state of afterward, or re-fit on the full
    gene set again if you do.

    Parameters
    ----------
    model : models.Model
        A model with ``cov_matrix`` already assigned (and
        ``compute_mu_taus``/``compute_base_mus`` already run, as
        required by ``estimate_cov_effects``).
    n_splits : int, default 5
        Number of gene-level folds.
    random_state : int or None, default None
        Passed to :class:`sklearn.model_selection.KFold` for
        reproducible fold assignment.
    excluded_samples : collection of str or None, default None
        Forwarded unchanged to both ``estimate_cov_effects`` and
        ``estimate_passenger_genes_r2`` in every fold.
    cov_effects_kwargs : dict or None, default None
        Extra keyword arguments forwarded to ``estimate_cov_effects``
        in every fold (e.g. ``sample``, ``tol``). ``train_genes`` is
        always set by this function and cannot be overridden this
        way.

    Returns
    -------
    dict
        - ``fold_r2`` : list[float], one R² per fold
        - ``mean`` : float, mean of ``fold_r2``
        - ``std`` : float, population std of ``fold_r2``
        - ``n_genes`` : int, size of the passenger-gene universe
          the folds were drawn from
    """
    if model.cov_matrix is None:
        raise ValueError(
            "model.cov_matrix is None. Call assign_cov_matrix() "
            "before cross-validating."
        )

    cov_effects_kwargs = dict(cov_effects_kwargs or {})

    passenger_genes = pd.Index(
        filter_passenger_genes_ensembl(model.cov_matrix.index)
    )

    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_r2 = []
    for fold, (train_idx, test_idx) in enumerate(
        kf.split(passenger_genes), start=1
    ):
        train_genes = passenger_genes[train_idx]
        test_genes = passenger_genes[test_idx]
        logger.info(
            f"Gene CV fold {fold}/{n_splits}: "
            f"{len(train_genes)} train genes, {len(test_genes)} test genes"
        )
        model.estimate_cov_effects(
            train_genes=train_genes,
            excluded_samples=excluded_samples,
            **cov_effects_kwargs,
        )
        r2 = model.estimate_passenger_genes_r2(
            genes=test_genes, excluded_samples=excluded_samples
        )
        fold_r2.append(r2)

    fold_r2 = np.array(fold_r2)
    return {
        "fold_r2": fold_r2.tolist(),
        "mean": float(fold_r2.mean()),
        "std": float(fold_r2.std()),
        "n_genes": len(passenger_genes),
    }


def channel_gene_cv_passenger_r2(
    model,
    n_splits=5,
    *,
    random_state=None,
    excluded_samples=None,
    channel_rg_kwargs=None,
):
    """Gene-level k-fold CV for the consequence-split ``r_g`` model.

    Same idea as :func:`gene_cv_passenger_r2`, but for the
    tau-dependent, two-channel Poisson, shared-``r_g`` model
    (:meth:`models.Model.estimate_channel_rg_cov_effects`): each fold
    fits ``c``/``theta`` with the non-silent channel restricted to the
    training passenger genes (``train_genes``), then scores on the
    non-silent COUNT target for the held-out test genes only, using
    the fold's own silent-only evaluation ``r_g``
    (:meth:`models.Model.compute_r_g_for_evaluation`) -- the same
    production recipe as
    ``estimate_passenger_genes_r2(target="non_silent_counts",
    gene_scaling=model.compute_r_g_for_evaluation())``, just gene-CV'd.

    Held-out genes still contribute their silent-channel counts to
    the fit (as do driver genes, via ``include_drivers``) -- only
    their non-silent signal, the one being scored, is withheld; see
    :meth:`models.Model._channel_gene_statistics`'s ``train_genes``
    docstring for why that is the correct restriction here rather
    than dropping held-out genes from the fit entirely.

    Mutates ``model`` in place, same caveat as
    :func:`gene_cv_passenger_r2`.

    Parameters
    ----------
    model : models.Model
        A model with ``cov_matrix`` assigned and
        ``compute_channel_base_mus()`` already run (as required by
        ``estimate_channel_rg_cov_effects``).
    n_splits : int, default 5
        Number of gene-level folds.
    random_state : int or None, default None
        Passed to :class:`sklearn.model_selection.KFold`.
    excluded_samples : collection of str or None, default None
        Forwarded unchanged to ``estimate_channel_rg_cov_effects`` and
        ``estimate_passenger_genes_r2`` in every fold.
    channel_rg_kwargs : dict or None, default None
        Extra keyword arguments forwarded to
        ``estimate_channel_rg_cov_effects`` in every fold (e.g.
        ``sample``, ``include_drivers``, ``separate_c``).
        ``train_genes`` is always set by this function and cannot be
        overridden this way.

    Returns
    -------
    dict
        Same shape as :func:`gene_cv_passenger_r2`'s return value:
        ``fold_r2``, ``mean``, ``std``, ``n_genes``.
    """
    if model.cov_matrix is None:
        raise ValueError(
            "model.cov_matrix is None. Call assign_cov_matrix() "
            "before cross-validating."
        )

    channel_rg_kwargs = dict(channel_rg_kwargs or {})

    complete_genes = model.cov_matrix.index[
        ~model.cov_matrix.isna().any(axis=1)
    ]
    passenger_genes = pd.Index(
        filter_passenger_genes_ensembl(complete_genes)
    )

    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_r2 = []
    for fold, (train_idx, test_idx) in enumerate(
        kf.split(passenger_genes), start=1
    ):
        train_genes = passenger_genes[train_idx]
        test_genes = passenger_genes[test_idx]
        logger.info(
            f"Channel gene CV fold {fold}/{n_splits}: "
            f"{len(train_genes)} train genes, {len(test_genes)} test genes"
        )
        model.estimate_channel_rg_cov_effects(
            train_genes=train_genes,
            excluded_samples=excluded_samples,
            **channel_rg_kwargs,
        )
        r2 = model.estimate_passenger_genes_r2(
            excluded_samples=excluded_samples,
            target="non_silent_counts",
            gene_scaling=model.compute_r_g_for_evaluation(),
            genes=test_genes,
        )
        fold_r2.append(r2)

    fold_r2 = np.array(fold_r2)
    return {
        "fold_r2": fold_r2.tolist(),
        "mean": float(fold_r2.mean()),
        "std": float(fold_r2.std()),
        "n_genes": len(passenger_genes),
    }
