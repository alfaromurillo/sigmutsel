"""Utility functions for sigmutsel.

This module contains general utility functions used across the
package, including PCA operations and data transformations.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def run_pca_on_covariates(
    cov_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_components: int | None = None,
    *,
    standardize: bool = True,
    dropna: str = "any",
    **pca_kwargs,
) -> pd.DataFrame:
    """Compute PCA over gene-level covariates and return scores.

    Parameters
    ----------
    cov_df : pandas.DataFrame
        Gene-indexed covariates (index = ensembl_gene_id).
    columns : list[str] | None
        Subset of columns to include. If None, use all numeric cols.
    standardize : bool, default True
        If True, z-score features before PCA.
    dropna : {'any','all','none'}, default 'any'
        How to handle NaNs across selected columns:
        - 'any': drop rows with any NaN
        - 'all': drop rows with all NaN
        - 'none': fill remaining NaNs with column means
    n_components : int | None, default None
        Number of principal components. If None, PCA decides based on
        provided parameters (e.g., if n_samples is larger than the
        number of covariates, then the number of covariates).
    **pca_kwargs
        Extra keyword arguments forwarded to
        sklearn.decomposition.PCA (e.g., whiten=True,
        svd_solver='full').

    Returns
    -------
    scores : pandas.DataFrame
        Gene-indexed PC scores with columns 'PC1', 'PC2', ...

        The returned DataFrame contains PCA metadata in
        ``scores.attrs``:
        - ``explained_variance_ratio`` : ndarray of shape (k,)
          Fraction of total variance explained by each principal
          component.
        - ``components`` : ndarray of shape (k, n_features)
          Principal axes (loadings) in feature space.

    Examples
    --------
    >>> import pandas as pd
    >>> cov_df = pd.DataFrame({
    ...     'gene_expr': [1, 2, 3, 4, 5],
    ...     'chromatin': [5, 4, 3, 2, 1]
    ... }, index=['ENSG001', 'ENSG002', 'ENSG003', 'ENSG004',
    ...            'ENSG005'])
    >>> pca_scores = run_pca_on_covariates(cov_df, n_components=2)
    >>> pca_scores.shape
    (5, 2)
    >>> pca_scores.columns.tolist()
    ['PC1', 'PC2']

    """
    if columns is None:
        # keep only numeric columns
        cols = [
            c
            for c in cov_df.columns
            if pd.api.types.is_numeric_dtype(cov_df[c])
        ]
    else:
        cols = columns

    X = cov_df[cols].copy()

    # handle missing data
    if dropna == "any":
        X = X.dropna(how="any")
    elif dropna == "all":
        X = X.dropna(how="all")
    elif dropna == "none":
        # fill remaining NaNs with column means
        X = X.fillna(X.mean())
    else:
        raise ValueError(
            f"dropna must be 'any', 'all', or 'none', got {dropna!r}"
        )

    # standardize features if requested
    if standardize:
        X = (X - X.mean()) / X.std()

    # run PCA
    pca = PCA(n_components=n_components, **pca_kwargs)
    scores = pca.fit_transform(X)

    # build output DataFrame
    k = scores.shape[1]
    pc_names = [f"PC{i+1}" for i in range(k)]
    result = pd.DataFrame(scores, index=X.index, columns=pc_names)

    # attach metadata
    result.attrs["explained_variance_ratio"] = (
        pca.explained_variance_ratio_
    )
    result.attrs["components"] = pca.components_

    return result
