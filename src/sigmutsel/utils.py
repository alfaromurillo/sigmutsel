"""Utility functions for sigmutsel.

This module contains general utility functions used across the
package, including PCA operations and data transformations.
"""

import pandas as pd
from sklearn.decomposition import PCA


def run_riemannian_stats_on_covariates(
    cov_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_components: int | None = None,
    *,
    standardize: bool = True,
    dropna: str = "any",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute Riemannian principal components over gene covariates.

    Implements the riemannian-stats algorithm (Rodriguez et al.):
    UMAP-geometry-aware PCA where the centering and standardization
    are done with respect to the Riemannian mean (the point with
    minimum total UMAP-weighted distance to all others) and
    pairwise differences are scaled by rho = 1 - UMAP_similarity.

    This implementation avoids the O(n²·p) memory bottleneck of the
    reference package by computing the Riemannian covariance without
    materialising the (n, n, p) difference tensor. Memory usage is
    O(n²) for the UMAP graph and O(n·p) for the data, which is
    feasible at genome scale.

    Parameters
    ----------
    cov_df : pandas.DataFrame
        Gene-indexed covariates (index = ensembl_gene_id).
    columns : list[str] | None
        Subset of columns to include. If None, use all numeric cols.
    n_components : int | None, default None
        Number of Riemannian components to return. If None, returns
        all p components.
    standardize : bool, default True
        If True, z-score features before decomposition.
    dropna : {'any','all','none'}, default 'any'
        How to handle NaNs across selected columns.
    n_neighbors : int, default 15
        Number of neighbors for the UMAP graph. 15 is UMAP's
        conventional recommendation for gene-scale data (the
        riemannian-stats package default of 3 suits iris-size data).
    min_dist : float, default 0.1
        UMAP minimum distance parameter.
    metric : str, default 'euclidean'
        Distance metric for UMAP.

    Returns
    -------
    scores : pandas.DataFrame
        Gene-indexed Riemannian component scores with columns
        'RC1', 'RC2', ...

        ``scores.attrs["explained_inertia"]`` : ndarray of shape (k,)
          Fraction of total Riemannian inertia per component.

    """
    import warnings
    import numpy as np

    if columns is None:
        cols = [
            c
            for c in cov_df.columns
            if pd.api.types.is_numeric_dtype(cov_df[c])
        ]
    else:
        cols = columns

    X = cov_df[cols].copy()

    if dropna == "any":
        X = X.dropna(how="any")
    elif dropna == "all":
        X = X.dropna(how="all")
    elif dropna == "none":
        X = X.fillna(X.mean())
    else:
        raise ValueError(
            f"dropna must be 'any', 'all', or 'none', got {dropna!r}"
        )

    if standardize:
        X = (X - X.mean()) / X.std()

    X_vals = X.values.astype(float)
    n, p = X_vals.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap as _umap

    # Build UMAP sparse graph → rho = 1 - similarity
    reducer = _umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    reducer.fit(X_vals)
    sim_sparse = reducer.graph_

    # Find Riemannian mean: argmin of sum_j rho[i,j]*||x_i - x_j||
    # = sum_j ||x_i-x_j|| - sum_{j in nbrs(i)} sim[i,j]*||x_i-x_j||
    # Compute sum_j ||x_i - x_j|| row by row to keep memory at O(n·p)
    row_dist_sums = np.zeros(n)
    for i in range(n):
        row_dist_sums[i] = np.linalg.norm(
            X_vals[i] - X_vals, axis=1
        ).sum()

    # Sparse correction for the neighbor-similarity term
    sim_coo = sim_sparse.tocoo()
    for i, j, s in zip(sim_coo.row, sim_coo.col, sim_coo.data):
        row_dist_sums[i] -= s * np.linalg.norm(X_vals[i] - X_vals[j])

    mean_idx = int(np.argmin(row_dist_sums))

    # rho column at mean_idx (sparse → dense; mostly 1s)
    rho_col = (
        1.0
        - np.array(sim_sparse.getcol(mean_idx).todense()).flatten()
    )

    # Riemannian-mean-centred data: shape (n, p)
    centered = rho_col[:, None] * (X_vals - X_vals[mean_idx])

    # Riemannian covariance and correlation (p × p)
    cov = (centered.T @ centered) / n
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    # Eigendecomposition (eigh exploits symmetry)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Riemannian standardise then project
    riemannian_std = np.sqrt((centered**2).sum(axis=0) / n)
    scores_all = (centered / riemannian_std) @ eigenvectors

    k = n_components if n_components is not None else p
    rc_names = [f"RC{i + 1}" for i in range(k)]
    result = pd.DataFrame(
        scores_all[:, :k].real, index=X.index, columns=rc_names
    )

    total = np.abs(eigenvalues).sum()
    result.attrs["explained_inertia"] = (
        np.abs(eigenvalues[:k]) / total
    )

    return result


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
