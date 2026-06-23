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

    Uses the riemannian-stats package (UMAP-geometry-aware PCA
    analog) to reduce dimensionality of gene-level covariates.

    Parameters
    ----------
    cov_df : pandas.DataFrame
        Gene-indexed covariates (index = ensembl_gene_id).
    columns : list[str] | None
        Subset of columns to include. If None, use all numeric cols.
    n_components : int | None, default None
        Number of Riemannian components to return. If None, returns
        all components (equal to the number of input features).
    standardize : bool, default True
        If True, z-score features before decomposition.
    dropna : {'any','all','none'}, default 'any'
        How to handle NaNs across selected columns.
    n_neighbors : int, default 15
        Number of neighbors for the UMAP graph. Larger values
        capture more global structure. Default 15 is UMAP's
        conventional recommendation for gene-scale data (the
        package default of 3 is designed for small datasets).
    min_dist : float, default 0.1
        UMAP minimum distance parameter.
    metric : str, default 'euclidean'
        Distance metric for UMAP.

    Returns
    -------
    scores : pandas.DataFrame
        Gene-indexed Riemannian component scores with columns
        'RC1', 'RC2', ...

        The returned DataFrame contains metadata in ``scores.attrs``:
        - ``explained_inertia_pc1_pc2`` : float
          Fraction of inertia explained by the first two components.

    """
    import warnings

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

    # riemannian_analysis requires a DataFrame (not bare numpy)
    X_df = pd.DataFrame(X.values, columns=cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from riemannian_stats import riemannian_analysis, utilities

    analysis = riemannian_analysis(
        X_df,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    corr = analysis.riemannian_correlation_matrix()
    all_comps = (
        analysis.riemannian_components_from_data_and_correlation(corr)
    )

    k = (
        n_components
        if n_components is not None
        else all_comps.shape[1]
    )
    scores = all_comps[:, :k]

    rc_names = [f"RC{i+1}" for i in range(k)]
    result = pd.DataFrame(scores, index=X.index, columns=rc_names)

    result.attrs["explained_inertia_pc1_pc2"] = (
        utilities.pca_inertia_by_components(corr, 0, 1)
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
