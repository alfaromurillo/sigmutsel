"""Compute variant and gene presence matrices across tumors.

This module provides functions to build binary presence matrices
indicating which variants or genes are mutated in each tumor sample.
It also includes utilities to filter passenger genes based on the
Cancer Gene Census.
"""

import re

from collections.abc import Sequence

import pandas as pd
import numpy as np

from .locations import location_cancer_gene_census

import logging


logger = logging.getLogger(__name__)


def compute_variants_present(
    db: pd.DataFrame, variants_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a 0/1 matrix of variant present per tumor.

    Parameters
    ----------
    db : pandas.DataFrame
        MAF-like table with at least the columns
        ``'Tumor_Sample_Barcode'`` and ``'variant'``.
    variants_df : pandas.DataFrame
        DataFrame whose index are the variant names you want in the
        result (e.g., the same index used for `compute_mu_m_per_tumor`).

    Returns
    -------
    pandas.DataFrame
        Variants × tumors matrix (uint8). Rows follow
        ``variants_df.index``; columns are tumor barcodes sorted
        alphabetically. Entry is 1 if the tumor carries the variant in
        ``db``, else 0.
    """
    logger.info(
        "Producing presence matrix for all tumors per variant..."
    )
    # drop rows without a variant label
    db_ = db.dropna(subset=["variant", "Tumor_Sample_Barcode"])

    # counts per (variant, tumor) -> convert to 0/1
    present = pd.crosstab(db_["variant"], db_["Tumor_Sample_Barcode"])
    present = (present > 0).astype("uint8")

    # enforce ordering: rows = variants_df.index, cols = sorted tumor list
    tumors_sorted = sorted(present.columns)  # alphabetical
    present = present.reindex(
        index=variants_df.index, columns=tumors_sorted, fill_value=0
    ).astype("uint8")

    logger.info("... done.")
    print("")
    return present


def compute_genes_present(db, scope=None):
    """Build a 0/1 matrix of gene presence per tumor.

    Parameters
    ----------
    db : pandas.DataFrame
        MAF-like table with at least 'ensembl_gene_id',
        'Tumor_Sample_Barcode', and 'Variant_Classification' columns.
    scope : {None, 'all', 'silent', 'non-silent'}, optional
        Filter variants by classification before computing presence:
        - None or 'all' (default): include all variants
        - 'silent': only Silent variants
        - 'non-silent': only non-Silent variants

    Returns
    -------
    pandas.DataFrame
        Genes × tumors matrix (int). Rows are Ensembl gene IDs;
        columns are tumor barcodes. Entry is 1 if the tumor has ≥1
        variant in that gene (filtered by scope), else 0.
    """
    logger.info(
        "Producing presence matrix for all tumors per gene..."
    )
    if scope == "silent" or "non-silent":
        logger.info(f"Restricting to {scope} variants")

    # Filter by variant classification scope
    if scope == "silent":
        db_filtered = db[db["Variant_Classification"] == "Silent"]
    elif scope == "non-silent":
        db_filtered = db[db["Variant_Classification"] != "Silent"]
    else:  # None or 'all'
        db_filtered = db

    present = pd.crosstab(
        db_filtered["ensembl_gene_id"],
        db_filtered["Tumor_Sample_Barcode"],
    )

    present = (present > 0).astype(int)

    logger.info("... done.")
    print("")
    return present


def filter_silent_variants(
    variants_df: pd.DataFrame, db: pd.DataFrame
) -> pd.Index:
    """Return variants (as an Index) that are annotated as 'Silent' in `db`.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        DataFrame indexed by variant name (e.g., "TP53 p.R175H").
    db : pandas.DataFrame
        MAF-like table containing columns 'Variant_Classification' and
        'variant'.

    Returns
    -------
    pandas.Index
        Variants present in `variants_df.index` whose classification in `db`
        is 'Silent', ordered as in `variants_df.index` for fast .loc[].

    """
    # All variants labeled 'Silent' in the MAF-like table
    silent_in_db = pd.Index(
        db.loc[db["Variant_Classification"].eq("Silent"), "variant"]
        .dropna()
        .unique()
    )

    # Keep only those present in variants_df, preserving its order
    return variants_df.index[variants_df.index.isin(silent_in_db)]


def filter_passenger_genes(db):
    """Return a list of passenger genes present in the mutation data.

    Passenger genes are defined as those not in the cancer gene census
    and with at least one variant annotated in the 'variant' column.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation data with 'gene' and 'variant' columns.

    Returns
    -------
    np.ndarray
        Array of passenger gene names.

    """
    census = pd.read_csv(location_cancer_gene_census, sep="\t")

    cancer_genes = census["Gene Symbol"].unique()

    passenger_genes = db[
        ~db["gene"].isin(cancer_genes) & db["variant"].notna()
    ]["gene"].unique()
    return passenger_genes


def filter_passenger_genes_ensembl(
    db: pd.DataFrame | pd.Series | pd.Index | Sequence[str],
    *,
    strip_version: bool = True,
) -> np.ndarray:
    """Return Ensembl IDs of passenger genes in `db`.

    Works in two modes:

    1) DataFrame mode
       - `db` is a DataFrame with columns 'ensembl_gene_id' and
         'variant'.
       - Returns Ensembl IDs for genes that are **not** in the Cancer
         Gene Census (CGC) (what we consider passenger genes) **and**
         have at least one non-null entry in 'variant'.

    2) Collection mode:
       - `db` is a sequence of Ensembl IDs (Index, list, Series, etc.).
       - Returns the subset of those IDs in `db` that are **not** in
         the CGC, that is those that we consider passenger. The
         'variant' criterion is not applied in this mode.

    Matching is done on Ensembl gene IDs (ENSG...). If *strip_version*
    is True, version suffixes ('.v') are removed on both sides before
    comparison.

    Parameters
    ----------
    db : pandas.DataFrame or sequence of str
        Either a mutation table with 'ensembl_gene_id' and 'variant'
        columns (DataFrame mode), or a 1-D collection of Ensembl IDs
        (collection mode).
    strip_version : bool, default True
        If True, drop the final '.number' version from Ensembl IDs in
        both the CGC set and `db` before computing membership.

    Returns
    -------
    numpy.ndarray
        Ensembl IDs **as they appear in the input** that are classified
        as passengers. In DataFrame mode, this means not in CGC and
        with ≥1 variant. In collection mode, this means not in CGC.

    Notes
    -----
    - Expects a module-level variable `location_cancer_gene_census`
      that points to the CGC TSV file.
    - Tries an explicit Ensembl column in the CGC (any column whose
      name contains 'ensembl', case-insensitive). If none exists, it
      extracts IDs from 'Synonyms' via the pattern
      ``ENSG\\d+(?:\\.\\d+)?``.

    """
    # ---- Load CGC ---------------------------------------------------
    census = pd.read_csv(
        location_cancer_gene_census, sep="\t", dtype=str
    )

    ens_candidates = [
        c
        for c in census.columns
        if re.search(r"ensembl", c, flags=re.I)
    ]
    if ens_candidates:
        ens_col = ens_candidates[0]
        cgc_ids = census[ens_col].dropna().astype(str)
    else:
        if "Synonyms" not in census.columns:
            raise KeyError(
                "CGC lacks an Ensembl column and 'Synonyms'."
            )
        cgc_ids = (
            census["Synonyms"]
            .dropna()
            .astype(str)
            .str.findall(r"ENSG\d+(?:\.\d+)?")
            .explode()
            .dropna()
        )

    if strip_version:
        cgc_ids = cgc_ids.str.replace(r"\.\d+$", "", regex=True)
    cgc_set = set(cgc_ids.tolist())

    # ---- Collection mode --------------------------------------------
    if not isinstance(db, pd.DataFrame):
        ids = pd.Index(db).astype(str)
        ids_norm = (
            ids.str.replace(r"\.\d+$", "", regex=True)
            if strip_version
            else ids
        )
        mask = ~ids_norm.isin(cgc_set)
        return ids[mask].unique()

    # ---- DataFrame mode ---------------------------------------------
    if "ensembl_gene_id" not in db.columns:
        raise KeyError(
            "db must contain 'ensembl_gene_id' (DataFrame)."
        )
    if "variant" not in db.columns:
        raise KeyError("db must contain 'variant' (DataFrame).")

    db_ids = db["ensembl_gene_id"].astype(str)
    db_ids_norm = (
        db_ids.str.replace(r"\.\d+$", "", regex=True)
        if strip_version
        else db_ids
    )

    mask = (~db_ids_norm.isin(cgc_set)) & (db["variant"].notna())
    return db.loc[mask, "ensembl_gene_id"].unique()
