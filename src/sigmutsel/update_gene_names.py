"""Update gene names to HGNC latest names."""


__author__ = "Jorge Alfaro-Murillo"

import logging
import pandas as pd

from pathlib import Path


logger = logging.getLogger(__name__)


def update_genes_with_gene_set(
    location_gene_set: str | Path,
    df: pd.DataFrame) -> pd.DataFrame:
    """Update gene symbols and Ensembl IDs to HGNC-approved values.

    A **MAF-like** DataFrame is required (must contain a ``'gene'``
    column).  Each symbol is replaced by its current HGNC‐approved
    symbol whenever it matches an entry in ``alias_symbol`` or
    ``prev_symbol``.  The corresponding stable
    ``ensembl_gene_id`` column is filled or updated at the same time.
    Every successful change is logged via ``logger.info``; the total
    number of unmapped symbols is logged with ``logger.warning``.

    Parameters
    ----------
    location_gene_set : str | Path
        Path to *hgnc_complete_set.txt* (tab-separated file).

    df : pandas.DataFrame
        Input table.  Must contain at least a ``'gene'`` column and may
        also include ``'variant'`` and ``'ensembl_gene_id'``.

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with harmonised ``'gene'``,
        ``'ensembl_gene_id'`` (and ``'variant'`` if present).
    """
    ...
    logger.info("Updating gene symbols and Ensembl IDs...")

    location_gene_set = Path(location_gene_set)

    # ── load HGNC table ──────────────────────────────────────────────
    keep_cols = ["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id"]
    hgnc = pd.read_csv(location_gene_set, sep="\t", usecols=keep_cols)

    approved: set[str] = set(hgnc["symbol"])
    symbol2ensg: dict[str, str] = (hgnc.set_index("symbol")["ensembl_gene_id"]
                                   .to_dict())

    # alias → approved symbol
    alias2sym: dict[str, str] = {}
    for _, row in hgnc.iterrows():
        for field in ("alias_symbol", "prev_symbol"):
            if pd.notna(row[field]):
                for alias in row[field].split("|"):
                    alias2sym[alias] = row["symbol"]

    def to_symbol(name: str) -> str:
        """Return approved symbol if known; otherwise original."""
        if name in approved:
            return name
        return alias2sym.get(name, name)

    new_df = df.copy()

    # --- update gene symbols ------------------------------------
    old_to_new: dict[str, str] = {}
    for g in new_df["gene"].unique():
        new = to_symbol(g)
        if new != g:
            logger.info(f"Updating 'gene' {g} to {new}")
            old_to_new[g] = new
    if old_to_new:
        new_df["gene"] = new_df["gene"].map(lambda g: old_to_new.get(g, g))

    # --- update Ensembl IDs -------------------------------------
    if "ensembl_gene_id" not in new_df.columns:
        new_df["ensembl_gene_id"] = pd.NA

    def mapped_ensg(sym: str):
        return symbol2ensg.get(sym, pd.NA)

    ens_mapped = new_df["gene"].map(mapped_ensg)
    mask = ens_mapped.notna() & (new_df["ensembl_gene_id"] != ens_mapped)

    for g in new_df.loc[mask, "gene"].unique():
        old_ens = new_df.loc[(mask) & (new_df["gene"] == g),
                             "ensembl_gene_id"].unique()[0]
        new_ens = symbol2ensg.get(g)
        logger.info(
            f"Updating 'ensembl_gene_id' for {g} "
            f"from {old_ens} to {new_ens}")

    new_df.loc[mask, "ensembl_gene_id"] = ens_mapped[mask]

    # --- update variant column ----------------------------------
    if "variant" in new_df.columns and old_to_new:
        repl = (new_df["gene"] + " ")
        new_df["variant"] = (
            repl + new_df["variant"].str.replace(r"^[^ ]+ ",
                                                 "",
                                                 regex=True))

    # --- warn about unmapped ------------------------------------
    unmapped = set(new_df["gene"].unique()) - approved - set(alias2sym)
    if unmapped:
        logger.warning(f"{len(unmapped)} genes could not be mapped: "
                       f"{sorted(unmapped)}")

    logger.info("... done.")
    return new_df
