"""Compute trinucleotide context counts per gene from FASTA files.

This module provides functions to count the 32 pyrimidine-centered
trinucleotide contexts (e.g., ACA, ACT, ..., TTG, TTT) for each gene
based on the longest transcript in Ensembl FASTA files. These context
counts are used to calculate the probability that a mutation of a
specific type occurs in a given gene.
"""

import logging
import pandas as pd

from pathlib import Path
from collections.abc import Iterable

from Bio import SeqIO

from .locations import location_cds_fasta

logger = logging.getLogger(__name__)


def compute_contexts_by_gene(
    fasta_files: str | Path | list[str | Path] | None = None,
    restrict_to_db: pd.DataFrame | Iterable[str] | None = None) -> pd.DataFrame:
    """Build a 32-context table for the longest transcript of each gene.

    The routine scans one or more Ensembl FASTA files (cDNA, ncRNA,
    pseudogene …), keeps only the longest transcript per stable
    **Ensembl gene ID** (version stripped), and counts the 32
    pyrimidine-centred trinucleotide contexts (ACA … TTT).  If
    *restrict_to_db* is supplied, contexts are computed **only** for the
    Ensembl IDs occurring there, which can cut runtime by an order of
    magnitude when you analyse a subset of genes.

    Parameters
    ----------
    fasta_files : str | Path | list[str | Path] | None
        One path or a list of paths to Ensembl nucleotide FASTA files
        (e.g. *cdna.all.fa.gz*, *ncrna.fa.gz*, *pseudogenes.fa.gz*).
        If None, defaults to locations.location_cds_fasta.

    restrict_to_db : pandas.DataFrame | iterable[str] | None, optional
        • *None* (default) – process **all** genes in the FASTA(s).
        • DataFrame with an ``'ensembl_gene_id'`` column *or* any
          iterable of Ensembl IDs – compute contexts only for those IDs
          (faster).

    Returns
    -------
    pandas.DataFrame
        Index → stable Ensembl gene ID (version stripped)
        Columns → 32 context counts (ACA … TTT)
    """
    logger.info("Build a 32-context table for the longest transcript of each gene....")
    # ---- normalise inputs ------------------------------------------------
    if fasta_files is None:
        fasta_files = location_cds_fasta

    if isinstance(fasta_files, (str, Path)):
        fasta_files = [fasta_files]
    fasta_paths = [Path(p) for p in fasta_files]

    # set of IDs to keep (None means keep all)
    if restrict_to_db is None:
        keep_ids = None

    elif isinstance(restrict_to_db, pd.DataFrame):
        # keep rows that have both a variant name *and* a gene ID
        mask = (restrict_to_db["variant"].notna()
                & restrict_to_db["ensembl_gene_id"].notna())
        keep_ids = set(restrict_to_db.loc[mask, "ensembl_gene_id"].astype(str))

    else:
        keep_ids = set(map(str, restrict_to_db))

    # ---- precompute helpers ---------------------------------------------
    contexts32 = [first + second + third
                  for second in "CT"
                  for first in "ACGT"
                  for third in "ACGT"]
    comp = str.maketrans("ACGT", "TGCA")
    valid = set("ACGT")

    best_seq: dict[str, str] = {}

    def extract_ensg(desc: str) -> str | None:
        """Return stable ENSG (no version) or None."""
        for tok in desc.split():
            if tok.startswith("gene:ENSG"):
                ensg = tok.split(":")[1].split(".")[0]

                # ----- hard-coded HGNC catch-up (FAM153B) ----------------
                if ensg == "ENSG00000289731":
                    ensg = "ENSG00000182230"
                # ---------------------------------------------------------
                return ensg
        return None

    # ---- pass 1: pick longest transcript per gene -----------------------
    for fasta in fasta_paths:
        for rec in SeqIO.parse(fasta, "fasta"):
            ensg = extract_ensg(rec.description)
            if ensg is None:
                logger.warning("Header without Ensembl gene ID skipped: "
                               f"{rec.id}")
                continue

            if keep_ids is not None and ensg not in keep_ids:
                continue

            seq = str(rec.seq).upper()
            if len(seq) > len(best_seq.get(ensg, "")):
                best_seq[ensg] = seq

    # ---- pass 2: count contexts -----------------------------------------
    rows = []
    for ensg, seq in best_seq.items():
        cnt = {ctx: 0 for ctx in contexts32}
        for i in range(len(seq) - 2):
            tri = seq[i:i + 3]
            if set(tri) - valid:
                continue
            if tri[1] in "CT":
                cnt[tri] += 1
            else:
                cnt[tri.translate(comp)[::-1]] += 1
        rows.append((ensg, cnt))

    df = (pd.DataFrame.from_dict(dict(rows), orient="index")
            .astype(int)
            .sort_index())

    logger.info("...done.")
    print("")
    return df


def load_or_generate_contexts_by_gene(
        location_contexts_df: str | Path,
        fastas: str | Path | list,
        restrict_to_db: pd.DataFrame | Iterable[str] | None = None,
        force_generation: bool = False) -> pd.DataFrame:
    """Load or generate trinucleotide context counts by gene.

    Loads a pre-computed table of trinucleotide context counts per
    gene if it exists at the specified location. Otherwise,
    computes context counts from FASTA sequence files, saves to
    disk in CSV format, and returns the resulting DataFrame.

    Trinucleotide contexts (e.g., ACA, TCG) are essential for
    mutation rate modeling as mutation probabilities are highly
    context-dependent.

    Parameters
    ----------
    location_contexts_df : str or Path
        Path to the saved (or to-be-saved) contexts DataFrame.
        Should use .csv extension for portability.
        Example: "data/contexts_by_gene.csv"
    fastas : str, Path, or list of str/Path
        Path(s) to FASTA files containing gene sequences.
        Typically includes:
        - Homo_sapiens.GRCh38.cdna.all.fa.gz (coding + UTR)
        - Homo_sapiens.GRCh38.ncrna.fa.gz (non-coding RNAs)
        Can be a single path or list of paths.
    restrict_to_db : pd.DataFrame, Iterable[str], or None, default None
        Optional filter to restrict computation to specific genes:
        - None: Process all genes in FASTA files
        - DataFrame: Must contain 'ensembl_gene_id' column;
          only process genes in this column
        - Iterable: Collection of Ensembl gene IDs to process
        Filtering can significantly speed up computation.
        Only applied during generation (when force_generation=True
        or file doesn't exist).
    force_generation : bool, default False
        If True, recompute context counts even if a saved version
        exists. Use this when:
        - FASTA files have been updated
        - Gene filtering criteria changed
        - Cached file may be incomplete

    Returns
    -------
    pd.DataFrame
        DataFrame with trinucleotide context counts:
        - index: Gene symbols (HGNC)
        - columns: 32 trinucleotide contexts
          (e.g., 'ACA', 'ACC', 'ACG', 'ACT', ...)
        - values: Integer counts of each trinucleotide in
          each gene's sequence

    Notes
    -----
    The 32 trinucleotide contexts correspond to all possible
    combinations where the middle base can mutate:
    - 16 contexts with pyrimidine middle base (C, T)
    - Their reverse complements (total 32)

    Uses CSV format for storage, providing:
    - Human readability
    - Version control friendly
    - Easy inspection and validation
    - Cross-platform compatibility

    Context counts are used downstream to:
    1. Compute expected mutation rates per gene
    2. Normalize for sequence composition biases
    3. Calculate signature-specific mutation probabilities

    Examples
    --------
    >>> # Load or generate from single FASTA
    >>> contexts = load_or_generate_contexts_by_gene(
    ...     "data/contexts_by_gene.csv",
    ...     "data/Homo_sapiens.GRCh38.cdna.all.fa.gz")

    >>> # Generate from multiple FASTAs
    >>> contexts = load_or_generate_contexts_by_gene(
    ...     "data/contexts_by_gene.csv",
    ...     ["data/cdna.fa.gz", "data/ncrna.fa.gz"])

    >>> # Restrict to specific genes in mutation database
    >>> contexts = load_or_generate_contexts_by_gene(
    ...     "data/contexts_by_gene.csv",
    ...     fasta_files,
    ...     restrict_to_db=mutation_db)

    >>> # Force regeneration
    >>> contexts = load_or_generate_contexts_by_gene(
    ...     "data/contexts_by_gene.csv",
    ...     fasta_files,
    ...     force_generation=True)

    >>> # Check context counts for a specific gene
    >>> print(contexts.loc['TP53'])
    """
    location_contexts_df = Path(location_contexts_df)

    if location_contexts_df.exists() and not force_generation:
        logger.info(
            f"Loading context table from {location_contexts_df}")
        df = pd.read_csv(location_contexts_df, index_col=0)
        logger.info("... done.")
        return df

    logger.info(
        f"Generating context table from {fastas}")

    df = compute_contexts_by_gene(fastas, restrict_to_db)

    df.to_csv(location_contexts_df, index=True)
    logger.info(f"Saved table to {location_contexts_df}")
    logger.info("... done.")

    return df
