"""Extract and annotate variants from mutation databases.

This module provides functions to extract unique protein-level
variants from MAF-style DataFrames, ensuring each variant maps to a
single gene and chromosome. It handles position inconsistencies and
assigns mutational types when available.
"""

import os
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def extract_variants_from_db(
    db: pd.DataFrame, position_tolerance: int = 3
) -> pd.DataFrame:
    """Extract unique variants from a mutation DataFrame.

    Ensures that each variant maps to exactly one gene, chromosome,
    and approximately one start position. Small inconsistencies in
    Start_Position (within ±position_tolerance bp) are allowed and
    averaged. Larger inconsistencies are assign NaN as
    'Start_Position' (those cases are rare and involve splice
    variants).

    Args:
        db (pd.DataFrame): Must contain columns:
            - 'variant': str
            - 'gene': str
            - 'Chromosome': str
            - 'Start_Position': int

        position_tolerance (int): Maximum allowed difference in
            Start_Position, default is 3 bp.

        num_tumor_tolerance (int): If a variant with Start_Position
            differences > position_tolerance appears in ≤ this many
            tumors, it will be included with NaN Start_Position and
            a warning. If it appears in more tumors, an error is raised.

    Returns:
        pd.DataFrame: Indexed by 'variant', with columns:
            - 'gene'
            - 'ensembl_gene_id'
            - 'Chromosome'
            - 'Start_Position' (rounded mean or NaN)

    """
    logger.info("Extracting all variants from db...")

    grouped = db.groupby("variant")
    records: list = []

    for variant, group in grouped:
        genes = group["gene"].unique()
        chroms = group["Chromosome"].unique()
        starts = group["Start_Position"].unique()

        if len(genes) > 1 or len(chroms) > 1:
            raise ValueError(
                f"Variant {variant!r} maps to multiple genes or "
                f"chromosomes:\n  Genes: {genes}\n  Chromosomes: "
                f"{chroms}"
            )

        # ── Ensembl ID: keep most frequent, warn on ties ────────────
        gene_id_counts = group["ensembl_gene_id"].value_counts()
        ensembl_id = gene_id_counts.index[0]
        if len(gene_id_counts) > 1:
            logger.warning(
                f"Variant {variant!r} maps to multiple Ensembl IDs "
                f"{list(gene_id_counts.index)}; using {ensembl_id}"
            )

        # ── Start position handling ────────────────────────────────
        mean_start: float | int = np.nan  # <- always defined

        if np.ptp(starts) > position_tolerance:
            logger.warning(
                f"Variant {variant!r} shows {len(starts)} distinct "
                f"Start_Positions across {len(group)} tumors that "
                f"exceed the ±{position_tolerance} bp tolerance: "
                f"{starts}.  Assigning NaN."
            )
        else:
            if len(starts) > 1:
                logger.info(
                    f"Variant {variant!r} has {len(starts)} "
                    f"Start_Positions within tolerance "
                    f"(±{position_tolerance} bp): {starts}.  "
                    f"Using mean."
                )
            # compute safely even if all values are NaN
            pos_mean = np.nanmean(starts)
            if not np.isnan(pos_mean):
                mean_start = int(round(pos_mean))

        records.append(
            (variant, genes[0], ensembl_id, chroms[0], mean_start)
        )

    result = pd.DataFrame(
        records,
        columns=[
            "variant",
            "gene",
            "ensembl_gene_id",
            "Chromosome",
            "Start_Position",
        ],
    ).set_index("variant")

    logger.info("... done.")
    print("")
    return result


def annotate_variants_with_types(
    variants_df: "pd.DataFrame", db: "pd.DataFrame"
) -> "pd.DataFrame":
    """Annotate variants with the mutational type(s).

    For each variant in `variants_df` adds the type(s) observed for
    that variant in the mutation `db`.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        Table whose rows correspond to unique protein-level variants
        (index named ``"variant"`` **or** a column called ``"variant"``).

    db : pandas.DataFrame
        Full mutation catalogue.  Must contain the columns

        - 'variant' – same identifiers as in *variants_df*
        - 'type' – SBS context string (e.g. 'G[C>T]T').

    Returns
    -------
    pandas.DataFrame
        Copy of `variants_df` with one extra column:

        - single type → stored as a plain string for speed/memory
        - multiple types → stored as a list of strings
        - no entry in *db* → NaN
    """
    logger.info("Annotating variants with types...")
    out = variants_df.copy()

    # Build Series: variant -> ndarray(unique types)
    uniq_types = db.groupby("variant")["type"].unique()

    # Collapse to string OR list (for variants with multiple types)
    tidy_types = uniq_types.map(
        lambda arr: arr[0] if len(arr) == 1 else list(arr)
    )

    # 3. align and add
    if "variant" in out.columns:
        key = out["variant"]
    else:  # index is the key
        key = out.index

    out["mut_types"] = tidy_types.reindex(key).values

    logger.info("... done.")
    print("")
    return out


def load_or_generate_variants_with_types(
    location_df: str, db: pd.DataFrame, force_generation: bool = False
) -> pd.DataFrame:
    """Load or generate a DataFrame of unique variants with type annotations.

    Loads a preprocessed variant database with annotated mutation
    types if it exists at the specified location. Otherwise,
    extracts unique variants from the mutation database, annotates
    them with their mutation types, saves to disk in CSV format,
    and returns the resulting DataFrame.

    This function creates a reference table of all unique variants
    observed in the dataset, with their associated mutation type
    classifications (e.g., C>A, C>G, indel categories).

    Parameters
    ----------
    location_df : str or Path
        Path to the saved (or to-be-saved) variants DataFrame.
        Should use .csv extension for portability.
        Example: "data/variants_with_types.csv"
    db : pd.DataFrame
        Mutation database from which to extract variants.
        Must contain columns:
        - variant : str
            Variant identifier (unique mutation signature)
        - gene : str
            Gene symbol where variant occurs
        - Chromosome : str
            Chromosome name
        - Start_Position : int
            Genomic position
        Additional columns used for type annotation.
    force_generation : bool, default False
        If True, regenerate the variants DataFrame even if a
        saved version exists. Use this when:
        - The mutation database has been updated
        - Type annotation logic has changed
        - Cached file may be incomplete or corrupted

    Returns
    -------
    pd.DataFrame
        DataFrame of unique variants with columns:
        - variant : str
            Variant identifier (index)
        - type : str or list
            Mutation type classification(s) for this variant
        - gene : str
            Example gene where this variant occurs
        - Chromosome : str
            Chromosome location
        - Start_Position : int
            Genomic position
        And additional annotation columns.

    Notes
    -----
    The function validates that the number of variants loaded
    matches the number of unique variants in the input database.
    If counts don't match, it automatically regenerates the file.

    Uses CSV format for storage, providing:
    - Human readability
    - Version control friendly (diff-able)
    - Cross-platform compatibility
    - Easy inspection in spreadsheet software

    Examples
    --------
    >>> # Load or generate variants from mutation database
    >>> variants = load_or_generate_variants_with_types(
    ...     "data/variants_with_types.csv",
    ...     mutation_db)

    >>> # Force regeneration after database update
    >>> variants = load_or_generate_variants_with_types(
    ...     "data/variants_with_types.csv",
    ...     mutation_db,
    ...     force_generation=True)

    >>> # Check number of unique variants
    >>> print(f"Found {len(variants)} unique variants")
    """
    if not os.path.exists(location_df) or force_generation:
        all_variants = extract_variants_from_db(db)
        all_variants = annotate_variants_with_types(all_variants, db)
        all_variants.to_csv(location_df, index=True)
        logger.info(f"Saved variants data frame to {location_df}")
        logger.info("... done.")
    else:
        logger.info(
            "Loading data frame with all variants and their types "
            f"from {location_df}"
        )
        all_variants = pd.read_csv(location_df, index_col=0)
        logger.info("... done.")
        if len(all_variants) != db["variant"].nunique():
            logger.info(
                f"Number of variants loaded from {location_df} "
                "does not match those of `db`, forcing generation..."
            )
            all_variants = load_or_generate_variants_with_types(
                location_df, db, True
            )

    return all_variants


def build_arrays_for_cov_effect_estimation(
    mu_j_m: np.ndarray, db: pd.DataFrame, covariates_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Build arrays needed for the covariates regression.

    Restrict a mu_j_m array to silent variants with covariate info. In
    other words, removes all non-silent variants that also have
    missing covariates for all covariates.

    Build the corresponding covariate matrix and presence matrix
    (whether or not the variant is on the tumor).

    Parameters
    ----------
    mu_j_m : ndarray (n_tumours, n_variants)
        Mutation-rate components, **columns ordered alphabetically by
        variant label** (same order as ``covariates_df.index`` when
        sorted).

    db : pandas.DataFrame
        Full mutation catalogue. Must contain the columns:
        - 'variant'
        - 'Variant_Classification' (e.g. 'Silent',
          'Missense_Mutation', etc)

    covariates_df : pandas.DataFrame
        Table indexed by 'variant' with one or more columns whose
        names start with "cov_" (e.g. 'cov_iz_early', etc.).

    Returns
    -------
    mu_restricted : ndarray (n_tumours, n_restricted) Slice of the
        original array containing only the selected
        silent-and-annotated variants.

    cov_matrix : ndarray (n_restricted, n_covariates) The numeric
        values of the "cov_" columns for the same variants, in the
        same order as the columns of `mu_restricted`.

    presence : ndarray  (n_tumours, n_restricted)
        Binary matrix: 1 if that variant is present in that tumour
        according to `db`, 0 otherwise.

    Notes
    -----
    - Assumes that the variant axis (axis 1) of `mu_j_m` follows the
      same alphabetical order as the index of `covariates_df`.
    - Covariate columns in `covariates_df` have to start with "cov_"
      to be considered covariates.

    """
    # Variants that are Silent in the main database
    silent_set = set(
        db.loc[db["Variant_Classification"] == "Silent", "variant"]
    )

    # Rows that have any non-NaN cov_ value
    cov_cols = [
        c for c in covariates_df.columns if c.startswith("cov_")
    ]
    has_cov = ~covariates_df[cov_cols].isna().all(axis=1)

    # Final selection: silent AND has_cov
    variants_order = sorted(
        covariates_df.index
    )  # assumes alphabetical
    keep_mask = [
        (v in silent_set) and has_cov[v] for v in variants_order
    ]
    variant_keep = [v for v, k in zip(variants_order, keep_mask) if k]

    # Slice arrays / frames
    mu_restricted = mu_j_m[:, keep_mask]  # (n_tumours, n_restricted)
    cov_matrix = covariates_df.loc[variant_keep, cov_cols].to_numpy()

    # Build tumour × variant presence matrix
    tumour_order = sorted(db["Tumor_Sample_Barcode"].unique())
    tumour_to_row = {t: i for i, t in enumerate(tumour_order)}
    variant_to_col = {v: j for j, v in enumerate(variant_keep)}

    presence = np.zeros(
        (len(tumour_order), len(variant_keep)), dtype=np.int8
    )

    sub_db = db[db["variant"].isin(variant_keep)]
    for t, v in (
        sub_db[["Tumor_Sample_Barcode", "variant"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        presence[tumour_to_row[t], variant_to_col[v]] = 1

    return mu_restricted, cov_matrix, presence
