"""Load maf files.

Validate each maf file to leave entries that contain real SNVs only,
retain only necessary columns, and merge all files.
"""

__author__ = "Jorge Alfaro-Murillo"

import os
import logging
import pandas as pd

from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from .update_gene_names import update_genes_with_gene_set

from .constants import nucleotides
from .constants import chromosomes
from .constants import reverse_complement

from .locations import location_hgnc_complete_set

logger = logging.getLogger(__name__)


# These are the original columns of the MAF files that will remain
# after compacting the data base. Also new constructed columns will
# remain as well, those are currently:
# -'gene'
# -'variant'
# -'type'
#
# For a full description of all the columns check
# :var:`constants.maf_column_descriptions`
cols_to_keep = [
    "Tumor_Sample_Barcode",
    "Chromosome",
    # # Gene is the Ensembl stable gene identifier
    # 'Gene',  # it will change to 'ensembl_gene_id'
    # 'CONTEXT',
    # 'Reference_Allele',
    # 'Tumor_Seq_Allele2',
    "Variant_Classification",
    # 'Variant_Type',
    # 'Codons',
    "Start_Position",
]


def filter_db(db, variant_type="SNP"):
    """Filter a MAF-style DataFrame to retain only `variant_type`.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation annotation format (MAF) DataFrame. Must contain a
        'Variant_Type' column.

    variant_type : str or list
        Type(s) of mutation to filter to, so far 'SNP' or 'ID' are
        supported. If multiple types, give them as a list,
        e.g. ['INS', 'DEL'].

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only single-nucleotide variants
        (rows where 'Variant_Type' == 'SNP').

    Raises
    ------
    KeyError
        If the 'Variant_Type' column is missing.

    """
    if "Variant_Type" not in db.columns:
        logger.error(
            "Missing 'Variant_Type' column in input DataFrame."
        )
        raise KeyError(
            "Input DataFrame must contain 'Variant_Type' column."
        )

    variant_type = variant_type.upper()

    # Handle COSMIC signature class aliases
    if variant_type in ("SBS", "RNA-SBS"):
        variant_type = "SNP"
    elif variant_type == "DBS":
        variant_type = "DBP"
    elif (variant_type == "INDEL") or (variant_type == "ID"):
        variant_type = ["INS", "DEL"]

    if isinstance(variant_type, list):
        return db[db["Variant_Type"].isin(variant_type)].copy()
    else:
        return db[db["Variant_Type"] == variant_type].copy()


def validate_chromosome(df):
    """Validate 'Chromosome' column and filter out invalid entries.

    This function checks whether each value in the 'Chromosome' column
    belongs to a predefined list of valid chromosome identifiers
    (`chromosomes`). It removes rows with invalid values and returns
    the cleaned DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a 'Chromosome' column.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows with valid chromosome
        values.

    """
    invalid_entries = df[~df["Chromosome"].isin(chromosomes)]

    if not invalid_entries.empty:
        logger.warning(
            f"Removed {len(invalid_entries)} invalid chromosomes"
        )
        logger.debug(
            f"Invalid rows:\n{invalid_entries[cols_to_keep]}"
        )
    else:
        logger.debug("All entries have valid chromosomes")

    return df[df["Chromosome"].isin(chromosomes)]


def validate_alleles_snv(df):
    """Validate SNV alleles in a MAF dataframe.

    Filters rows where both 'Reference_Allele' and 'Tumor_Seq_Allele2'
    are valid nucleotides of length 1.  Removes indels, ambiguous
    codes, and malformed entries.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'Reference_Allele' and 'Tumor_Seq_Allele2'
        columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with only valid SNVs (one-letter alleles).

    """
    is_valid_ref = df["Reference_Allele"].str.len() == 1
    is_valid_alt = df["Tumor_Seq_Allele2"].str.len() == 1

    is_nuc_ref = df["Reference_Allele"].str.upper().isin(nucleotides)
    is_nuc_alt = df["Tumor_Seq_Allele2"].str.upper().isin(nucleotides)

    is_valid = is_valid_ref & is_valid_alt & is_nuc_ref & is_nuc_alt
    invalid_entries = df[~is_valid]

    relevant_cols = cols_to_keep + [
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]

    if not invalid_entries.empty:
        logger.warning(
            f"Removed {len(invalid_entries)} invalid alleles"
        )
        logger.debug(
            f"Invalid rows:\n{invalid_entries[relevant_cols]}"
        )
    else:
        logger.debug("All entries have valid one-letter alleles")

    return df[is_valid]


def validate_context_snv(df, *, context_length=11):
    """Validate the 'CONTEXT' column for expected format and characters.

    This function ensures that the CONTEXT column contains strings of
    the correct length and composed only of valid DNA nucleotide
    characters (A, C, G, T). Rows failing either condition are removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame expected to contain a 'CONTEXT' column.

    context_length : int, optional
        The expected length of the context string. Default is 11.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with only rows containing valid CONTEXT
        strings.

    Notes
    -----
    Assumes a global set `nucleotides` is defined, containing
    valid characters: {'A', 'C', 'G', 'T'}.
    """
    is_valid_length = df["CONTEXT"].str.len() == context_length
    is_valid_chars = (
        df["CONTEXT"]
        .str.upper()
        .apply(lambda s: set(s).issubset(nucleotides))
    )

    is_valid_context = is_valid_length & is_valid_chars
    invalid_entries = df[~is_valid_context]

    if not invalid_entries.empty:
        logger.warning(
            f"Removed {len(invalid_entries)} rows due to invalid CONTEXT "
            f"(not length {context_length} or invalid characters)."
        )
        logger.debug(
            f"Invalid rows:\n{invalid_entries[cols_to_keep]}"
        )
    else:
        logger.debug("All CONTEXT entries are valid.")

    return df[is_valid_context]


def validate_reference_matches_context_snv(df, *, context_length=11):
    """Validate that Reference_Allele matches center of CONTEXT.

    This function checks that the value in the 'Reference_Allele'
    column matches the central base of the 'CONTEXT' string. Rows
    where these values do not agree are removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame expected to contain 'Reference_Allele' and
        'CONTEXT' columns.

    context_length : int, optional
        The expected length of the context string. The function
        compares the Reference_Allele to the middle base in CONTEXT.
        Default is 11.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with rows where Reference_Allele matches
        the central base of CONTEXT.

    Notes
    -----
    The central base is taken to be position (context_length - 1) // 2.
    """
    mid_index = (context_length - 1) // 2
    is_match = df["Reference_Allele"] == df["CONTEXT"].str[mid_index]
    invalid_entries = df[~is_match]

    if not invalid_entries.empty:
        logger.warning(
            f"Removed {len(invalid_entries)} rows where Reference_Allele "
            f"does not match CONTEXT[{mid_index}]."
        )
        logger.debug(
            "Invalid rows:\n"
            f"{invalid_entries[cols_to_keep + ['Reference_Allele']]}"
        )
    else:
        logger.debug(
            "All entries have Reference_Allele matching the "
            "center of CONTEXT."
        )

    return df[is_match]


def validate_full(df, *, variant_type="SNP", context_length=11):
    """Run all validation checks on a MAF DataFrame.

    Applies a sequence of filters to ensure the MAF file contains
    valid entries only. The following checks are applied in order:

    1. Valid chromosome values.
    2. For 'SNP': Valid alleles (single-letter, A/C/G/T only).
    3. For 'SNP': Valid CONTEXT strings (length and valid characters).
    4. For 'SNP': Reference allele matches the center base of CONTEXT.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame expected to contain columns:
        'Chromosome', 'Reference_Allele', 'Tumor_Seq_Allele2',
        and 'CONTEXT'.

    context_length : int, optional
        Expected length of the CONTEXT string (default is 11).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only valid SNV entries.

    """
    logger.debug("Starting full MAF validation pipeline.")
    new_df = validate_chromosome(df)
    if variant_type == "SNP":
        new_df = validate_alleles_snv(new_df)
        new_df = validate_context_snv(
            new_df, context_length=context_length
        )
        new_df = validate_reference_matches_context_snv(
            new_df, context_length=context_length
        )
    logger.debug(
        f"Validation complete. Remaining rows: {len(new_df)}."
    )
    return new_df


def create_mutation_type_column(df, variant_type="SNP", **kwargs):
    """Add a mutation type column in COSMIC-style.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with columns:
        'CONTEXT', 'Reference_Allele', 'Tumor_Seq_Allele2'.

    variant_type : str
        Type of mutation to filter to, so far 'SNP' or 'ID' are
        supported.

    **kwargs : dict
        Additional keyword arguments to be passed to the appropiate
        create_mutation_type_column_`variant_type` function.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with a new column 'type'.

    """
    variant_type = variant_type.upper()

    # Handle COSMIC signature class aliases
    if variant_type in ("SBS", "RNA-SBS"):  # alias for SNP
        variant_type = "SNP"

    if variant_type == "SNP":
        return create_mutation_type_column_snp(df, **kwargs)

    if variant_type == "INDEL":  # alias
        variant_type = "ID"

    if variant_type == "ID":
        return create_mutation_type_column_id(df, **kwargs)


def create_mutation_type_column_snp(df, *, context_length=11):
    """Add a mutation type column in COSMIC-style SBS96 format.

    Ensures all mutations are represented with a pyrimidine (C or T)
    as the reference base, consistent with COSMIC SBS96 convention.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with columns:
        'CONTEXT', 'Reference_Allele', 'Tumor_Seq_Allele2'.

    context_length : int, optional
        Length of the CONTEXT sequence. Default is 11. The reference
        base is assumed to be at the center.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with a new column 'type' in
        COSMIC SBS96 format.

    """
    df = df.copy()
    mid = (context_length - 1) // 2

    def get_type(row):
        context = row["CONTEXT"]
        ref = row["Reference_Allele"]
        alt = row["Tumor_Seq_Allele2"]

        if ref in {"C", "T"}:
            left = context[mid - 1]
            right = context[mid + 1]
            return f"{left}[{ref}>{alt}]{right}"
        else:
            # Reverse complement context and alleles
            rc_context = reverse_complement(context)
            rc_ref = reverse_complement(ref)
            rc_alt = reverse_complement(alt)
            left = rc_context[mid - 1]
            right = rc_context[mid + 1]
            return f"{left}[{rc_ref}>{rc_alt}]{right}"

    df["type"] = df.apply(get_type, axis=1)

    return df


def reduce_context_to_trinucleotides(df, *, context_length=11):
    """Trim CONTEXT column to the central trinucleotide.

    Reduces each CONTEXT string to its three central bases,
    which correspond to the standard trinucleotide used in
    mutational signature analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'CONTEXT' column with
        fixed-length nucleotide strings.

    context_length : int, optional
        Total length of the CONTEXT string. Default is 11.
        The center of this sequence is assumed to be the
        reference position.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with new 'context' column, which is
        the central 3-base substring of 'CONTEXT'.

    Notes
    -----
    Assumes that CONTEXT strings are of the specified length,
    and that the middle base corresponds to the SNV site.

    """
    df = df.copy()
    mid = (context_length - 1) // 2
    df["context"] = df["CONTEXT"].str[mid - 1 : mid + 2]
    return df


def build_id83_lookup(seqinfo_dir):
    """Create a DataFrame with all mutation type.

    Concatenate all *_seqinfo.txt files in `seqinfo_dir`
    and return a single DataFrame to be used as a lookup.

    The seqinfo.txt files come from running
    SigProfilerMatrixGeneratorFunc

    """

    def _read_seqinfo_file(path):
        """Read one chrN_seqinfo.txt into a tidy DataFrame."""
        cols = [
            "Tumor_Sample_Barcode",
            "chr_num",
            "pos",
            "id83",
            "ref",
            "alt",
            "strand_flag",
        ]
        df = pd.read_table(
            path, header=None, names=cols, usecols=[0, 1, 2, 3]
        )
        df["Chromosome"] = "chr" + df.chr_num.astype(str)
        df.drop("chr_num", axis=1, inplace=True)
        return df

    frames = [
        _read_seqinfo_file(fp)
        for fp in Path(seqinfo_dir).glob("*_seqinfo.txt")
    ]
    lookup = pd.concat(frames, ignore_index=True)
    lookup.drop_duplicates(inplace=True)
    return lookup


def create_mutation_type_column_id(
    df: pd.DataFrame, seqinfo_dir: str | Path, *, max_shift: int = 2
) -> pd.DataFrame:
    """Annotate a mutation Data-frame with its COSMIC ID-83 type.

    The function first calls :func:`build_id83_lookup` to read all
    *_seqinfo.txt files produced by
    :mod:`SigProfilerMatrixGenerator`. It merges those rows with the
    MAF on (sample, chromosome, coordinate).

    Because TCGA MAF coordinates are left-aligned inside the indel
    whereas VCF/seqinfo coordinates point to the anchor base before
    the indel, the same event can differ by up to its length. The
    merge is therefore retried for shifts −max_shift... +max_shift bp
    until a match is found or every seqinfo row has been used once.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns 'Tumor_Sample_Barcode', 'Chromosome',
        'Start_Position'. Other columns are preserved.
    seqinfo_dir : str
        Directory that holds the per-chromosome *_seqinfo.txt files
        written with seqInfo=True.
    max_shift : int
        Largest offset (in bp) to try left and right when the exact
        coordinate fails. Default 2.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with one extra column 'type' containing the
        ID-83 label (e.g. '1:Del:C:4'). Variants that are not
        indels—or could not be reconciled within ±max_shift—get
        NaN.

    Notes
    -----
    - Some substitutions (like _splice, or stop_gained) have no ID-83
      counterpart and therefore stay NaN.
    - Keep the intermediate id83 column if you need strand
      information (it still carries the U:, T:, ... prefix).

    """
    # All SigProfilerMatrixGeneratorFunc results in a DataFrame
    df = df.reset_index(drop=True).copy()
    lookup_all = build_id83_lookup(seqinfo_dir)

    samples_here = df[
        "Tumor_Sample_Barcode"
    ].unique()  # probably just one
    lookup = lookup_all[
        lookup_all["Tumor_Sample_Barcode"].isin(samples_here)
    ]

    df["pos_tmp"] = df.Start_Position

    # First try for exact location matching
    merged = df.merge(
        lookup,
        how="left",
        left_on=["Tumor_Sample_Barcode", "Chromosome", "pos_tmp"],
        right_on=["Tumor_Sample_Barcode", "Chromosome", "pos"],
    )

    # Else, start trying to shift and stop when either:
    #     - every seqinfo row has been used,  OR
    #     - nothing is missing any more. (Well only those mutations
    #       without calls from SigProfilerMatrixGenerator)
    n_lookup = len(lookup)
    n_df = len(df)
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue

        still_missing = merged.id83.isna()
        if not still_missing.any():  # all done
            break

        # if (missing + lookup_rows) == total_rows, the remaining NA
        # are variants that never occur in seqinfo, so we can stop
        if still_missing.sum() + n_lookup == n_df:
            break

        retry = df.loc[still_missing].copy()
        retry["pos_tmp"] = retry.Start_Position + shift

        merged.loc[still_missing, "id83"] = retry.merge(
            lookup,
            how="left",
            left_on=["Tumor_Sample_Barcode", "Chromosome", "pos_tmp"],
            right_on=["Tumor_Sample_Barcode", "Chromosome", "pos"],
        ).id83.values

    unresolved = merged.id83.isna().sum()
    expected_nan = len(df) - len(lookup)

    if unresolved != expected_nan:
        logger.warning(
            "In the ID-83 annotation for "
            f"{df['Tumor_Sample_Barcode'].unique()} there are {unresolved} "
            f"unresolved rows, but there were {expected_nan} expected "
            "(len(df) − len(lookup)). This suggests at least one "
            "seqinfo line was matched to multiple MAF rows. "
            "Try reducing `max_shift`."
        )

    # Clean up the type column, drop other created columns
    merged["type"] = merged["id83"].str.split(":", n=1).str[1]
    merged.drop(columns=["pos_tmp", "pos", "id83"], inplace=True)

    return merged


def compact_data(df, *, variant_type="SNP", **kwargs):
    """Reduce MAF DataFrame to minimal columns to continue analysis.

    This function extracts essential information from a MAF-like
    DataFrame, including gene name, amino acid variant, and mutation
    context. It renames and constructs key columns, computes the
    standard 96-type mutation label, and optionally reduces the
    CONTEXT field to a central trinucleotide.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with standard MAF fields. Must contain at least:

        - Tumor_Sample_Barcode: ID of the sample
        - Hugo_Symbol: The gene
        - Gene: The Ensembl stable gene identifier
        - HGVSp_Short: The variant (but does not include the gene)
        - Chromosome: The chromosome, any of: chr1, chr2, ..., chr21,
          chrX, chrY
        - For SNP, CONTEXT: Generally 11-nucleotide context. If it is different
          then make sure to change `context_lenght`
        - Reference_Allele: The allele in the reference genome at this
          position
        - Tumor_Seq_Allele2: The other allele observed in the tumor
          (usually the variant)
        - Tumor_Seq_Allele1: One allele observed in the tumor sample
          (can be reference or alt) NOT NECESSARY but here for
          reference in case you wondered why the 2 in the previous one
        - Variant_Classification: type of mutation, like synonymous,
          nonsynonymous, missense, nonsense, splice site, etc

    variant_type : str
        Type(s) of mutation to filter to, so far 'SNP' or 'ID' are
        supported.

    **kwargs : dict
        Additional keyword arguments to be passed to
        `:func:create_mutation_type_column`.

    Returns
    -------
    pandas.DataFrame
        A reduced DataFrame containing only the columns in
        :var:`cols_to_keep`, plus added columns:
        - 'gene'
        - 'variant' (in HGVS protein-level nomenclature)
        - 'ensembl_gene_id'
        - 'type' (standard W[X>Y]Z format for 'SNP')

    """
    df = df.copy()

    # Drop entries that do not have a gene
    bad_mask = df["Hugo_Symbol"].eq("Unknown") | df["Gene"].isna()
    if bad_mask.any():
        logger.warning(
            f"Dropping:\n{df.loc[bad_mask]}\n"
            "because of gene='Unknown' and/or missing Ensembl gene ID."
        )
        df = df.loc[~bad_mask]

    df["gene"] = df["Hugo_Symbol"]  # usual gene name
    df["ensembl_gene_id"] = df[
        "Gene"
    ]  # Ensembl stable gene identifier
    df["variant"] = df["Hugo_Symbol"] + " " + df["HGVSp_Short"]

    df = create_mutation_type_column(
        df, variant_type=variant_type, **kwargs
    )

    final_cols_to_keep = cols_to_keep + [
        "gene",
        "ensembl_gene_id",
        "variant",
        "type",
    ]

    df = df[final_cols_to_keep]

    return df


def process_single_maf(maf_file, variant_type="SNP", **kwargs):
    """Process a single MAF file into a cleaned and compact DataFrame.

    This function reads a MAF file, filters and validates for
    `variant_type`, filters invalid rows, and reduces the DataFrame to
    key mutation data including the mutation type.

    Parameters
    ----------
    maf_file : pathlib.Path
        Path to the MAF file to be processed.

    variant_type : str
        Type of mutation to filter to, so far 'SNP' or 'ID' are
        supported.

    **kwargs : dict
        Additional keyword arguments to be passed to
        `func:validate_full` and `func:compact_data`.

    Returns
    -------
    pandas.DataFrame or None
        A cleaned and compact mutation DataFrame. Returns None if the
        file could not be processed (e.g., invalid format or error on
        reading).

    """
    try:
        logger.debug(f"Reading MAF file: {maf_file.name}")

        df = pd.read_csv(
            maf_file, sep="\t", comment="#", low_memory=False
        )

        df = filter_db(df, variant_type)
        df = validate_full(df, variant_type=variant_type)
        df = compact_data(df, variant_type=variant_type, **kwargs)

        logger.debug(f"Successfully processed: {maf_file.name}")
        return df

    except Exception as e:
        logger.warning(f"Failed to process {maf_file.name}: {e}")
        return None


def load_validate_compact_all_maf_files_parallel(
    maf_dir, variant_type="SNP", **kwargs
):
    """Load and process all MAF files in a directory in parallel.

    This function scans a directory for MAF files, processes each file
    in parallel, validates and compacts the mutation data, and
    combines the resulting DataFrames into a single table.

    Parameters
    ----------
    maf_dir : str or pathlib.Path
        Directory containing .maf files to process.

    variant_type : str
        Type of mutation to filter to, so far 'SNP' or 'ID' are
        supported. Default 'SNP'.

    **kwargs : dict
        Additional keyword arguments to be passed to
        `func:process_single_maf`.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing all valid and compacted
        mutation records from all processed MAF files.

    Raises
    ------
    ValueError
        If no valid MAF files were found or all processing failed.

    """
    maf_dir = Path(maf_dir)
    all_files = [
        f
        for f in maf_dir.iterdir()
        if f.is_file() and f.suffix == ".maf"
    ]

    if not all_files:
        raise ValueError("No .maf files found in directory.")

    logger.info(f"Found {len(all_files)} MAF files to process.")

    func = partial(
        process_single_maf, variant_type=variant_type, **kwargs
    )

    with Pool(processes=cpu_count()) as pool:
        all_dataframes = pool.map(func, all_files)

    all_dataframes = [df for df in all_dataframes if df is not None]

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(
            "Successfully processed "
            f"{len(all_dataframes)}/{len(all_files)} files."
        )
        return combined_df
    else:
        raise ValueError(
            "No valid MAF files were " "successfully processed."
        )


def generate_compact_db(
    maf_dir: str,
    *,
    signature_class: str = "SBS",
    location_db: str | None = None,
    location_gene_set: str | None = location_hgnc_complete_set,
    **kwargs,
) -> pd.DataFrame:
    """Generate a compact mutation database from MAF files.

    Always processes raw MAF files to generate a compact mutation
    database. Optionally saves the result to disk in Parquet format.
    This function does NOT check for existing cached files - it
    always regenerates from scratch. Use
    :func:`load_or_generate_compact_db` for caching behavior.

    Parameters
    ----------
    maf_dir : str or Path
        Path to directory containing raw MAF files to process.
        All .maf or .maf.gz files in this directory will be
        loaded and consolidated.
    signature_class : str, default "SBS"
        COSMIC signature class to include in the database.
        Must be one of:
        - "SBS": Single base substitution signatures
        - "DBS": Doublet base substitution signatures
        - "ID": Insertion/deletion signatures
        - "CN": Copy number signatures
        - "SV": Structural variant signatures
        - "RNA-SBS": RNA single base substitution signatures
    location_db : str, Path, or None, default None
        Path where compact database will be saved (Parquet format).
        If None, database is generated in memory only and not saved
        to disk. Example: "data/tcga_compact_db_sbs.parquet"
    location_gene_set : str, Path, or None, default location_hgnc_complete_set
        Path to gene set file for standardizing gene names and
        Ensembl IDs. Gene identifiers will be updated using
        :func:`update_gene_names.update_genes_with_gene_set`.
        Defaults to HGNC complete set for consistent gene
        naming. Set to None to skip gene name updates.
    **kwargs : dict
        Additional keyword arguments passed to
        :func:`load_validate_compact_all_maf_files_parallel`.

        **Required for ID signature class:**
        - seqinfo_dir : str or Path
            Directory containing *_seqinfo.txt files generated by
            SigProfilerMatrixGenerator (with seqInfo=True). These
            files provide COSMIC ID-83 mutation type annotations
            for indel variants.

    Returns
    -------
    pd.DataFrame
        Compact mutation database with standardized columns:
        - gene : str
            Gene symbol (HGNC)
        - ensembl_gene_id : str
            Ensembl gene identifier
        - Tumor_Sample_Barcode : str
            Sample identifier
        - Chromosome : str
            Chromosome name
        - Start_Position : int
            Genomic position
        - variant : str
            Variant identifier
        - type : str
            Mutation type classification
        And additional columns depending on signature_class.

    Notes
    -----
    This function always regenerates the database from MAF files,
    even if a cached version exists. For better performance with
    repeated calls, use :func:`load_or_generate_compact_db` which
    caches results.

    Gene names are automatically standardized using HGNC
    nomenclature unless location_gene_set is set to None.

    **Important for ID (indel) signature class:**
    When signature_class="ID", you MUST provide seqinfo_dir in
    kwargs. This directory should contain *_seqinfo.txt files
    generated by running SigProfilerMatrixGenerator with
    seqInfo=True. These files are needed to annotate indels
    with their COSMIC ID-83 mutation types.

    Examples
    --------
    >>> # Generate SBS database and save
    >>> db = generate_compact_db(
    ...     "data/tcga/maf_files",
    ...     signature_class="SBS",
    ...     location_db="data/mutations_sbs.parquet")

    >>> # Generate in memory only (no save)
    >>> db = generate_compact_db(
    ...     "data/tcga/maf_files",
    ...     signature_class="SBS")

    >>> # Generate ID database
    >>> # (requires seqinfo_dir, provide custom path if needed)
    >>> db_indels = generate_compact_db(
    ...     "data/tcga/maf_files",
    ...     signature_class="ID",
    ...     location_db="data/mutations_id.parquet",
    ...     seqinfo_dir="data/tcga/maf_files/output/vcf_files/ID")

    See Also
    --------
    load_or_generate_compact_db : Load cached or generate if needed
    load_validate_compact_all_maf_files_parallel : Core MAF processing
    """
    logger.info(
        f"Generating compact mutation database for "
        f"{signature_class} from raw MAF files..."
    )

    # Map signature class to variant type for MAF processing
    # COSMIC signature classes (SBS, DBS, ID) need to be mapped to
    # MAF Variant_Type values (SNP, DBP, INS/DEL)
    signature_to_variant_type = {
        "SBS": "SNP",
        "DBS": "DBP",
        "ID": "ID",
        "CN": "CN",
        "SV": "SV",
        "RNA-SBS": "SNP",
    }

    variant_type = signature_to_variant_type.get(
        signature_class, signature_class
    )

    db = load_validate_compact_all_maf_files_parallel(
        maf_dir, variant_type=variant_type, **kwargs
    )

    if location_gene_set is not None:
        db = update_genes_with_gene_set(location_gene_set, db)

    if location_db is not None:
        db.to_parquet(location_db, index=False)
        logger.info(
            f"Saved compact mutation database to {location_db}"
        )

    logger.info("... done generating compact mutation database.")
    print("")

    return db


def load_or_generate_compact_db(
    location_db: str,
    maf_dir: str,
    *,
    signature_class: str = "SBS",
    location_gene_set: str | None = location_hgnc_complete_set,
    force_generation: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load or generate a compact database of mutations.

    Loads a preprocessed mutation database if it exists at the
    specified location. Otherwise, calls
    :func:`generate_compact_db` to process raw MAF files,
    save the result to disk in Parquet format, and return the
    resulting DataFrame.

    The compact database consolidates mutation records from
    multiple MAF (Mutation Annotation Format) files into a single
    DataFrame optimized for downstream analysis. It includes
    standardized columns for gene names, genomic coordinates,
    variant types, and sample identifiers.

    Parameters
    ----------
    location_db : str or Path
        Path to the saved (or to-be-saved) compact database file.
        Should use .parquet extension for best performance.
        Example: "data/tcga_compact_db_sbs.parquet"
    maf_dir : str or Path
        Path to directory containing raw MAF files to process.
        All .maf or .maf.gz files in this directory will be
        loaded and consolidated.
    signature_class : str, default "SBS"
        COSMIC signature class to include in the database.
        Must be one of:
        - "SBS": Single base substitution signatures
        - "DBS": Doublet base substitution signatures
        - "ID": Insertion/deletion signatures
        - "CN": Copy number signatures
        - "SV": Structural variant signatures
        - "RNA-SBS": RNA single base substitution signatures
    location_gene_set : str, Path, or None, default location_hgnc_complete_set
        Path to gene set file for standardizing gene names and
        Ensembl IDs. Gene identifiers will be updated using
        :func:`update_gene_names.update_genes_with_gene_set`.
        Defaults to HGNC complete set for consistent gene
        naming. Set to None to skip gene name updates.
    force_generation : bool, default False
        If True, regenerate the database from MAF files even if
        a saved version exists at location_db. Use this when:
        - MAF files have been updated
        - Processing parameters have changed
        - Cached file may be corrupted
    **kwargs : dict
        Additional keyword arguments passed to
        :func:`load_validate_compact_all_maf_files_parallel`.

        **Required for ID signature class:**
        - seqinfo_dir : str or Path
            Directory containing *_seqinfo.txt files generated by
            SigProfilerMatrixGenerator (with seqInfo=True). These
            files provide COSMIC ID-83 mutation type annotations
            for indel variants.

    Returns
    -------
    pd.DataFrame
        Compact mutation database with standardized columns:
        - gene : str
            Gene symbol (HGNC)
        - ensembl_gene_id : str
            Ensembl gene identifier
        - Tumor_Sample_Barcode : str
            Sample identifier
        - Chromosome : str
            Chromosome name
        - Start_Position : int
            Genomic position
        - variant : str
            Variant identifier
        - type : str
            Mutation type classification
        And additional columns depending on signature_class.

    Notes
    -----
    The function uses Parquet format for storage, which provides:
    - Faster read/write than CSV or pickle
    - Automatic data type preservation
    - Efficient compression
    - Cross-platform compatibility

    Requires pyarrow package for Parquet support.

    Gene names are automatically standardized using HGNC
    nomenclature unless location_gene_set is set to None.

    COSMIC signature classes follow the nomenclature from
    https://cancer.sanger.ac.uk/signatures/

    **Important for ID (indel) signature class:**
    When signature_class="ID", you MUST provide seqinfo_dir in
    kwargs. This directory should contain *_seqinfo.txt files
    generated by running SigProfilerMatrixGenerator with
    seqInfo=True. These files are needed to annotate indels
    with their COSMIC ID-83 mutation types.

    Examples
    --------
    >>> # Load or generate SBS database
    >>> db = load_or_generate_compact_db(
    ...     "data/mutations_sbs.parquet",
    ...     "data/tcga/maf_files",
    ...     signature_class="SBS")

    >>> # Skip gene name updates
    >>> db = load_or_generate_compact_db(
    ...     "data/mutations_sbs.parquet",
    ...     "data/tcga/maf_files",
    ...     signature_class="SBS",
    ...     location_gene_set=None)

    >>> # Load indel database
    >>> # (requires seqinfo_dir for generation, provide custom path if needed)
    >>> db_indels = load_or_generate_compact_db(
    ...     "data/mutations_id.parquet",
    ...     "data/tcga/maf_files",
    ...     signature_class="ID",
    ...     seqinfo_dir="data/tcga/maf_files/output/vcf_files/ID")

    See Also
    --------
    generate_compact_db : Always generate from MAF files
    load_validate_compact_all_maf_files_parallel : Core MAF processing
    """
    if os.path.exists(location_db) and not force_generation:
        logger.info(
            f"Loading compact mutation database for {signature_class}"
            f" from {location_db}"
        )
        db = pd.read_parquet(location_db)
        if location_gene_set is not None:
            db = update_genes_with_gene_set(location_gene_set, db)
        logger.info("... done loading compact mutation database.")
    else:
        # Use generate_compact_db to create the database
        db = generate_compact_db(
            maf_dir,
            signature_class=signature_class,
            location_db=location_db,
            location_gene_set=location_gene_set,
            **kwargs,
        )
    return db
