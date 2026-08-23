"""Per-sample sequencing-quality flags, independent of mutation count.

Unlike :mod:`qc`, which tags and drops individual mutation *rows*,
this module scores whole *samples* against external evidence of
sequencing quality (tumor purity, VAF distribution shape) and returns
the flags for the caller to act on -- it does not drop or modify
anything itself. This mirrors :func:`qc.check_sample_overlap`'s
"return a result, let the caller decide" shape rather than
:func:`qc.apply_qc`'s tag-and-drop one, since a flagged sample here
may be kept and downweighted rather than dropped, which is a
per-call decision, not something this module should assume.

Neither function fetches its own reference data: callers pass in a
purity table and a copy-number segment table (e.g. TCGA's Pan-Cancer
Atlas ABSOLUTE purity/ploidy and segmentation files) rather than this
module downloading anything TCGA-specific itself, keeping the package
data-source-agnostic.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def flag_low_purity_samples(
    purity_table,
    *,
    barcode_column="array",
    purity_column="purity",
    threshold=0.30,
):
    """Flag samples with tumor purity below `threshold`.

    Low purity means the somatic-variant caller had less tumor signal
    to work with relative to normal-cell contamination, which can
    cause real mutations to go undetected -- independent evidence for
    "this sample was probably under-sequenced" that doesn't rely on
    its own mutation count (see the L_low low-burden-correction
    rework this function was built for).

    Parameters
    ----------
    purity_table : pandas.DataFrame
        A purity table with a sample-barcode column and a purity
        column, e.g. TCGA's Pan-Cancer Atlas ABSOLUTE purity/ploidy
        file (``TCGA_mastercalls.abs_tables_JSedit.fixed.txt``),
        loaded by the caller -- this function does no I/O itself.
    barcode_column : str, default "array"
        Column holding the sample barcode to index by. Must be at
        the same barcode granularity as
        :attr:`MutationDataset.mutation_db`'s
        ``Tumor_Sample_Barcode`` (or a prefix of it -- see
        `barcode_length`).
    purity_column : str, default "purity"
        Column holding the purity estimate, 0-1 scale.
    threshold : float, default 0.30
        Samples with purity strictly below this are flagged. Default
        based on Cheng et al. 2023 (colorectal cancer, MuTect2 +
        validation against TCGA via MuSE/SomaticSniper/VarScan2 --
        overlapping methods with MC3, which called the MAF data this
        pipeline consumes): false-negative mutation calls rose
        sharply below 30% purity.

    Returns
    -------
    pandas.Series
        Boolean, indexed by `barcode_column`'s values. ``True`` means
        flagged. Samples with missing/NaN purity are *not* flagged
        (no evidence either way, not evidence of a problem).
    """
    purity = purity_table.set_index(barcode_column)[purity_column]
    if purity.index.has_duplicates:
        dupes = (
            purity.index[purity.index.duplicated()].unique().tolist()
        )
        logger.warning(
            f"{len(dupes)} duplicate barcode(s) in purity_table "
            f"(e.g. {dupes[:5]}); keeping the first occurrence of each."
        )
        purity = purity[~purity.index.duplicated(keep="first")]
    flagged = purity < threshold
    logger.info(
        f"Purity flag: {int(flagged.sum())}/{len(flagged)} samples "
        f"below purity={threshold} "
        f"({int(purity.isna().sum())} samples had no purity estimate, "
        "not flagged by this check)."
    )
    return flagged.rename("low_purity")


def load_copy_number_segments(
    segments_table,
    *,
    sample_col="Sample",
    chrom_col="Chromosome",
    start_col="Start",
    end_col="End",
    cn_col="Modal_Total_CN",
):
    """Load a copy-number segment table into a per-(sample, chromosome)
    lookup structure.

    Parameters
    ----------
    segments_table : pandas.DataFrame
        A copy-number segment table with sample, chromosome, start,
        end, and total-copy-number columns, e.g. TCGA's Pan-Cancer
        Atlas ABSOLUTE-annotated seg file
        (``TCGA_mastercalls.abs_segtabs.fixed.txt``), loaded by the
        caller -- this function does no I/O itself. Chromosome values
        may be numeric (as in that file, X coded as 23) or
        ``"chr"``-prefixed strings; both are normalized to a bare
        number string internally.
    sample_col, chrom_col, start_col, end_col, cn_col : str
        Column names.

    Returns
    -------
    dict[tuple[str, str], tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
        ``(sample_barcode, chromosome) -> (starts, ends, copy_numbers)``,
        sorted by start. Coordinates are used as given (1-based
        inclusive, the usual segment-file convention) -- see
        :func:`annotate_local_copy_number` for the matching lookup.
    """
    segments = {}
    df = segments_table.dropna(subset=[chrom_col, start_col, end_col])
    chrom_norm = df[chrom_col].map(_normalize_chromosome)
    for (sample, chrom), group in df.groupby(
        [df[sample_col], chrom_norm]
    ):
        g = group.sort_values(start_col)
        segments[(sample, chrom)] = (
            g[start_col].to_numpy(),
            g[end_col].to_numpy(),
            g[cn_col].to_numpy(),
        )
    return segments


def _normalize_chromosome(chrom):
    """Normalize a chromosome value to a bare number string ("1".."23"
    for autosomes/X), matching ABSOLUTE's encoding (X=23)."""
    s = str(chrom)
    s = s.removeprefix("chr")
    if s in ("X", "x"):
        return "23"
    if s in ("Y", "y"):
        return "24"
    s = s.removesuffix(".0")
    return s


def annotate_local_copy_number(
    mutation_db,
    copy_number_segments,
    *,
    sample_col="Tumor_Sample_Barcode",
    chrom_col="Chromosome",
    position_col="Start_Position",
    barcode_length=15,
    out_col="local_cn",
):
    """Add a local total-copy-number column to `mutation_db`.

    Looks up each mutation's overlapping segment by
    (sample barcode truncated to `barcode_length`, normalized
    chromosome, position falling in [start, end]).

    Parameters
    ----------
    mutation_db : pandas.DataFrame
        A compact mutation database with `sample_col`, `chrom_col`,
        `position_col` columns.
    copy_number_segments : dict
        As returned by :func:`load_copy_number_segments`.
    barcode_length : int, default 15
        Length to truncate `sample_col`'s barcode to before matching
        against `copy_number_segments`'s keys (TCGA sample-level
        barcodes, e.g. ``TCGA-OR-A5J1-01``, are shorter than the full
        aliquot barcode MAFs use).
    out_col : str, default "local_cn"
        Name of the added column.

    Returns
    -------
    pandas.DataFrame
        `mutation_db` with `out_col` added (``NaN`` where no segment
        was found -- no coverage for that sample/chromosome, e.g. Y,
        which ABSOLUTE does not call).
    """
    result = mutation_db.copy()
    result[out_col] = np.nan
    short_barcode = result[sample_col].str.slice(0, barcode_length)
    chrom_norm = result[chrom_col].map(_normalize_chromosome)

    for (sample, chrom), group in result.groupby(
        [short_barcode, chrom_norm]
    ):
        key = (sample, chrom)
        if key not in copy_number_segments:
            continue
        starts, ends, cn = copy_number_segments[key]
        if len(starts) == 0:
            continue
        positions = group[position_col].to_numpy()
        idx = np.searchsorted(starts, positions, side="right") - 1
        in_segment = np.zeros(len(positions), dtype=bool)
        valid = idx >= 0
        in_segment[valid] = positions[valid] <= ends[idx[valid]]
        hit_index = group.index[in_segment]
        result.loc[hit_index, out_col] = cn[idx[in_segment]]

    return result


def compute_vaf_shape_score(
    sample_mutation_db_rows,
    purity,
    *,
    depth_col="t_depth",
    alt_col="t_alt_count",
    cn_col="local_cn",
    min_depth=20,
    min_variants=5,
):
    """Score how well one sample's VAF pattern matches its purity.

    Estimates each variant's cancer cell fraction (CCF) -- what
    fraction of tumor cells carry it -- from its observed VAF, the
    sample's purity, and the mutation's local total copy number:

    .. math::
        \\hat{c}_{j,m} = \\text{VAF}_{j,m} \\cdot
            \\frac{2(1-\\pi_j) + \\pi_j\\, q_{j,m}}{\\pi_j}

    (the same relationship subclonal-reconstruction tools like
    PyClone use, here just a point estimate, not a full posterior --
    full derivation in the "purity/CN-corrected VAF QC" write-up).
    A truly clonal mutation should give :math:`\\hat{c}\\approx1`
    regardless of its local copy number, once corrected for; genuine
    under-calling depresses VAF -- and so :math:`\\hat{c}` -- across
    the board, including mutations that should be clonal. The score
    is the 75th percentile of :math:`\\hat{c}_{j,m}` across the
    sample's variants: using an upper quantile rather than every
    variant is what makes this robust to real subclonal mutations
    (which sit below 1 legitimately) without needing to model them.

    Assumes multiplicity 1 (mutation on exactly one copy) -- if the
    true multiplicity is greater, :math:`\\hat{c}` is biased *upward*
    by that factor, which cannot cause a false flag (this test flags
    low values) but can mask genuine under-calling in heavily
    copy-number-amplified samples (false-negative risk). Variants at
    homozygously-deleted loci (local copy number 0) are excluded --
    a real mutation shouldn't be observable there at all, so a call
    there is more likely an artifact than informative signal.

    Parameters
    ----------
    sample_mutation_db_rows : pandas.DataFrame
        One sample's rows from a compact mutation database, with
        `depth_col`/`alt_col`/`cn_col` columns (`cn_col` from
        :func:`annotate_local_copy_number`).
    purity : float
        This sample's tumor purity estimate (0-1 scale).
    depth_col, alt_col, cn_col : str
        Column names for read depth, alt-allele read count, and
        local total copy number.
    min_depth : int, default 20
        Variants below this depth are excluded -- a *per-variant*
        filter, not a minimum sample size (see `min_variants`).
    min_variants : int, default 5
        Minimum depth-filtered, copy-number-annotated variants
        required to score a sample at all.

    Returns
    -------
    float
        :math:`\\hat{c}^{(75)}_j`, the 75th percentile of estimated
        CCFs; low means an anomalous VAF shape (flag-worthy). ``NaN``
        if `purity` is missing/NaN or fewer than `min_variants`
        variants have both sufficient depth and a known,
        non-homozygously-deleted local copy number (not enough
        evidence either way).
    """
    if purity is None or (
        isinstance(purity, float) and np.isnan(purity)
    ):
        return np.nan

    rows = sample_mutation_db_rows[
        (sample_mutation_db_rows[depth_col] >= min_depth)
        & sample_mutation_db_rows[cn_col].notna()
        & (sample_mutation_db_rows[cn_col] > 0)
    ]
    if len(rows) < min_variants:
        return np.nan

    vaf = rows[alt_col] / rows[depth_col]
    ccf_hat = (
        vaf * (2 * (1 - purity) + purity * rows[cn_col]) / purity
    )
    return float(np.percentile(ccf_hat, 75))


def flag_vaf_shape_samples(
    mutation_db,
    purity_table,
    copy_number_segments,
    *,
    barcode_column="array",
    purity_column="purity",
    threshold=0.7,
    depth_col="t_depth",
    alt_col="t_alt_count",
    min_depth=20,
    min_variants=5,
):
    """Flag samples whose VAF distribution doesn't match their purity.

    Batch driver: joins local copy number (see
    :func:`annotate_local_copy_number`), then applies
    :func:`compute_vaf_shape_score` per sample. Requires `depth_col`/
    `alt_col` to have survived compaction (see
    :func:`load_maf_files.compact_data`).

    Parameters
    ----------
    mutation_db : pandas.DataFrame
        A dataset's compact mutation database, with
        ``Tumor_Sample_Barcode``, ``Chromosome``, ``Start_Position``,
        `depth_col`, `alt_col` columns.
    purity_table : pandas.DataFrame
        Same purity table as :func:`flag_low_purity_samples`.
    copy_number_segments : dict
        As returned by :func:`load_copy_number_segments`.
    threshold : float, default 0.7
        Samples with an estimated 75th-percentile CCF below this are
        flagged.
    Other parameters : see :func:`compute_vaf_shape_score`.

    Returns
    -------
    pandas.Series
        Boolean, indexed by ``Tumor_Sample_Barcode``. ``True`` means
        flagged. Samples with no purity estimate or too few
        depth-filtered, copy-number-annotated variants are not
        flagged (no evidence either way).
    """
    purity = purity_table.set_index(barcode_column)[purity_column]
    if purity.index.has_duplicates:
        purity = purity[~purity.index.duplicated(keep="first")]

    barcode_length = len(purity.index[0])
    annotated = annotate_local_copy_number(
        mutation_db,
        copy_number_segments,
        barcode_length=barcode_length,
    )

    scores = {}
    for sample_barcode, group in annotated.groupby(
        "Tumor_Sample_Barcode"
    ):
        short = sample_barcode[:barcode_length]
        scores[sample_barcode] = compute_vaf_shape_score(
            group,
            purity.get(short, np.nan),
            depth_col=depth_col,
            alt_col=alt_col,
            min_depth=min_depth,
            min_variants=min_variants,
        )
    scores = pd.Series(scores, name="ccf_p75")
    flagged = scores < threshold
    n_scored = scores.notna().sum()
    logger.info(
        f"VAF-shape flag: {int(flagged.sum())}/{n_scored} scored samples "
        f"below ccf_p75={threshold} "
        f"({len(scores) - n_scored} samples could not be scored -- no "
        "purity estimate or too few usable variants)."
    )
    return flagged.rename("anomalous_vaf_shape")


def combine_sample_flags(*flags, how="any"):
    """Combine per-sample boolean flag Series into one.

    Parameters
    ----------
    *flags : pandas.Series
        Boolean Series to combine, e.g. the outputs of
        :func:`flag_low_purity_samples` and
        :func:`flag_vaf_shape_samples`. Indices need not match --
        combined on their union, treating a sample absent from one
        flag as not-flagged by that check (no evidence, not a
        positive flag).
    how : {"any", "all"}, default "any"
        "any": flagged if any input flags it. "all": flagged only if
        every input flags it.

    Returns
    -------
    pandas.Series
        Boolean, indexed by the union of all input indices.
    """
    if how not in ("any", "all"):
        raise ValueError(f"how must be 'any' or 'all', got {how!r}")
    combined = pd.concat(flags, axis=1)
    result = (
        combined.any(axis=1) if how == "any" else combined.all(axis=1)
    )
    return result.rename("sample_qc_flag")
