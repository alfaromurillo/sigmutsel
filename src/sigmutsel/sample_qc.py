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
purity table (e.g. TCGA's Pan-Cancer Atlas ABSOLUTE purity/ploidy
calls) rather than this module downloading anything TCGA-specific
itself, keeping the package data-source-agnostic.
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


def compute_vaf_shape_score(
    sample_mutation_db_rows,
    purity,
    *,
    depth_col="t_depth",
    alt_col="t_alt_count",
    min_depth=20,
    min_variants=5,
):
    """Score how well one sample's VAF pattern matches its purity.

    Under a correctly-sequenced, clonal, diploid null, each variant's
    alt-read count should follow ``Binomial(depth, purity / 2)``. This
    computes a two-sided binomial-test p-value per variant (a
    deliberately simple flat-diploid approximation -- it ignores
    subclonality and local copy-number, which is a reasonable
    simplification for a QC gate rather than a precision clonal-
    architecture tool; see `TODO.md`'s "Low priority" section for the
    PyClone-VI-based alternative this could grow into), then runs a
    one-sample Kolmogorov-Smirnov test of those p-values against
    Uniform(0, 1) -- under a correctly-specified null the per-variant
    p-values should be uniform; systematic under-calling (real
    mutations missed near the detection threshold) skews them toward
    0.

    Parameters
    ----------
    sample_mutation_db_rows : pandas.DataFrame
        One sample's rows from a compact mutation database (e.g.
        ``MutationDataset.mutation_db`` filtered to one
        ``Tumor_Sample_Barcode``), with `depth_col`/`alt_col` columns.
    purity : float
        This sample's tumor purity estimate (0-1 scale).
    depth_col, alt_col : str
        Column names for read depth and alt-allele read count.
    min_depth : int, default 20
        Variants below this depth are excluded -- low-depth variants
        add sampling noise that isn't the failure mode this test is
        designed to catch.
    min_variants : int, default 5
        Minimum depth-filtered variants required to run the test.

    Returns
    -------
    float
        The KS test's p-value; low means an anomalous VAF shape
        (flag-worthy). ``NaN`` if `purity` is missing/NaN or fewer
        than `min_variants` variants pass the depth filter (not
        enough evidence either way).
    """
    from scipy.stats import binomtest, kstest

    if purity is None or (
        isinstance(purity, float) and np.isnan(purity)
    ):
        return np.nan

    rows = sample_mutation_db_rows[
        sample_mutation_db_rows[depth_col] >= min_depth
    ]
    if len(rows) < min_variants:
        return np.nan

    p_expected = purity / 2
    pvals = [
        binomtest(
            int(alt),
            int(depth),
            p=p_expected,
            alternative="two-sided",
        ).pvalue
        for alt, depth in zip(rows[alt_col], rows[depth_col])
    ]
    return kstest(pvals, "uniform").pvalue


def flag_vaf_shape_samples(
    mutation_db,
    purity_table,
    *,
    barcode_column="array",
    purity_column="purity",
    threshold=0.01,
    depth_col="t_depth",
    alt_col="t_alt_count",
    min_depth=20,
    min_variants=5,
):
    """Flag samples whose VAF distribution doesn't match their purity.

    Batch driver for :func:`compute_vaf_shape_score` over every
    sample in `mutation_db`. Requires `depth_col`/`alt_col` to have
    survived compaction (see :func:`load_maf_files.compact_data`).

    Parameters
    ----------
    mutation_db : pandas.DataFrame
        A dataset's compact mutation database, with
        ``Tumor_Sample_Barcode``, `depth_col`, `alt_col` columns.
    purity_table : pandas.DataFrame
        Same purity table as :func:`flag_low_purity_samples`.
    threshold : float, default 0.01
        Samples with a VAF-shape score below this are flagged.
    Other parameters : see :func:`compute_vaf_shape_score`.

    Returns
    -------
    pandas.Series
        Boolean, indexed by ``Tumor_Sample_Barcode`` (truncated to
        `barcode_column`'s length -- TCGA aliquot barcodes are longer
        than the purity table's sample-level barcodes). ``True`` means
        flagged. Samples with no purity estimate or too few
        depth-filtered variants are not flagged (no evidence either
        way).
    """
    purity = purity_table.set_index(barcode_column)[purity_column]
    if purity.index.has_duplicates:
        purity = purity[~purity.index.duplicated(keep="first")]

    barcode_length = len(purity.index[0])

    scores = {}
    for sample_barcode, group in mutation_db.groupby(
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
    scores = pd.Series(scores, name="vaf_shape_pvalue")
    flagged = scores < threshold
    n_scored = scores.notna().sum()
    logger.info(
        f"VAF-shape flag: {int(flagged.sum())}/{n_scored} scored samples "
        f"below p={threshold} "
        f"({len(scores) - n_scored} samples could not be scored -- no "
        "purity estimate or too few depth-filtered variants)."
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
