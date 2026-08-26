"""Preprocessing QC for MAF-loaded mutation data.

Structured problem tagging, germline-frequency filtering,
adjacent-SNV (MNV/DBS) merging, and a cross-sample overlap/
contamination detector -- conventions adapted from cancereffectsizeR
(Townsend-Lab-Yale/cancereffectsizeR, an R package;
https://github.com/Townsend-Lab-Yale/cancereffectsizeR), a widely
used tool for the same mutation-rate/selection-intensity estimation
problem this package addresses.

Every function here tags rows with a ``"problem"`` column rather
than dropping them outright, matching cancereffectsizeR's
``preload_maf()`` convention (``problem`` is ``None``/``NaN`` for a
clean row, else a reason string). ``apply_qc`` runs the full
sequence and is the usual entry point; the individual functions are
exposed for callers who want just one check or a custom order.
Problem strings deliberately mirror cancereffectsizeR's own names
where the check is equivalent (``duplicate_record``,
``germline_variant_site``, ``merged_into_dbs_variant``,
``merged_with_nearby_variant``), so a reader familiar with either
package recognizes the vocabulary.
"""

import logging
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from .load_maf_files import (
    validate_alleles_snv,
    validate_chromosome,
    validate_context_snv,
    validate_reference_matches_context_snv,
)
from .wes_target import _merge_intervals

logger = logging.getLogger(__name__)

DEFAULT_GERMLINE_AF_COLUMN = "gnomAD_non_cancer_MAX_AF_adj"
DEFAULT_GERMLINE_AF_THRESHOLD = 0.001

# UCSC's rmsk table dump has no header row; these are its 17
# columns in order (bin, swScore, milliDiv, milliDel, milliIns,
# genoName, genoStart, genoEnd, genoLeft, strand, repName, repClass,
# repFamily, repStart, repEnd, repLeft, id) -- verified directly
# against a live download, 2026-08-22, not from memory. Only the
# three we use are named; the rest go in _unused_N placeholders so
# `usecols` can still select by position.
_RMSK_COLUMN_NAMES = [
    "_unused_0",
    "_unused_1",
    "_unused_2",
    "_unused_3",
    "_unused_4",
    "chrom",
    "start",
    "end",
    "_unused_8",
    "_unused_9",
    "_unused_10",
    "_unused_11",
    "_unused_12",
    "_unused_13",
    "_unused_14",
    "_unused_15",
    "_unused_16",
]


def _ensure_problem_column(df):
    if "problem" not in df.columns:
        df = df.copy()
        df["problem"] = None
    return df


def validate_full_with_problems(
    df, *, variant_type="SNP", context_length=11
):
    """Like :func:`load_maf_files.validate_full`, but tags, not drops.

    Runs the same sequence of checks (chromosome validity, and for
    SNPs: allele validity, CONTEXT validity, reference-matches-
    CONTEXT), but instead of silently filtering, every row is kept
    and a ``"problem"`` column records why a row would have been
    dropped (``None`` for rows that pass all checks). Only rows
    still untagged (``problem`` is ``None``) are checked at each
    step, so a row keeps its *first* failure reason.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame, as read from a raw MAF file.
    variant_type : str, default "SNP"
        Only "SNP" runs the allele/context/reference checks; any
        other value runs just the chromosome check, matching
        :func:`load_maf_files.validate_full`.
    context_length : int, default 11
        Forwarded to the CONTEXT-dependent checks.

    Returns
    -------
    pandas.DataFrame
        Same rows and columns as *df*, plus ``"problem"``.
    """
    df = _ensure_problem_column(df)
    remaining = df[df["problem"].isna()]

    checks = [("invalid_chromosome", validate_chromosome)]
    if variant_type == "SNP":
        checks += [
            ("invalid_allele", validate_alleles_snv),
            (
                "invalid_context",
                lambda d: validate_context_snv(
                    d, context_length=context_length
                ),
            ),
            (
                "reference_mismatch",
                lambda d: validate_reference_matches_context_snv(
                    d, context_length=context_length
                ),
            ),
        ]

    for name, fn in checks:
        kept = fn(remaining)
        dropped_idx = remaining.index.difference(kept.index)
        df.loc[dropped_idx, "problem"] = name
        remaining = kept

    return df


def flag_exact_duplicates(df):
    """Tag exact-duplicate mutation records.

    A duplicate is a row sharing ``Tumor_Sample_Barcode``,
    ``Chromosome``, ``Start_Position``, ``Reference_Allele``, and
    ``Tumor_Seq_Allele2`` with an earlier row -- the same call
    appearing twice in one sample, e.g. from an upstream merge of
    overlapping MAF sources. The first occurrence is left untagged;
    later ones get ``problem = "duplicate_record"``. Rows already
    tagged by another check are left alone.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the five columns above.

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated.
    """
    df = _ensure_problem_column(df)
    dup_cols = [
        "Tumor_Sample_Barcode",
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]
    is_dup = df.duplicated(subset=dup_cols, keep="first")
    untagged = df["problem"].isna()
    df.loc[is_dup & untagged, "problem"] = "duplicate_record"
    return df


def flag_germline_variants(
    df,
    *,
    af_column=DEFAULT_GERMLINE_AF_COLUMN,
    threshold=DEFAULT_GERMLINE_AF_THRESHOLD,
):
    """Tag likely-germline sites by gnomAD population frequency.

    A row with a population allele frequency above *threshold* in
    gnomAD's non-cancer cohort is far more likely a germline variant
    than a real somatic mutation, and gets
    ``problem = "germline_variant_site"``. Rows with a missing/NaN
    frequency (common -- most somatic calls have no gnomAD match at
    all) are left untouched: absence of a population-frequency
    record is not evidence of germline origin.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw MAF-derived DataFrame. If *af_column* is absent, this is
        a no-op (with a warning) rather than an error, since not
        every MAF source populates gnomAD annotation columns.
    af_column : str, default "gnomAD_non_cancer_MAX_AF_adj"
        The maximum adjusted allele frequency across gnomAD's
        non-cancer subpopulations -- avoids the circularity of
        gnomAD's own cancer-cohort subjects, and guards against a
        variant that's rare globally but common in one population.
    threshold : float, default 0.001
        Standard somatic-calling germline cutoff (0.1%); the same
        order of magnitude used by COSMIC's own germline-site tier
        and by cancereffectsizeR's ``cosmic_site_tier`` filter.

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated.
    """
    if af_column not in df.columns:
        logger.warning(
            f"{af_column!r} not present in this MAF; "
            "skipping germline-frequency filter."
        )
        return _ensure_problem_column(df)

    df = _ensure_problem_column(df)
    af = pd.to_numeric(df[af_column], errors="coerce")
    is_germline = af > threshold
    untagged = df["problem"].isna()
    df.loc[is_germline & untagged, "problem"] = (
        "germline_variant_site"
    )
    return df


def detect_mnv_dbs(df):
    """Tag adjacent same-sample SNVs likely from one real event.

    Groups already-untagged rows by (``Tumor_Sample_Barcode``,
    ``Chromosome``), sorts by ``Start_Position``, and clusters
    consecutive positions no more than 2bp apart -- these are very
    likely one real multi-nucleotide mutation event that an
    upstream variant caller split into separate single-base calls
    (a classic case: UV-signature CC>TT dinucleotide changes),
    rather than two independent single-base substitutions. Left
    uncorrected, every row in a cluster would be counted as an
    independent SBS mutation, inflating burden and distorting
    signature decomposition.

    A cluster of exactly 2 positions exactly 1bp apart gets
    ``problem = "merged_into_dbs_variant"`` (a genuine doublet-base
    substitution); any other cluster (3+ positions, or a 2bp gap)
    gets ``problem = "merged_with_nearby_variant"``. Every row in a
    cluster is tagged, including the first -- sigmutsel has no DBS/
    MNV modeling downstream today (SBS-only), so the correct
    treatment is to exclude the whole event from SBS counting, not
    to keep one representative row as if it were a normal SBS.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``Tumor_Sample_Barcode``, ``Chromosome``, and
        ``Start_Position``. Rows already tagged with a ``"problem"``
        are excluded from clustering (both as cluster members and as
        neighbors), so an already-dropped row can't chain two
        otherwise-unrelated SNVs together.

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated.
    """
    df = _ensure_problem_column(df)
    untagged = df[df["problem"].isna()]
    if untagged.empty:
        return df

    ordered = untagged.sort_values(
        ["Tumor_Sample_Barcode", "Chromosome", "Start_Position"]
    )
    same_group = (
        ordered["Tumor_Sample_Barcode"]
        == ordered["Tumor_Sample_Barcode"].shift()
    ) & (ordered["Chromosome"] == ordered["Chromosome"].shift())
    gap = (
        ordered["Start_Position"] - ordered["Start_Position"].shift()
    )
    starts_new_cluster = ~(same_group & (gap <= 2))
    cluster_id = starts_new_cluster.cumsum()

    for _, group in ordered.groupby(cluster_id):
        if len(group) < 2:
            continue
        positions = group["Start_Position"].tolist()
        if len(group) == 2 and positions[1] - positions[0] == 1:
            problem = "merged_into_dbs_variant"
        else:
            problem = "merged_with_nearby_variant"
        df.loc[group.index, "problem"] = problem

    return df


def load_repeat_intervals(rmsk_path):
    """Load a downloaded rmsk table into merged per-chromosome intervals.

    Parameters
    ----------
    rmsk_path : str or pathlib.Path
        Path to an ``rmsk.<build>.txt.gz`` file, as downloaded by
        :func:`setup.download_repeatmasker_bed`.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Chromosome -> (merged_starts, merged_ends), 0-based
        half-open (UCSC convention), sorted and overlap-merged --
        the format :func:`flag_repetitive_regions` expects. Reuses
        :func:`wes_target._merge_intervals`, the same interval-merge
        already used for the WES-target gene universe.
    """
    rmsk = pd.read_csv(
        rmsk_path,
        sep="\t",
        header=None,
        names=_RMSK_COLUMN_NAMES,
        usecols=["chrom", "start", "end"],
    )
    intervals = {}
    for chrom, group in rmsk.groupby("chrom"):
        arr = group[["start", "end"]].sort_values("start").to_numpy()
        starts, ends = _merge_intervals(arr[:, 0], arr[:, 1])
        intervals[chrom] = (starts, ends)
    return intervals


def flag_repetitive_regions(df, repeat_intervals):
    """Tag rows overlapping a RepeatMasker-annotated region.

    Sequencing/calling artifacts cluster in low-complexity and
    repetitive regions; a variant call there is more likely a
    mapping artifact than a real somatic mutation. Tags matching
    rows with ``problem = "repetitive_region"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``Chromosome`` and ``Start_Position`` (1-based,
        MAF convention).
    repeat_intervals : dict
        As returned by :func:`load_repeat_intervals`.

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated.
    """
    df = _ensure_problem_column(df)
    untagged = df[df["problem"].isna()]
    if untagged.empty:
        return df

    for chrom, group in untagged.groupby("Chromosome"):
        if chrom not in repeat_intervals:
            continue
        starts, ends = repeat_intervals[chrom]
        if len(starts) == 0:
            continue
        # MAF's Start_Position is 1-based; UCSC intervals are
        # 0-based half-open. Convert to 0-based first so membership
        # is the standard, unambiguous start <= p0 < end check (the
        # same pattern wes_target._overlapping_gene_ids uses).
        positions_0based = group["Start_Position"].to_numpy() - 1
        idx = (
            np.searchsorted(starts, positions_0based, side="right")
            - 1
        )
        in_repeat = np.zeros(len(positions_0based), dtype=bool)
        valid = idx >= 0
        in_repeat[valid] = positions_0based[valid] < ends[idx[valid]]
        hit_index = group.index[in_repeat]
        df.loc[hit_index, "problem"] = "repetitive_region"

    return df


def flag_artifact_signature_mutations(
    df, artifact_probability, threshold=0.5
):
    """Tag mutations dominantly attributed to a technical-artifact signature.

    Unlike the other checks in this module, this one needs external,
    dataset-wide input that doesn't exist per-MAF-file or before a
    signature-decomposition fit has run: ``artifact_probability``,
    each mutation's total probability mass on signatures COSMIC
    itself flags as likely sequencing artifacts (see
    :func:`sigmutsel.signature_attribution.compute_signature_probability_mass`
    and ``constants.ARTIFACT_SIGNATURES``). Because of that, this
    check is *not* one of :func:`apply_qc`'s per-file checks (those
    run before any signature decomposition, one MAF file at a time,
    inside :func:`load_maf_files.process_single_maf`) -- it's meant to
    be called once, explicitly, on the whole cohort's already-built
    mutation database, after the first ("pass A") fit of the two-pass
    artifact-detection procedure (fit with artifact signatures present
    in the basis so they can absorb their own mutations, tag and drop
    the mutations dominantly attributed to them, then refit ("pass B")
    on the cleaned mutation set with artifact signatures excluded --
    see this project's plan for the full design and
    :func:`sigmutsel.signature_decomposition.resolve_exclusion_list`'s
    ``exclude_artifacts`` parameter for pass B's exclusion side).

    Parameters
    ----------
    df : pandas.DataFrame
        Mutation dataframe (e.g. a dataset's whole-cohort
        ``mutation_db``, not a single MAF file's rows).
    artifact_probability : pandas.Series or array-like
        Per-mutation summed probability mass on artifact signatures.
        If a Series, aligned to *df* by index (missing entries treated
        as 0, i.e. not flagged); otherwise must be the same length and
        row order as *df*.
    threshold : float, default 0.5
        A mutation is flagged if its artifact-probability mass exceeds
        this value. Not derived from theory -- tune against the real
        distribution observed on a few pilot cohorts (see this
        project's plan's Phase 0/2 empirical validation).

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated
        (``"artifact_signature_mutation"`` for flagged rows).
    """
    df = _ensure_problem_column(df)
    untagged_mask = df["problem"].isna()
    if not untagged_mask.any():
        return df

    if isinstance(artifact_probability, pd.Series):
        probs = artifact_probability.reindex(df.index).fillna(0.0)
    else:
        probs = pd.Series(artifact_probability, index=df.index)

    flag_mask = untagged_mask & (probs > threshold)
    df.loc[flag_mask, "problem"] = "artifact_signature_mutation"
    return df


def flag_treatment_signature_samples(
    df, treatment_load, threshold=0.2
):
    """Tag every mutation of a sample with implausible treatment load.

    TCGA's "treatment-naive" clinical annotation is retrospective
    self-report and can miss real cases (e.g. neoadjuvant chemotherapy
    before resection, common for several cancer types, is not always
    captured). This check flags whole *samples* rather than individual
    mutations, unlike :func:`flag_artifact_signature_mutations` --
    real treatment mutagenesis, once present, reshapes a tumor's whole
    subsequent evolutionary history rather than a confinable subset of
    mutations the way a sequencing artifact does, so dropping the
    sample is the right granularity (same as the purity/VAF-shape
    sample-level QC in ``tcga_analysis``, not the artifact mechanism).

    Like :func:`flag_artifact_signature_mutations`, this needs
    dataset-wide input from a signature-decomposition fit and is meant
    to be called once, explicitly, on the whole cohort's mutation
    database -- not one of :func:`apply_qc`'s per-file checks.

    Parameters
    ----------
    df : pandas.DataFrame
        Mutation dataframe with a ``"Tumor_Sample_Barcode"`` column
        (e.g. a dataset's whole-cohort ``mutation_db``).
    treatment_load : pandas.Series
        Per-sample fraction of fitted mutation burden attributed to
        ``constants.TREATMENT_ASSOCIATED_SIGNATURES``, indexed by
        ``Tumor_Sample_Barcode``. Compute this from an
        unrestricted fit only -- a fit that already excludes
        treatment signatures can never show this signal. Samples the
        caller considers too low-burden for this fraction to be
        trustworthy should be ``NaN`` here, not left out or set to 0:
        ``NaN > threshold`` is ``False``, so they're never flagged,
        which is the desired behavior (see the mutation_rates
        project's 2026-08-25 groundwork -- a naive threshold with no
        burden gate is dominated by low-count NNLS noise, not real
        contamination).
    threshold : float, default 0.2
        A sample is flagged if its treatment-signature load exceeds
        this fraction of its fitted burden. Not derived from theory --
        tune against the real per-cohort distribution (see this
        project's plan).

    Returns
    -------
    pandas.DataFrame
        *df* with ``"problem"`` added/updated
        (``"treatment_signature_sample"`` for every row of a flagged
        sample).
    """
    df = _ensure_problem_column(df)
    untagged_mask = df["problem"].isna()
    if not untagged_mask.any():
        return df

    flagged_samples = treatment_load[treatment_load > threshold].index
    sample_mask = df["Tumor_Sample_Barcode"].isin(flagged_samples)

    flag_mask = untagged_mask & sample_mask
    df.loc[flag_mask, "problem"] = "treatment_signature_sample"
    return df


def apply_qc(
    df,
    *,
    variant_type="SNP",
    context_length=11,
    germline_af_column=DEFAULT_GERMLINE_AF_COLUMN,
    germline_af_threshold=DEFAULT_GERMLINE_AF_THRESHOLD,
    check_duplicates=True,
    check_germline=True,
    check_mnv_dbs=True,
    repeat_intervals=None,
):
    """Run the full QC sequence and return a problem-tagged DataFrame.

    Order: :func:`validate_full_with_problems`, then (if enabled)
    :func:`flag_exact_duplicates`, :func:`flag_germline_variants`,
    :func:`detect_mnv_dbs`, :func:`flag_repetitive_regions`. Each
    step only tags rows still untagged by an earlier step, so a
    row's ``problem`` reflects the first issue found, in this order.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw MAF-derived DataFrame.
    variant_type : str, default "SNP"
        Forwarded to :func:`validate_full_with_problems`.
    context_length : int, default 11
        Forwarded to :func:`validate_full_with_problems`.
    germline_af_column, germline_af_threshold
        Forwarded to :func:`flag_germline_variants`.
    check_duplicates, check_germline, check_mnv_dbs : bool
        Individually disable a step (all default ``True``).
    repeat_intervals : dict or None, default None
        As returned by :func:`load_repeat_intervals`. The
        repetitive-region check only runs if this is given -- unlike
        the other checks it needs an external download
        (:func:`setup.download_repeatmasker_bed`), so there's no
        sensible "default on" the way there is for the others.

    Returns
    -------
    pandas.DataFrame
        *df* plus ``"problem"``, same row count as input.
    """
    df = validate_full_with_problems(
        df, variant_type=variant_type, context_length=context_length
    )
    if check_duplicates:
        df = flag_exact_duplicates(df)
    if check_germline and variant_type == "SNP":
        df = flag_germline_variants(
            df,
            af_column=germline_af_column,
            threshold=germline_af_threshold,
        )
    if check_mnv_dbs and variant_type == "SNP":
        df = detect_mnv_dbs(df)
    if repeat_intervals is not None:
        df = flag_repetitive_regions(df, repeat_intervals)
    return df


def summarize_problems(df):
    """Return a ``{problem: count}`` summary, for logging.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have a ``"problem"`` column (e.g. from :func:`apply_qc`).

    Returns
    -------
    collections.Counter
        Counts per non-null problem value, most common first. Empty
        if every row is clean.
    """
    return Counter(df["problem"].dropna().value_counts().to_dict())


def _overlap_flag(n1, n2, n_shared):
    """Port of cancereffectsizeR's check_sample_overlap() thresholds.

    Mirrors ``check_sample_overlap`` in cancereffectsizeR's
    ``R/preload_maf.R`` exactly (both total-mutation-count and
    shared-mutation-count thresholds, read directly from that file,
    not paraphrased from its documentation).
    """
    if n1 < 21 and n2 < 21 and n_shared > 1:
        return True
    if n1 < 6 and n2 < 6 and n_shared > 0:
        return True
    if (n1 < 3 or n2 < 3) and n_shared > 0:
        return True
    greater_overlap = max(
        n_shared / n1 if n1 else 0.0,
        n_shared / n2 if n2 else 0.0,
    )
    return bool(n_shared > 2 and greater_overlap > 0.01)


def check_sample_overlap(mutation_db):
    """Flag sample pairs with suspiciously high mutation overlap.

    Catches *unknown* duplication -- a sample processed twice under
    different barcodes, or real cross-sample contamination -- which
    TCGA-barcode-based dedup (:mod:`tcga_sample_selection`) can't
    see, since that only catches *known* same-patient duplicates by
    barcode. Ported from cancereffectsizeR's
    ``check_sample_overlap()``.

    For efficiency, only mutations recurrent across >=2 samples are
    considered (most mutations are private to one sample), and only
    samples sharing at least one recurrent mutation are compared
    pairwise -- this keeps the cost far below a full O(n^2) over the
    whole cohort for any real TCGA-sized dataset.

    Parameters
    ----------
    mutation_db : pandas.DataFrame
        A dataset's compact mutation database (e.g.
        ``MutationDataset.mutation_db``), with at least
        ``Tumor_Sample_Barcode``, ``Chromosome``, ``Start_Position``,
        and ``type`` columns (``type`` plus position is enough to
        identify "the same mutation" -- it already encodes the
        normalized ref>alt substitution, so raw
        ``Reference_Allele``/``Tumor_Seq_Allele2`` columns, dropped
        by :func:`load_maf_files.compact_data`, aren't needed here).

    Returns
    -------
    pandas.DataFrame
        One row per flagged sample pair: ``sample_1``, ``sample_2``,
        ``n_shared`` (shared recurrent mutations), ``n_windows``
        (how many distinct 1000bp windows those fall in -- a few
        shared hotspots in different windows is more likely
        coincidence than a contiguous block of shared calls, which
        looks like contamination), ``n_total_1``, ``n_total_2``
        (each sample's total mutation count). Empty (with these
        columns) if nothing was flagged.
    """
    columns = [
        "sample_1",
        "sample_2",
        "n_shared",
        "n_windows",
        "n_total_1",
        "n_total_2",
    ]
    required = {
        "Tumor_Sample_Barcode",
        "Chromosome",
        "Start_Position",
        "type",
    }
    missing = required - set(mutation_db.columns)
    if missing:
        raise KeyError(
            f"mutation_db is missing required column(s): {missing}"
        )

    df = mutation_db[
        [
            "Tumor_Sample_Barcode",
            "Chromosome",
            "Start_Position",
            "type",
        ]
    ].copy()
    df["mut_id"] = (
        df["Chromosome"].astype(str)
        + "_"
        + df["Start_Position"].astype(str)
        + "_"
        + df["type"].astype(str)
    )
    df["window"] = (
        df["Chromosome"].astype(str)
        + "."
        + (df["Start_Position"] % 1000).astype(str)
    )

    sample_totals = mutation_db.groupby("Tumor_Sample_Barcode").size()

    mut_sample_counts = df.groupby("mut_id")[
        "Tumor_Sample_Barcode"
    ].nunique()
    recurrent_ids = mut_sample_counts[mut_sample_counts > 1].index
    if len(recurrent_ids) == 0:
        logger.info(
            "No recurrent mutations in this dataset; no sample "
            "overlap is possible."
        )
        return pd.DataFrame(columns=columns)

    recurrent = df[df["mut_id"].isin(recurrent_ids)]
    samples_to_check = sorted(
        recurrent["Tumor_Sample_Barcode"].unique()
    )
    if len(samples_to_check) < 2:
        return pd.DataFrame(columns=columns)

    sample_muts = {
        s: set(
            recurrent.loc[
                recurrent["Tumor_Sample_Barcode"] == s, "mut_id"
            ]
        )
        for s in samples_to_check
    }
    mut_windows = recurrent.drop_duplicates("mut_id").set_index(
        "mut_id"
    )["window"]

    n_pairs = len(samples_to_check) * (len(samples_to_check) - 1) // 2
    logger.info(
        f"Comparing {n_pairs} pairs of samples sharing a recurrent "
        "mutation for suspicious overlap..."
    )

    flagged = []
    for s1, s2 in combinations(samples_to_check, 2):
        shared = sample_muts[s1] & sample_muts[s2]
        n_shared = len(shared)
        if n_shared == 0:
            continue
        n1 = int(sample_totals.get(s1, 0))
        n2 = int(sample_totals.get(s2, 0))
        if not _overlap_flag(n1, n2, n_shared):
            continue
        n_windows = mut_windows.loc[list(shared)].nunique()
        flagged.append(
            {
                "sample_1": s1,
                "sample_2": s2,
                "n_shared": n_shared,
                "n_windows": n_windows,
                "n_total_1": n1,
                "n_total_2": n2,
            }
        )

    result = pd.DataFrame(flagged, columns=columns)
    if result.empty:
        logger.info(
            "No sample pairs met overlap reporting thresholds."
        )
    else:
        logger.warning(
            f"{len(result)} sample pair(s) flagged for suspicious "
            "mutation overlap -- review before trusting them as "
            "independent samples."
        )
    return result
