"""A per-tumor record of what each quality-control step removes.

Most of the call QC drops rows while the MAF files are read, before
anything is stored, so the counts are lost unless they are kept as
they happen. The ledger keeps them: one row per (tumor, step) with
the number of rows that step removed or set aside for that tumor,
saved with the dataset (:attr:`models.MutationDataset.qc_ledger`).

Columns
-------
``tumor``
    The tumor's ``Tumor_Sample_Barcode`` (for a MAF file with no data
    rows, the file name).
``level``
    ``"calls"`` (``n`` counts MAF rows of that tumor) or ``"tumors"``
    (``n`` is 1: the tumor itself left at that step).
``step``
    The step's name; see :data:`STEP_ORDER`. The call-QC problem tags
    of :func:`qc.apply_qc` keep their names.
``order``
    Where the step runs, so a ledger sorts into production order.
``n``
    Rows removed or set aside. The one exception is ``"maf_rows"``,
    the tumor's total rows in its MAF file, which every other call
    step is subtracted from.

Steps that later stages can always recompute from the saved dataset
(the channel universe, calls on germline-masked alleles, sample-level
QC flags, the exon-end synonymous channel) are not recorded: their
inputs are saved, and recording them twice could only let the two
drift apart.
"""

import pandas as pd

LEDGER_COLUMNS = ["tumor", "level", "step", "order", "n"]

# Production order. Call steps at load (`load_maf_files.
# process_single_maf` with ``qc_mode``): the file's rows, the non-SNV
# rows, then `qc.apply_qc`'s tags in the order it assigns them (a row
# keeps only the first), then rows `load_maf_files.compact_data` drops
# (no gene). The multi-base tags count the rows set aside in
# `MutationDataset.multi_base_db`, not removed. Then the two-pass
# signature step (`MutationDataset.run_two_pass_signature_
# decomposition`). Tumor steps before loading are a caller's own
# selection (e.g. which MAF files are used) and take orders below 100.
STEP_ORDER = {
    "maf_rows": 100,
    "not_snv": 110,
    "invalid_chromosome": 120,
    "invalid_allele": 121,
    "invalid_context": 122,
    "duplicate_record": 130,
    "germline_variant_site": 140,
    "merged_into_dbs_variant": 150,
    "merged_with_nearby_variant": 151,
    "repetitive_region": 160,
    "compact_failed": 170,
    "treatment_signature_load": 200,
    "treatment_signature_sample": 201,
    "artifact_signature_mutation": 210,
}

# An apply_qc tag this module does not know yet still gets a place:
# after the known load-time tags, before compaction.
_UNKNOWN_LOAD_STEP_ORDER = 165

# The tags that set rows aside rather than drop them.
SET_ASIDE_STEPS = (
    "not_snv",
    "merged_into_dbs_variant",
    "merged_with_nearby_variant",
)


def empty_ledger():
    """A ledger with no rows and the ledger's columns."""
    return pd.DataFrame(
        {
            "tumor": pd.Series(dtype=object),
            "level": pd.Series(dtype=object),
            "step": pd.Series(dtype=object),
            "order": pd.Series(dtype="int64"),
            "n": pd.Series(dtype="int64"),
        }
    )


def step_order(step):
    """The production order of a step (see :data:`STEP_ORDER`)."""
    return STEP_ORDER.get(step, _UNKNOWN_LOAD_STEP_ORDER)


def ledger_rows(counts, level, step=None, order=None):
    """Ledger rows from per-tumor counts.

    Parameters
    ----------
    counts : pandas.Series
        Indexed by tumor, or by (tumor, step) when *step* is None.
        Zero counts are dropped.
    level : {"calls", "tumors"}
    step : str or None
        The step of every row; None reads it from the index's second
        level.
    order : int or None
        Overrides :func:`step_order` (for a caller's own steps).

    Returns
    -------
    pandas.DataFrame
        Rows with :data:`LEDGER_COLUMNS`.
    """
    counts = counts[counts > 0]
    if counts.empty:
        return empty_ledger()
    if step is None:
        frame = counts.rename("n").reset_index()
        frame.columns = ["tumor", "step", "n"]
    else:
        frame = counts.rename("n").reset_index()
        frame.columns = ["tumor", "n"]
        frame["step"] = step
    frame["level"] = level
    frame["order"] = (
        frame["step"].map(step_order) if order is None else order
    )
    frame["n"] = frame["n"].astype("int64")
    frame["order"] = frame["order"].astype("int64")
    return frame[LEDGER_COLUMNS]


def load_rows(raw, snv, tagged, compacted, tumor_col):
    """The load-time ledger rows of one MAF file.

    Parameters
    ----------
    raw : pandas.DataFrame
        Every row read from the file.
    snv : pandas.DataFrame
        The single-base rows (`load_maf_files.filter_db`).
    tagged : pandas.DataFrame
        *snv* with `qc.apply_qc`'s ``problem`` column.
    compacted : pandas.DataFrame
        What `load_maf_files.compact_data` returns for the rows kept
        (clean rows and multi-base rows), with a ``multi_base`` column
        when multi-base rows were kept.
    tumor_col : str
        The tumor column (``"Tumor_Sample_Barcode"``).

    Returns
    -------
    pandas.DataFrame
        Rows with :data:`LEDGER_COLUMNS`. Per tumor, ``maf_rows``
        equals the sum of the other steps plus the clean rows of
        *compacted*.
    """
    from .qc import MULTI_BASE_PROBLEMS

    parts = []
    total = raw.groupby(tumor_col).size()
    parts.append(ledger_rows(total, "calls", "maf_rows"))
    n_snv = (
        snv.groupby(tumor_col)
        .size()
        .reindex(total.index, fill_value=0)
    )
    parts.append(ledger_rows(total - n_snv, "calls", "not_snv"))

    problem = tagged["problem"]
    removed = tagged[
        problem.notna() & ~problem.isin(MULTI_BASE_PROBLEMS)
    ]
    if len(removed):
        parts.append(
            ledger_rows(
                removed.groupby([tumor_col, "problem"]).size(),
                "calls",
            )
        )
    # Multi-base rows count as set aside once compacted, so a row that
    # fails compaction is counted there and only there.
    if "multi_base" in compacted.columns:
        tag_of = {v: k for k, v in MULTI_BASE_PROBLEMS.items()}
        aside = compacted[compacted["multi_base"].notna()]
        if len(aside):
            by_tag = aside.groupby(
                [aside[tumor_col], aside["multi_base"].map(tag_of)]
            ).size()
            parts.append(ledger_rows(by_tag, "calls"))
    kept_before = (
        tagged[problem.isna() | problem.isin(MULTI_BASE_PROBLEMS)]
        .groupby(tumor_col)
        .size()
    )
    kept_after = compacted.groupby(tumor_col).size()
    failed = kept_before.sub(kept_after, fill_value=0)
    parts.append(ledger_rows(failed, "calls", "compact_failed"))
    return concat(parts)


def concat(ledgers):
    """Concatenate ledgers, sorted by tumor and production order."""
    ledgers = [x for x in ledgers if x is not None and len(x)]
    if not ledgers:
        return empty_ledger()
    out = pd.concat(ledgers, ignore_index=True)
    return out.sort_values(["tumor", "order", "step"]).reset_index(
        drop=True
    )


def replace_steps(ledger, rows, steps):
    """*ledger* with *steps* replaced by *rows* (a step rerun)."""
    if ledger is None:
        return None
    kept = ledger[~ledger["step"].isin(steps)]
    return concat([kept, rows])


def totals(ledger):
    """Per-step totals over tumors, in production order.

    Returns
    -------
    pandas.DataFrame
        Columns ``level``, ``step``, ``order``, ``n`` (summed) and
        ``tumors`` (tumors with a nonzero count).
    """
    return (
        ledger.groupby(["level", "step", "order"], as_index=False)
        .agg(n=("n", "sum"), tumors=("tumor", "nunique"))
        .sort_values(["order", "step"])
        .reset_index(drop=True)
    )
