"""Tests for qc.py.

Covers each check individually (mostly via small hand-built
DataFrames, since the full validate_* pipeline is already tested via
load_maf_files) and check_sample_overlap's threshold logic, ported
from cancereffectsizeR.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.qc import (
    apply_qc,
    check_sample_overlap,
    detect_mnv_dbs,
    flag_artifact_signature_mutations,
    flag_exact_duplicates,
    flag_germline_variants,
    flag_repetitive_regions,
    load_repeat_intervals,
    summarize_problems,
    validate_full_with_problems,
)


def _valid_snv_row(**overrides):
    row = {
        "Tumor_Sample_Barcode": "TCGA-AA-0001-01A",
        "Chromosome": "chr1",
        "Variant_Classification": "Missense_Mutation",
        "Start_Position": 1000,
        "Reference_Allele": "C",
        "Tumor_Seq_Allele2": "T",
        "CONTEXT": "AAAAACAAAAA",
    }
    row.update(overrides)
    return row


# --- validate_full_with_problems -----------------------------------------


def test_validate_full_with_problems_keeps_all_rows():
    df = pd.DataFrame(
        [
            _valid_snv_row(),
            _valid_snv_row(Chromosome="chrZZ"),  # invalid chromosome
        ]
    )
    tagged = validate_full_with_problems(df)
    assert len(tagged) == len(df)  # nothing dropped
    assert tagged["problem"].isna().sum() == 1
    assert tagged.loc[1, "problem"] == "invalid_chromosome"


def test_validate_full_with_problems_reports_first_failure_only():
    # Bad chromosome AND bad allele -- should only report the first
    # check that catches it (chromosome runs before allele checks).
    df = pd.DataFrame(
        [_valid_snv_row(Chromosome="bad", Reference_Allele="N")]
    )
    tagged = validate_full_with_problems(df)
    assert tagged.loc[0, "problem"] == "invalid_chromosome"


# --- flag_exact_duplicates ------------------------------------------------


def test_flag_exact_duplicates_keeps_first():
    df = pd.DataFrame([_valid_snv_row(), _valid_snv_row()])
    tagged = flag_exact_duplicates(df)
    assert tagged.loc[0, "problem"] is None
    assert tagged.loc[1, "problem"] == "duplicate_record"


def test_flag_exact_duplicates_no_false_positive():
    df = pd.DataFrame(
        [_valid_snv_row(), _valid_snv_row(Start_Position=1001)]
    )
    tagged = flag_exact_duplicates(df)
    assert tagged["problem"].isna().all()


# --- flag_germline_variants -----------------------------------------------


def test_flag_germline_variants_above_threshold():
    df = pd.DataFrame(
        [
            _valid_snv_row(gnomAD_non_cancer_MAX_AF_adj=0.5),
            _valid_snv_row(gnomAD_non_cancer_MAX_AF_adj=0.0001),
        ]
    )
    tagged = flag_germline_variants(df)
    assert tagged.loc[0, "problem"] == "germline_variant_site"
    assert tagged.loc[1, "problem"] is None


def test_flag_germline_variants_missing_af_not_flagged():
    df = pd.DataFrame(
        [_valid_snv_row(gnomAD_non_cancer_MAX_AF_adj=None)]
    )
    tagged = flag_germline_variants(df)
    assert tagged["problem"].isna().all()


def test_flag_germline_variants_missing_column_is_noop():
    df = pd.DataFrame([_valid_snv_row()])
    tagged = flag_germline_variants(df)
    assert tagged["problem"].isna().all()


# --- detect_mnv_dbs --------------------------------------------------------


def test_detect_mnv_dbs_merges_dinucleotide_pair():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1001),
        ]
    )
    tagged = detect_mnv_dbs(df)
    assert (tagged["problem"] == "merged_into_dbs_variant").all()


def test_detect_mnv_dbs_merges_larger_cluster_as_other():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1001),
            _valid_snv_row(Start_Position=1002),
        ]
    )
    tagged = detect_mnv_dbs(df)
    assert (tagged["problem"] == "merged_with_nearby_variant").all()


def test_detect_mnv_dbs_2bp_gap_is_other_not_dbs():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1002),
        ]
    )
    tagged = detect_mnv_dbs(df)
    assert (tagged["problem"] == "merged_with_nearby_variant").all()


def test_detect_mnv_dbs_leaves_isolated_snv_alone():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=5000),
        ]
    )
    tagged = detect_mnv_dbs(df)
    assert tagged["problem"].isna().all()


def test_detect_mnv_dbs_does_not_chain_across_samples():
    df = pd.DataFrame(
        [
            _valid_snv_row(
                Tumor_Sample_Barcode="A", Start_Position=1000
            ),
            _valid_snv_row(
                Tumor_Sample_Barcode="B", Start_Position=1001
            ),
        ]
    )
    tagged = detect_mnv_dbs(df)
    assert tagged["problem"].isna().all()


def test_detect_mnv_dbs_skips_already_tagged_rows():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1001),
        ]
    )
    df["problem"] = ["duplicate_record", None]
    tagged = detect_mnv_dbs(df)
    # The already-tagged row keeps its original reason and can't
    # chain the untagged one into a false pair.
    assert tagged.loc[0, "problem"] == "duplicate_record"
    assert tagged.loc[1, "problem"] is None


# --- apply_qc ---------------------------------------------------------


def test_apply_qc_runs_all_checks():
    df = pd.DataFrame(
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1000),  # exact dup
            _valid_snv_row(Chromosome="bad"),  # invalid
        ]
    )
    tagged = apply_qc(df)
    assert len(tagged) == 3
    problems = summarize_problems(tagged)
    assert problems["duplicate_record"] == 1
    assert problems["invalid_chromosome"] == 1


# --- check_sample_overlap --------------------------------------------


def _mutation_db_row(sample, pos, mtype="A[C>T]A", chrom="chr1"):
    return {
        "Tumor_Sample_Barcode": sample,
        "Chromosome": chrom,
        "Start_Position": pos,
        "type": mtype,
    }


def test_check_sample_overlap_flags_small_samples_with_any_shared():
    # Both samples have <6 total mutations and share one -- flagged
    # by the smallest threshold (variants_A<6 & variants_B<6 & shared>0).
    rows = [
        _mutation_db_row("S1", 1000),
        _mutation_db_row("S1", 1001),
        _mutation_db_row("S2", 1000),
        _mutation_db_row("S2", 2000),
    ]
    df = pd.DataFrame(rows)
    result = check_sample_overlap(df)
    assert len(result) == 1
    assert set(result.iloc[0][["sample_1", "sample_2"]]) == {
        "S1",
        "S2",
    }
    assert result.iloc[0]["n_shared"] == 1


def test_check_sample_overlap_clean_cohort_returns_empty():
    rows = [
        _mutation_db_row("S1", 1000),
        _mutation_db_row("S2", 2000),
        _mutation_db_row("S3", 3000),
    ]
    df = pd.DataFrame(rows)
    result = check_sample_overlap(df)
    assert result.empty


def test_check_sample_overlap_shared_hotspot_alone_not_flagged():
    # Two large, otherwise-independent samples sharing one common
    # hotspot mutation (e.g. both have TP53 R175H) should NOT be
    # flagged -- this is the normal, expected case, not contamination.
    rows = [_mutation_db_row("S1", 1000)] + [
        _mutation_db_row("S1", 2000 + i) for i in range(30)
    ]
    rows += [_mutation_db_row("S2", 1000)] + [
        _mutation_db_row("S2", 9000 + i) for i in range(30)
    ]
    df = pd.DataFrame(rows)
    result = check_sample_overlap(df)
    assert result.empty


def test_check_sample_overlap_requires_columns():
    df = pd.DataFrame({"Tumor_Sample_Barcode": ["S1"]})
    with pytest.raises(KeyError):
        check_sample_overlap(df)


# --- flag_repetitive_regions / load_repeat_intervals ------------------


def _repeat_intervals(chrom, pairs):
    starts = np.array([p[0] for p in pairs])
    ends = np.array([p[1] for p in pairs])
    return {chrom: (starts, ends)}


def test_flag_repetitive_regions_inside_interval():
    # 0-based half-open [1000, 1010) covers 1-based positions
    # 1001..1010.
    intervals = _repeat_intervals("chr1", [(1000, 1010)])
    df = pd.DataFrame(
        [
            _valid_snv_row(Chromosome="chr1", Start_Position=1005),
            _valid_snv_row(Chromosome="chr1", Start_Position=1010),
        ]
    )
    tagged = flag_repetitive_regions(df, intervals)
    assert (tagged["problem"] == "repetitive_region").all()


def test_flag_repetitive_regions_boundary_excluded():
    intervals = _repeat_intervals("chr1", [(1000, 1010)])
    df = pd.DataFrame(
        [
            _valid_snv_row(Chromosome="chr1", Start_Position=1000),
            _valid_snv_row(Chromosome="chr1", Start_Position=1011),
        ]
    )
    tagged = flag_repetitive_regions(df, intervals)
    assert tagged["problem"].isna().all()


def test_flag_repetitive_regions_unknown_chrom_is_noop():
    intervals = _repeat_intervals("chr1", [(1000, 1010)])
    df = pd.DataFrame(
        [_valid_snv_row(Chromosome="chr2", Start_Position=1005)]
    )
    tagged = flag_repetitive_regions(df, intervals)
    assert tagged["problem"].isna().all()


# --- flag_artifact_signature_mutations ---------------------------------


def test_flag_artifact_signature_mutations_above_threshold():
    df = pd.DataFrame(
        [_valid_snv_row(), _valid_snv_row(), _valid_snv_row()]
    )
    probs = pd.Series([0.9, 0.4, 0.5], index=df.index)
    tagged = flag_artifact_signature_mutations(
        df, probs, threshold=0.5
    )
    assert tagged.loc[0, "problem"] == "artifact_signature_mutation"
    assert pd.isna(tagged.loc[1, "problem"])
    # exactly at threshold is not flagged (strictly greater than)
    assert pd.isna(tagged.loc[2, "problem"])


def test_flag_artifact_signature_mutations_plain_array_same_order():
    df = pd.DataFrame([_valid_snv_row(), _valid_snv_row()])
    tagged = flag_artifact_signature_mutations(
        df, [0.9, 0.1], threshold=0.5
    )
    assert tagged.loc[0, "problem"] == "artifact_signature_mutation"
    assert pd.isna(tagged.loc[1, "problem"])


def test_flag_artifact_signature_mutations_respects_prior_tags():
    df = pd.DataFrame([_valid_snv_row(), _valid_snv_row()])
    df["problem"] = ["germline_variant_site", None]
    probs = pd.Series([0.9, 0.9], index=df.index)
    tagged = flag_artifact_signature_mutations(
        df, probs, threshold=0.5
    )
    # row 0 keeps its earlier tag rather than being overwritten
    assert tagged.loc[0, "problem"] == "germline_variant_site"
    assert tagged.loc[1, "problem"] == "artifact_signature_mutation"


def test_flag_artifact_signature_mutations_missing_series_entries_not_flagged():
    df = pd.DataFrame(
        [_valid_snv_row(), _valid_snv_row()], index=[10, 11]
    )
    # Series only covers index 10 -- index 11 should default to 0
    # probability (not flagged), not raise or propagate NaN.
    probs = pd.Series([0.9], index=[10])
    tagged = flag_artifact_signature_mutations(
        df, probs, threshold=0.5
    )
    assert tagged.loc[10, "problem"] == "artifact_signature_mutation"
    assert pd.isna(tagged.loc[11, "problem"])


def test_load_repeat_intervals_parses_rmsk_format(tmp_path):
    # No header, 17 tab-separated columns -- matches UCSC's raw rmsk
    # dump (verified live 2026-08-22; see qc.py's _RMSK_COLUMN_NAMES
    # comment). Two overlapping rows on chr1 should merge into one
    # interval; chr2 stays separate.
    rows = [
        "585\t463\t13\t6\t17\tchr1\t1000\t1010\t-1\t+\tA\tB\tC\t1\t2\t0\t1",
        "585\t463\t13\t6\t17\tchr1\t1005\t1020\t-1\t+\tA\tB\tC\t1\t2\t0\t2",
        "585\t463\t13\t6\t17\tchr2\t5000\t5010\t-1\t+\tA\tB\tC\t1\t2\t0\t3",
    ]
    # read_csv sniffs .gz by extension; use a plain (uncompressed)
    # filename since pandas reads either transparently.
    plain_path = tmp_path / "rmsk.hg38.txt"
    plain_path.write_text("\n".join(rows) + "\n")

    intervals = load_repeat_intervals(plain_path)
    assert set(intervals.keys()) == {"chr1", "chr2"}
    chr1_starts, chr1_ends = intervals["chr1"]
    assert list(chr1_starts) == [1000]
    assert list(chr1_ends) == [1020]
    chr2_starts, chr2_ends = intervals["chr2"]
    assert list(chr2_starts) == [5000]
    assert list(chr2_ends) == [5010]
