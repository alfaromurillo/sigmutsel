"""Tests for sample_qc.py.

Per-sample (not per-mutation-row) confidence flags -- tumor purity
and VAF-distribution shape -- built for the L_low low-burden-
correction rework. See TODO.md's "TCGA sample selection and MAF
preprocessing QC" entry for the background.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.sample_qc import (
    annotate_local_copy_number,
    combine_sample_flags,
    compute_vaf_shape_score,
    flag_low_purity_samples,
    flag_unverified_samples,
    flag_vaf_shape_samples,
    load_copy_number_segments,
    load_copy_number_segments_from_file,
)

# --- flag_low_purity_samples ----------------------------------------


def test_flag_low_purity_samples_below_threshold():
    purity_table = pd.DataFrame(
        {
            "array": [
                "TCGA-AA-0001-01",
                "TCGA-AA-0002-01",
                "TCGA-AA-0003-01",
            ],
            "purity": [0.1, 0.5, 0.9],
        }
    )
    flags = flag_low_purity_samples(purity_table, threshold=0.30)
    assert (
        flags["TCGA-AA-0001-01"] is True or flags["TCGA-AA-0001-01"]
    )
    assert not flags["TCGA-AA-0002-01"]
    assert not flags["TCGA-AA-0003-01"]


def test_flag_low_purity_samples_nan_not_flagged():
    purity_table = pd.DataFrame(
        {"array": ["TCGA-AA-0001-01"], "purity": [np.nan]}
    )
    flags = flag_low_purity_samples(purity_table, threshold=0.30)
    assert not flags["TCGA-AA-0001-01"]


def test_flag_low_purity_samples_duplicate_barcode_keeps_first():
    purity_table = pd.DataFrame(
        {
            "array": ["TCGA-AA-0001-01", "TCGA-AA-0001-01"],
            "purity": [0.1, 0.9],
        }
    )
    flags = flag_low_purity_samples(purity_table, threshold=0.30)
    assert len(flags) == 1
    assert bool(flags["TCGA-AA-0001-01"])


def test_flag_low_purity_samples_custom_columns():
    purity_table = pd.DataFrame(
        {"Sample.ID": ["TCGA-AA-0001-01"], "CPE": [0.1]}
    )
    flags = flag_low_purity_samples(
        purity_table,
        barcode_column="Sample.ID",
        purity_column="CPE",
        threshold=0.30,
    )
    assert bool(flags["TCGA-AA-0001-01"])


# --- flag_unverified_samples --------------------------------------------


def test_flag_unverified_samples_flags_missing_purity():
    purity_table = pd.DataFrame(
        {
            "array": ["TCGA-AA-0001-01", "TCGA-AA-0002-01"],
            "purity": [0.6, np.nan],
        }
    )
    sample_barcodes = [
        "TCGA-AA-0001-01A-11D-0001-01",
        "TCGA-AA-0002-01A-11D-0001-01",
    ]
    unverified = flag_unverified_samples(
        sample_barcodes, purity_table
    )
    assert not unverified["TCGA-AA-0001-01A-11D-0001-01"]
    assert unverified["TCGA-AA-0002-01A-11D-0001-01"]


def test_flag_unverified_samples_flags_absent_from_purity_table():
    purity_table = pd.DataFrame(
        {"array": ["TCGA-AA-0001-01"], "purity": [0.6]}
    )
    sample_barcodes = ["TCGA-ZZ-9999-01A-11D-0001-01"]
    unverified = flag_unverified_samples(
        sample_barcodes, purity_table
    )
    assert unverified["TCGA-ZZ-9999-01A-11D-0001-01"]


def test_flag_unverified_samples_custom_columns():
    purity_table = pd.DataFrame(
        {"Sample.ID": ["TCGA-AA-0001-01"], "CPE": [np.nan]}
    )
    sample_barcodes = ["TCGA-AA-0001-01A-11D-0001-01"]
    unverified = flag_unverified_samples(
        sample_barcodes,
        purity_table,
        barcode_column="Sample.ID",
        purity_column="CPE",
    )
    assert unverified["TCGA-AA-0001-01A-11D-0001-01"]


def test_flag_unverified_samples_preserves_full_barcode_index():
    purity_table = pd.DataFrame(
        {"array": ["TCGA-AA-0001-01"], "purity": [0.6]}
    )
    sample_barcodes = ["TCGA-AA-0001-01A-11D-0001-01"]
    unverified = flag_unverified_samples(
        sample_barcodes, purity_table
    )
    assert list(unverified.index) == sample_barcodes


# --- load_copy_number_segments / annotate_local_copy_number ----------


def _segments_table():
    return pd.DataFrame(
        {
            "Sample": [
                "TCGA-AA-0001-01",
                "TCGA-AA-0001-01",
                "TCGA-AA-0002-01",
            ],
            "Chromosome": [1.0, 1.0, 23.0],
            "Start": [1, 1001, 1],
            "End": [1000, 2000, 5000],
            "Modal_Total_CN": [2.0, 4.0, 1.0],
        }
    )


def test_load_copy_number_segments_keys_and_sorts():
    segments = load_copy_number_segments(_segments_table())
    starts, ends, cn = segments[("TCGA-AA-0001-01", "1")]
    assert list(starts) == [1, 1001]
    assert list(ends) == [1000, 2000]
    assert list(cn) == [2.0, 4.0]
    # X coded as 23 in the source table, normalized the same way here
    assert ("TCGA-AA-0002-01", "23") in segments


def test_annotate_local_copy_number_matches_by_position():
    mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": [
                "TCGA-AA-0001-01A-11D-0001-01",
                "TCGA-AA-0001-01A-11D-0001-01",
            ],
            "Chromosome": ["chr1", "chr1"],
            "Start_Position": [500, 1500],
        }
    )
    segments = load_copy_number_segments(_segments_table())
    annotated = annotate_local_copy_number(mutation_db, segments)
    assert list(annotated["local_cn"]) == [2.0, 4.0]


def test_annotate_local_copy_number_no_segment_is_nan():
    mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["TCGA-AA-0001-01A-11D-0001-01"],
            "Chromosome": ["chrY"],  # ABSOLUTE never calls Y
            "Start_Position": [500],
        }
    )
    segments = load_copy_number_segments(_segments_table())
    annotated = annotate_local_copy_number(mutation_db, segments)
    assert annotated["local_cn"].isna().all()


def test_annotate_local_copy_number_position_outside_any_segment():
    mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["TCGA-AA-0001-01A-11D-0001-01"],
            "Chromosome": ["chr1"],
            "Start_Position": [1_000_000],  # past both segments
        }
    )
    segments = load_copy_number_segments(_segments_table())
    annotated = annotate_local_copy_number(mutation_db, segments)
    assert annotated["local_cn"].isna().all()


# --- compute_vaf_shape_score -----------------------------------------


def test_compute_vaf_shape_score_nan_purity():
    rows = pd.DataFrame(
        {
            "t_depth": [50] * 10,
            "t_alt_count": [25] * 10,
            "local_cn": [2.0] * 10,
        }
    )
    assert np.isnan(compute_vaf_shape_score(rows, np.nan))
    assert np.isnan(compute_vaf_shape_score(rows, None))


def test_compute_vaf_shape_score_too_few_variants():
    rows = pd.DataFrame(
        {
            "t_depth": [50, 50],
            "t_alt_count": [25, 25],
            "local_cn": [2.0, 2.0],
        }
    )
    assert np.isnan(
        compute_vaf_shape_score(rows, 0.5, min_variants=5)
    )


def test_compute_vaf_shape_score_clonal_diploid_gives_ccf_near_one():
    # VAF generated exactly at the clonal expectation for purity=0.6,
    # copy number 2 -- ccf_hat should recover ~1.
    rng = np.random.default_rng(0)
    purity = 0.6
    depth = np.full(200, 100)
    alt = rng.binomial(depth, purity / 2)
    rows = pd.DataFrame(
        {
            "t_depth": depth,
            "t_alt_count": alt,
            "local_cn": [2.0] * 200,
        }
    )
    score = compute_vaf_shape_score(rows, purity)
    # Taking the 75th percentile of a noisy-but-unbiased estimator is
    # expected to land a bit above the true value 1.0 by construction
    # (75% of the sampling distribution sits below it) -- a wider,
    # not tight, tolerance here.
    assert 0.85 < score < 1.25


def test_compute_vaf_shape_score_under_called_sample_gives_low_ccf():
    # Variants generated as if purity were much lower than claimed --
    # simulates under-calling/technical VAF depression.
    rng = np.random.default_rng(0)
    claimed_purity = 0.9
    true_purity = 0.1
    depth = np.full(200, 200)
    alt = rng.binomial(depth, true_purity / 2)
    rows = pd.DataFrame(
        {
            "t_depth": depth,
            "t_alt_count": alt,
            "local_cn": [2.0] * 200,
        }
    )
    score = compute_vaf_shape_score(rows, claimed_purity)
    assert score < 0.7


def test_compute_vaf_shape_score_corrects_for_amplified_locus():
    # Same clonal mutation, but on a copy-number-4 segment instead of
    # diploid -- VAF is naturally lower (more total copies), but
    # ccf_hat should still recover ~1 once corrected, showing the
    # copy-number term in the formula is doing real work (this is
    # exactly the case the earlier flat-diploid design got wrong).
    rng = np.random.default_rng(1)
    purity = 0.6
    q = 4
    depth = np.full(200, 200)
    p_true = purity / (2 * (1 - purity) + purity * q)
    alt = rng.binomial(depth, p_true)
    rows = pd.DataFrame(
        {"t_depth": depth, "t_alt_count": alt, "local_cn": [q] * 200}
    )
    score = compute_vaf_shape_score(rows, purity)
    assert 0.85 < score < 1.15


def test_compute_vaf_shape_score_respects_min_depth():
    rows = pd.DataFrame(
        {
            "t_depth": [5, 5, 5, 5, 5],
            "t_alt_count": [2, 2, 2, 2, 2],
            "local_cn": [2.0] * 5,
        }
    )
    assert np.isnan(compute_vaf_shape_score(rows, 0.5, min_depth=20))


def test_compute_vaf_shape_score_excludes_homozygous_deletion():
    # local_cn == 0 rows should be dropped, not divide by a degenerate
    # denominator or otherwise produce a usable-looking value.
    rows = pd.DataFrame(
        {
            "t_depth": [50] * 5,
            "t_alt_count": [25] * 5,
            "local_cn": [0.0] * 5,
        }
    )
    assert np.isnan(
        compute_vaf_shape_score(rows, 0.5, min_variants=5)
    )


# --- flag_vaf_shape_samples -------------------------------------------


def _binomial_mutation_db(
    sample_barcode,
    purity,
    n_variants=200,
    depth=100,
    seed=0,
    chromosome="chr1",
    start=500,
):
    rng = np.random.default_rng(seed)
    d = np.full(n_variants, depth)
    alt = rng.binomial(d, purity / 2)
    return pd.DataFrame(
        {
            "Tumor_Sample_Barcode": [sample_barcode] * n_variants,
            "Chromosome": [chromosome] * n_variants,
            "Start_Position": [start] * n_variants,
            "t_depth": d,
            "t_alt_count": alt,
        }
    )


def _diploid_segments(*short_barcodes):
    return load_copy_number_segments(
        pd.DataFrame(
            {
                "Sample": list(short_barcodes),
                "Chromosome": [1.0] * len(short_barcodes),
                "Start": [1] * len(short_barcodes),
                "End": [10_000_000] * len(short_barcodes),
                "Modal_Total_CN": [2.0] * len(short_barcodes),
            }
        )
    )


def test_flag_vaf_shape_samples_end_to_end():
    good_sample = "TCGA-AA-0001-01A-11D-0001-01"
    bad_sample = "TCGA-AA-0002-01A-11D-0001-01"

    mutation_db = pd.concat(
        [
            _binomial_mutation_db(good_sample, purity=0.6, seed=1),
            # bad_sample's variants generated at a much lower true
            # purity than the table claims -- mismatch should flag it.
            _binomial_mutation_db(bad_sample, purity=0.1, seed=2),
        ],
        ignore_index=True,
    )
    purity_table = pd.DataFrame(
        {
            "array": ["TCGA-AA-0001-01", "TCGA-AA-0002-01"],
            "purity": [0.6, 0.9],
        }
    )
    segments = _diploid_segments("TCGA-AA-0001-01", "TCGA-AA-0002-01")

    flags = flag_vaf_shape_samples(
        mutation_db, purity_table, segments, threshold=0.7
    )

    assert not flags[good_sample]
    assert flags[bad_sample]


def test_flag_vaf_shape_samples_barcode_truncation():
    # purity table's barcode is 15 chars; mutation_db's is the longer
    # full aliquot barcode -- must match on the shared prefix.
    sample = "TCGA-AA-0001-01A-11D-0001-01"
    mutation_db = _binomial_mutation_db(sample, purity=0.6, seed=3)
    purity_table = pd.DataFrame(
        {"array": ["TCGA-AA-0001-01"], "purity": [0.6]}
    )
    segments = _diploid_segments("TCGA-AA-0001-01")
    flags = flag_vaf_shape_samples(
        mutation_db, purity_table, segments, threshold=0.7
    )
    assert sample in flags.index


# --- combine_sample_flags ---------------------------------------------


def test_combine_sample_flags_any():
    a = pd.Series({"s1": True, "s2": False, "s3": False})
    b = pd.Series({"s1": False, "s2": True, "s3": False})
    combined = combine_sample_flags(a, b, how="any")
    assert combined["s1"]
    assert combined["s2"]
    assert not combined["s3"]


def test_combine_sample_flags_all():
    a = pd.Series({"s1": True, "s2": True})
    b = pd.Series({"s1": False, "s2": True})
    combined = combine_sample_flags(a, b, how="all")
    assert not combined["s1"]
    assert combined["s2"]


def test_combine_sample_flags_union_of_indices():
    a = pd.Series({"s1": True})
    b = pd.Series({"s2": True})
    combined = combine_sample_flags(a, b, how="any")
    assert combined["s1"]
    assert combined["s2"]


def test_combine_sample_flags_invalid_how():
    a = pd.Series({"s1": True})
    with pytest.raises(ValueError):
        combine_sample_flags(a, how="bogus")


def test_combine_sample_flags_warns_above_threshold(caplog):
    # 2/20 = 10% flagged, above the default 5% warn_threshold
    a = pd.Series({f"s{i}": i < 2 for i in range(20)})
    with caplog.at_level("WARNING", logger="sigmutsel.sample_qc"):
        combine_sample_flags(a)
    assert any("10.0%" in record.message for record in caplog.records)


def test_combine_sample_flags_no_warning_below_threshold(caplog):
    # 1/20 = 5%, not strictly above the default 5% threshold
    a = pd.Series({f"s{i}": i < 1 for i in range(20)})
    with caplog.at_level("WARNING", logger="sigmutsel.sample_qc"):
        combine_sample_flags(a)
    assert not any(
        "warn_threshold" in record.message
        or "flagged" in record.message
        for record in caplog.records
    )


def test_combine_sample_flags_warn_threshold_none_disables():
    a = pd.Series({f"s{i}": True for i in range(20)})
    # Should not raise/log even though 100% is flagged -- just
    # confirm it runs cleanly with warnings disabled.
    combined = combine_sample_flags(a, warn_threshold=None)
    assert combined.all()


def test_combine_sample_flags_custom_warn_threshold(caplog):
    a = pd.Series({f"s{i}": i < 1 for i in range(20)})  # 5%
    with caplog.at_level("WARNING", logger="sigmutsel.sample_qc"):
        combine_sample_flags(a, warn_threshold=0.01)
    assert any("5.0%" in record.message for record in caplog.records)


# --- load_copy_number_segments_from_file (caching) ---------------------


def test_load_copy_number_segments_from_file_builds_and_caches(
    tmp_path,
):
    segments_path = tmp_path / "segments.txt"
    _segments_table().to_csv(segments_path, sep="\t", index=False)
    cache_path = tmp_path / "segments.pkl"

    assert not cache_path.exists()
    segments = load_copy_number_segments_from_file(
        segments_path, cache_path=cache_path
    )
    assert cache_path.exists()
    _, _, cn = segments[("TCGA-AA-0001-01", "1")]
    assert list(cn) == [2.0, 4.0]


def test_load_copy_number_segments_from_file_cache_hit_skips_parse(
    tmp_path, monkeypatch
):
    segments_path = tmp_path / "segments.txt"
    _segments_table().to_csv(segments_path, sep="\t", index=False)
    cache_path = tmp_path / "segments.pkl"

    # Build the cache once for real.
    load_copy_number_segments_from_file(
        segments_path, cache_path=cache_path
    )

    # Second call: reading segments_path at all must not happen.
    real_read_csv = pd.read_csv

    def _boom(path, *args, **kwargs):
        if str(path) == str(segments_path):
            raise AssertionError(
                "should not re-read segments_path on a cache hit"
            )
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _boom)
    segments = load_copy_number_segments_from_file(
        segments_path, cache_path=cache_path
    )
    _, _, cn = segments[("TCGA-AA-0001-01", "1")]
    assert list(cn) == [2.0, 4.0]


def test_load_copy_number_segments_from_file_default_cache_path(
    tmp_path,
):
    segments_path = tmp_path / "segments.txt"
    _segments_table().to_csv(segments_path, sep="\t", index=False)

    load_copy_number_segments_from_file(segments_path)
    assert (tmp_path / "segments.txt.pkl").exists()
