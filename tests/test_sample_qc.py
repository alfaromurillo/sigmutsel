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
    combine_sample_flags,
    compute_vaf_shape_score,
    flag_low_purity_samples,
    flag_vaf_shape_samples,
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


# --- compute_vaf_shape_score -----------------------------------------


def test_compute_vaf_shape_score_nan_purity():
    rows = pd.DataFrame(
        {"t_depth": [50] * 10, "t_alt_count": [25] * 10}
    )
    assert np.isnan(compute_vaf_shape_score(rows, np.nan))
    assert np.isnan(compute_vaf_shape_score(rows, None))


def test_compute_vaf_shape_score_too_few_variants():
    rows = pd.DataFrame(
        {"t_depth": [50, 50], "t_alt_count": [25, 25]}
    )
    assert np.isnan(
        compute_vaf_shape_score(rows, 0.5, min_variants=5)
    )


def test_compute_vaf_shape_score_matches_null_gives_high_pvalue():
    rng = np.random.default_rng(0)
    purity = 0.6
    depth = np.full(200, 100)
    alt = rng.binomial(depth, purity / 2)
    rows = pd.DataFrame({"t_depth": depth, "t_alt_count": alt})
    score = compute_vaf_shape_score(rows, purity)
    assert score > 0.01


def test_compute_vaf_shape_score_mismatched_purity_gives_low_pvalue():
    # Variants generated as if purity were much lower than claimed --
    # simulates under-calling/technical VAF depression.
    rng = np.random.default_rng(0)
    claimed_purity = 0.9
    true_purity = 0.1
    depth = np.full(200, 200)
    alt = rng.binomial(depth, true_purity / 2)
    rows = pd.DataFrame({"t_depth": depth, "t_alt_count": alt})
    score = compute_vaf_shape_score(rows, claimed_purity)
    assert score < 0.01


def test_compute_vaf_shape_score_respects_min_depth():
    rows = pd.DataFrame(
        {"t_depth": [5, 5, 5, 5, 5], "t_alt_count": [2, 2, 2, 2, 2]}
    )
    assert np.isnan(compute_vaf_shape_score(rows, 0.5, min_depth=20))


# --- flag_vaf_shape_samples -------------------------------------------


def _binomial_mutation_db(
    sample_barcode, purity, n_variants=200, depth=100, seed=0
):
    rng = np.random.default_rng(seed)
    d = np.full(n_variants, depth)
    alt = rng.binomial(d, purity / 2)
    return pd.DataFrame(
        {
            "Tumor_Sample_Barcode": [sample_barcode] * n_variants,
            "t_depth": d,
            "t_alt_count": alt,
        }
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

    flags = flag_vaf_shape_samples(
        mutation_db, purity_table, threshold=0.01
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
    flags = flag_vaf_shape_samples(
        mutation_db, purity_table, threshold=0.01
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
