"""Tests for compute_mutation_burden.py's estimate_ell_hats().

Written alongside the L_low low-burden-correction rework: this
function previously had no explicit None-handling and relied on a
pandas `Series < None` quirk (silently all-False) to no-op --
`estimate_ell_hats`'s new explicit `L_low=None` default/early-return
is what this file locks in.
"""

import pandas as pd

from sigmutsel.compute_mutation_burden import (
    estimate_ell_hats,
)


def _burden_df(totals):
    return pd.DataFrame(
        {
            "total_mutations": totals,
            "synonymous_mutations": [0] * len(totals),
        },
        index=[f"s{i}" for i in range(len(totals))],
    )


def test_estimate_ell_hats_none_is_a_noop():
    mb = _burden_df([5, 30, 64, 200])
    result = estimate_ell_hats(mb)
    pd.testing.assert_series_equal(
        result, mb["total_mutations"].rename("ell_hats")
    )


def test_estimate_ell_hats_none_explicit_matches_default():
    mb = _burden_df([5, 30, 64, 200])
    assert estimate_ell_hats(mb, None, None).equals(
        estimate_ell_hats(mb)
    )


def test_estimate_ell_hats_blends_low_burden_samples():
    mb = _burden_df([10, 100])
    result = estimate_ell_hats(mb, L_low=64, L_high=500)
    # sample below L_low gets blended upward, toward L_low
    assert result["s0"] > 10
    assert result["s0"] < 64
    # sample above L_low is untouched
    assert result["s1"] == 100


def test_estimate_ell_hats_cut_at_l_low_clips_to_l_low():
    mb = _burden_df([10, 100])
    result = estimate_ell_hats(
        mb, L_low=64, L_high=500, cut_at_L_low=True
    )
    assert result["s0"] == 64
    assert result["s1"] == 100


def test_estimate_ell_hats_accepts_raw_mutation_df():
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["s0"] * 3 + ["s1"],
            "Variant_Classification": ["Missense_Mutation"] * 4,
        }
    )
    result = estimate_ell_hats(db)
    assert result["s0"] == 3
    assert result["s1"] == 1
