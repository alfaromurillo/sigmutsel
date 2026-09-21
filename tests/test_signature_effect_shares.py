"""Tests for signature effect shares.

The cohort average is checked against numbers worked out by hand on
a two-tumor, two-signature cohort, so the test does not recompute
the implementation's own formula. Beyond that, the properties the
quantity exists for: a tumor counts once however many mutations it
carries, the source shares it is compared against cover the same
tumors, and units without a fitted gamma drop out rather than count
as zero.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.models import Model, MutationDataset
from sigmutsel.signature_attribution import (
    compute_signature_effect_shares,
    compute_signature_probabilities,
    compute_signature_probability_mass,
)

_SIG_MATRIX = pd.DataFrame(
    {"S1": [0.8, 0.4], "S2": [0.2, 0.6]},
    index=pd.Index(["A[C>T]A", "A[C>A]A"], name="MutationType"),
)


def _db(rows):
    """Rows of (tumor, gene, type, classification)."""
    return pd.DataFrame(
        rows,
        columns=[
            "Tumor_Sample_Barcode",
            "ensembl_gene_id",
            "type",
            "Variant_Classification",
        ],
    ).assign(variant="X p.A1B")


_ROWS = [
    ("T1", "G1", "A[C>T]A", "Missense_Mutation"),
    ("T1", "G1", "A[C>A]A", "Missense_Mutation"),
    ("T2", "G1", "A[C>T]A", "Missense_Mutation"),
    ("T2", "G2", "A[C>A]A", "Missense_Mutation"),
]

# Chosen so alphas come out (0.5, 0.5) and (0.25, 0.75): every
# sample has two mutations, so the counts are halves of them.
_ASSIGNMENTS = pd.DataFrame(
    {"S1": [1.0, 0.5], "S2": [1.0, 1.5]}, index=["T1", "T2"]
)

_GAMMAS = {"G1": 10.0, "G2": 1.0}


def _shares(db=None, assignments=None, **kwargs):
    db = _db(_ROWS) if db is None else db
    assignments = _ASSIGNMENTS if assignments is None else assignments
    from sigmutsel.compute_alphas import estimate_alphas

    alphas = estimate_alphas(db, assignments)
    probabilities = compute_signature_probabilities(
        db, assignments, _SIG_MATRIX, alphas=alphas
    )
    return compute_signature_effect_shares(
        probabilities,
        db["Tumor_Sample_Barcode"].values,
        db["ensembl_gene_id"].values,
        kwargs.pop("gammas", _GAMMAS),
        source_shares=alphas,
        **kwargs,
    )


def test_attribution_is_bayes_over_the_weight_simplex():
    db = _db(_ROWS)
    probabilities = compute_signature_probabilities(
        db, _ASSIGNMENTS, _SIG_MATRIX
    )
    # T1 has alphas (0.5, 0.5): A[C>T]A gives (0.4, 0.1) before
    # normalizing, so (0.8, 0.2). T2 has (0.25, 0.75), so A[C>A]A
    # gives (0.1, 0.45) -> (2/11, 9/11).
    assert np.allclose(probabilities.iloc[0], [0.8, 0.2])
    assert np.allclose(probabilities.iloc[3], [2 / 11, 9 / 11])
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_probabilities_agree_with_the_summed_mass():
    """The full breakdown must sum to what the mass function gives."""
    db = _db(_ROWS)
    probabilities = compute_signature_probabilities(
        db, _ASSIGNMENTS, _SIG_MATRIX
    )
    mass = compute_signature_probability_mass(
        db, _ASSIGNMENTS, _SIG_MATRIX, target_signatures=["S2"]
    )
    assert np.allclose(probabilities["S2"].values, mass)


def test_cohort_effect_shares_match_hand_computation():
    result = _shares()

    # T1: both mutations in G1 (gamma 10), so
    # 10*(0.8, 0.2) + 10*(0.4, 0.6) = (12, 8) -> (0.6, 0.4).
    # T2: 10*(4/7, 3/7) + 1*(2/11, 9/11) = (5.896104, 5.103896),
    # summing to 11 -> (0.536009, 0.463991).
    assert np.allclose(result["by_sample"].loc["T1"], [0.6, 0.4])
    assert np.allclose(
        result["by_sample"].loc["T2"], [0.5360095, 0.4639905]
    )
    assert np.allclose(
        result["average_effect_shares"], [0.5680048, 0.4319952]
    )

    # S1 makes 37.5% of the cohort's mutations but 56.8% of its
    # effect -- the comparison the quantity exists to make.
    assert np.allclose(
        result["average_source_shares"], [0.375, 0.625]
    )
    assert result["n_samples"] == 2
    assert result["n_units"] == 2
    assert result["n_mutations"] == 4


def test_a_hypermutator_counts_once():
    """Per-tumor normalization before averaging is the point."""
    rows = (
        list(_ROWS)
        + [("T3", "G1", "A[C>A]A", "Missense_Mutation")] * 100
    )
    assignments = pd.concat(
        [
            _ASSIGNMENTS,
            pd.DataFrame({"S1": [40.0], "S2": [60.0]}, index=["T3"]),
        ]
    )

    result = _shares(db=_db(rows), assignments=assignments)

    # T3's 100 mutations are all the same gene and type, so its own
    # share is that mutation's attribution -- and the cohort average
    # is the plain mean of three tumors, not of 104 mutations.
    assert result["n_samples"] == 3
    assert np.allclose(
        result["average_effect_shares"],
        result["by_sample"].mean(axis=0),
    )
    assert result["by_sample"].loc["T3"].max() < 1.0


def test_units_without_a_gamma_are_dropped_not_zeroed():
    result = _shares(gammas={"G1": 10.0})

    assert result["n_units"] == 1
    assert result["n_mutations"] == 3
    # T2's only fitted mutation is the G1 one, so its shares are
    # that mutation's attribution alone -- G2 contributes nothing
    # rather than pulling the tumor toward zero effect.
    assert np.allclose(result["by_sample"].loc["T2"], [4 / 7, 3 / 7])


def test_source_shares_cover_the_same_tumors():
    """A tumor with no fitted unit is in neither average."""
    rows = list(_ROWS) + [
        ("T3", "G3", "A[C>T]A", "Missense_Mutation"),
        ("T3", "G3", "A[C>T]A", "Missense_Mutation"),
    ]
    assignments = pd.concat(
        [
            _ASSIGNMENTS,
            pd.DataFrame({"S1": [2.0], "S2": [0.0]}, index=["T3"]),
        ]
    )

    result = _shares(db=_db(rows), assignments=assignments)

    assert list(result["by_sample"].index) == ["T1", "T2"]
    # T3 is pure S1 and would have pulled the source shares to
    # (0.583, 0.417) had it been included on that side only.
    assert np.allclose(
        result["average_source_shares"], [0.375, 0.625]
    )


def test_no_matching_unit_raises():
    with pytest.raises(ValueError, match="None of the mutations"):
        _shares(gammas={"G9": 1.0})


def test_mismatched_row_counts_raise():
    db = _db(_ROWS)
    probabilities = compute_signature_probabilities(
        db, _ASSIGNMENTS, _SIG_MATRIX
    )
    with pytest.raises(ValueError, match="same rows"):
        compute_signature_effect_shares(
            probabilities,
            db["Tumor_Sample_Barcode"].values[:2],
            db["ensembl_gene_id"].values,
            _GAMMAS,
        )


def test_gamma_draws_give_a_distribution_around_the_estimate():
    n_draws = 32
    result = _shares(
        gamma_draws={
            "G1": np.full(n_draws, 10.0),
            "G2": np.full(n_draws, 1.0),
        }
    )

    draws = result["effect_share_draws"]
    assert draws.shape == (n_draws, 2)
    # Constant draws must reproduce the point estimate exactly.
    assert np.allclose(
        draws.mean(axis=0), result["average_effect_shares"]
    )

    spread = _shares(
        gamma_draws={
            "G1": np.linspace(1.0, 100.0, n_draws),
            "G2": np.full(n_draws, 1.0),
        }
    )["effect_share_draws"]
    assert spread["S1"].std() > 0


def _dataset_for_model(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = _db(
        list(_ROWS) + [("T1", "G1", "A[C>T]A", "Silent")]
    )
    dataset._sig_assignments = pd.DataFrame(
        {"S1": [1.5, 0.5], "S2": [1.5, 1.5]}, index=["T1", "T2"]
    )
    dataset._signature_matrix = _SIG_MATRIX
    return dataset


def test_model_gene_level_ignores_silent_mutations(tmp_path):
    """Gamma is fitted on the non-synonymous channel."""
    model = Model(_dataset_for_model(tmp_path))
    model.gammas = dict(_GAMMAS)

    result = model.signature_effect_shares(level="gene")

    assert result["level"] == "gene"
    assert result["n_mutations"] == 4
    # The silent row is dropped, so this is the hand-computed cohort
    # of test_cohort_effect_shares_match_hand_computation -- but its
    # alphas come from all five mutations, as the burden does.
    assert result["n_samples"] == 2


def test_model_variant_level_uses_variant_keys(tmp_path):
    model = Model(_dataset_for_model(tmp_path))
    model.gammas = {"X p.A1B": 2.0}

    result = model.signature_effect_shares(level="variant")

    assert result["n_units"] == 1
    # Every row carries this variant, silent one included: at the
    # variant level there is no consequence filter.
    assert result["n_mutations"] == 5


def test_model_rejects_keys_from_the_other_level(tmp_path):
    model = Model(_dataset_for_model(tmp_path))
    model.gammas = {"X p.A1B": 2.0}

    with pytest.raises(ValueError, match="level="):
        model.signature_effect_shares(level="gene")
