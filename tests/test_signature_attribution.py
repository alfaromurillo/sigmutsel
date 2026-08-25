"""Tests for signature_attribution.py's per-mutation probability mass.

Uses a small synthetic 2-signature, 2-mutation-type setup where each
signature deterministically produces only one mutation type, so the
expected per-mutation probabilities are exact (0.0 or 1.0), not just
plausible ranges.
"""

import pandas as pd
import pytest

from sigmutsel.signature_attribution import (
    compute_signature_probability_mass,
)


@pytest.fixture
def sig_matrix_path(tmp_path):
    # SIG_A produces only T1; SIG_ARTIFACT produces only T2.
    df = pd.DataFrame(
        {
            "MutationType": ["T1", "T2"],
            "SIG_A": [1.0, 0.0],
            "SIG_ARTIFACT": [0.0, 1.0],
        }
    )
    path = tmp_path / "sig_matrix.txt"
    df.to_csv(path, sep="\t", index=False)
    return path


def test_artifact_mass_matches_deterministic_type_assignment(
    sig_matrix_path,
):
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S1", "S1"],
            "type": ["T1", "T1", "T2", "T2"],
            "Variant_Classification": ["Missense_Mutation"] * 4,
        }
    )
    # Raw per-signature mutation counts for S1: 8 from SIG_A, 2 from
    # SIG_ARTIFACT -- alphas become 0.8 / 0.2 after normalization.
    assignments = pd.DataFrame(
        {"SIG_A": [8], "SIG_ARTIFACT": [2]}, index=["S1"]
    )

    mass = compute_signature_probability_mass(
        db,
        assignments,
        sig_matrix_path,
        target_signatures=["SIG_ARTIFACT"],
    )

    assert mass.shape == (4,)
    # T1 mutations can only come from SIG_A -- artifact mass exactly 0.
    assert mass[0] == pytest.approx(0.0)
    assert mass[1] == pytest.approx(0.0)
    # T2 mutations can only come from SIG_ARTIFACT -- artifact mass
    # exactly 1, regardless of SIG_A's much larger overall share.
    assert mass[2] == pytest.approx(1.0)
    assert mass[3] == pytest.approx(1.0)


def test_zero_burden_sample_gives_zero_not_nan(sig_matrix_path):
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S2"],
            "type": ["T1"],
            "Variant_Classification": ["Missense_Mutation"],
        }
    )
    assignments = pd.DataFrame(
        {"SIG_A": [0], "SIG_ARTIFACT": [0]}, index=["S2"]
    )

    mass = compute_signature_probability_mass(
        db,
        assignments,
        sig_matrix_path,
        target_signatures=["SIG_ARTIFACT"],
    )

    assert mass[0] == 0.0


def test_target_signature_absent_from_matrix_is_ignored(
    sig_matrix_path,
):
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1"],
            "type": ["T2"],
            "Variant_Classification": ["Missense_Mutation"],
        }
    )
    assignments = pd.DataFrame(
        {"SIG_A": [8], "SIG_ARTIFACT": [2]}, index=["S1"]
    )

    # "SIG_NONEXISTENT" isn't in the signature matrix at all -- should
    # be silently dropped from the target set rather than erroring.
    mass = compute_signature_probability_mass(
        db,
        assignments,
        sig_matrix_path,
        target_signatures=["SIG_NONEXISTENT"],
    )
    assert mass[0] == pytest.approx(0.0)
