"""The exposure floor: no type may have rate 0 in any tumor.

A fit on a few mutations can land on signatures that give some type
probability 0 (SBS84 alone emits no T>A), and then a mutation the
tumor carries has rate 0 and its gamma cannot be fitted. The floor
mixes ``kappa`` pseudo-mutations of a signature with no zero types
into every tumor's exposures.
"""

import pandas as pd
import pytest

from sigmutsel.estimate_mus import compute_mu_tau_per_tumor


@pytest.fixture
def setup(tmp_path):
    # SIG_Z emits only T1; SIG5 emits both types.
    sigs = pd.DataFrame(
        {"SIG_Z": [1.0, 0.0], "SIG5": [0.5, 0.5]},
        index=pd.Index(["T1", "T2"], name="MutationType"),
    )
    path = tmp_path / "sigs.tsv"
    sigs.to_csv(path, sep="\t")
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["A"] * 3 + ["B"] * 100,
            "Variant_Classification": ["Missense_Mutation"] * 103,
        }
    )
    # A: three mutations, all fitted to SIG_Z. B: well powered, both.
    assignments = pd.DataFrame(
        {"SIG_Z": [3, 50], "SIG5": [0, 50]}, index=["A", "B"]
    )
    return db, path, assignments


def test_without_floor_a_type_has_rate_zero(setup):
    db, path, assignments = setup
    mus = compute_mu_tau_per_tumor(db, path, assignments)
    assert mus.at["A", "T2"] == 0


def test_floor_gives_every_type_a_rate_and_keeps_the_burden(setup):
    db, path, assignments = setup
    mus = compute_mu_tau_per_tumor(
        db,
        path,
        assignments,
        floor_signature="SIG5",
        floor_pseudocount=1,
    )
    # alpha_A = (3 * SIG_Z + 1 * SIG5) / 4 = (0.75, 0.25).
    assert mus.at["A", "T2"] == pytest.approx(3 * 0.25 * 0.5)
    assert mus.loc["A"].sum() == pytest.approx(3)
    # B moves by 1/101 of its exposures, its burden not at all.
    assert mus.loc["B"].sum() == pytest.approx(100)
    # alpha_B = (50, 51) / 101 over (SIG_Z, SIG5); T2 comes from SIG5.
    assert mus.at["B", "T2"] == pytest.approx(100 * (51 / 101) * 0.5)


def test_zero_types_scope_leaves_complete_spectra_alone(setup):
    db, path, assignments = setup
    plain = compute_mu_tau_per_tumor(db, path, assignments)
    mus = compute_mu_tau_per_tumor(
        db,
        path,
        assignments,
        floor_signature="SIG5",
        floor_pseudocount=1,
        floor_scope="zero_types",
    )
    assert mus.at["A", "T2"] > 0
    pd.testing.assert_series_equal(mus.loc["B"], plain.loc["B"])


def test_floor_signature_with_a_zero_type_is_refused(setup):
    db, path, assignments = setup
    with pytest.raises(ValueError, match="probability 0"):
        compute_mu_tau_per_tumor(
            db,
            path,
            assignments,
            floor_signature="SIG_Z",
            floor_pseudocount=1,
        )


def test_pooled_share_shrinks_toward_the_cohort_spectrum(setup):
    db, path, assignments = setup
    db = db.assign(
        type=["T1"] * 3 + ["T1"] * 60 + ["T2"] * 40, gene="P"
    )
    mus = compute_mu_tau_per_tumor(
        db,
        path,
        assignments,
        floor_signature="SIG5",
        floor_pseudocount=1,
        floor_pooled_share=0.9,
    )
    # Pooled observed spectrum (63, 40) / 103; e = 0.9 pooled + 0.1 SIG5.
    e_t2 = 0.9 * 40 / 103 + 0.1 * 0.5
    # A: unfloored spectrum (1, 0), n = 3 -> T2 share e_t2 / 4.
    assert mus.at["A", "T2"] == pytest.approx(3 * e_t2 / 4)
    assert mus.loc["A"].sum() == pytest.approx(3)
    # B: spectrum (0.75, 0.25), n = 100.
    assert mus.at["B", "T2"] == pytest.approx(
        100 * (100 * 0.25 + e_t2) / 101
    )


def test_pooled_share_is_refused_with_separate_per_sigma(setup):
    db, path, assignments = setup
    with pytest.raises(ValueError, match="separate_per_sigma"):
        compute_mu_tau_per_tumor(
            db,
            path,
            assignments,
            separate_per_sigma=True,
            floor_signature="SIG5",
            floor_pseudocount=1,
            floor_pooled_share=0.5,
        )


def test_pooled_spectrum_leaves_out_census_genes(setup, monkeypatch):
    # A recurrent driver's calls must not shape the pool: thyroid
    # BRAF V600E would otherwise raise its own rate in every tumor.
    import sigmutsel.estimate_mus as em

    monkeypatch.setattr(em, "_census_genes", lambda: {"DRIVER"})
    db, path, assignments = setup
    db = db.assign(
        type=["T1"] * 3 + ["T1"] * 60 + ["T2"] * 40,
        gene=["P"] * 63 + ["DRIVER"] * 40,
    )
    mus = compute_mu_tau_per_tumor(
        db,
        path,
        assignments,
        floor_signature="SIG5",
        floor_pseudocount=1,
        floor_pooled_share=0.9,
    )
    # The pool is the passengers' (63, 0) / 63: T2 only from SIG5.
    e_t2 = 0.1 * 0.5
    assert mus.at["A", "T2"] == pytest.approx(3 * e_t2 / 4)
