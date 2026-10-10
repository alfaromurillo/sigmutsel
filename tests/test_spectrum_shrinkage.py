"""Spectrum shrinkage: the floor and the hybrid.

    q_jt  = (n_j p_jt + kappa_1 e_t) / (n_j + kappa_1)
    pi_jt = (n_jt + kappa_2 q_jt) / (n_j. + kappa_2)

with ``e`` the cohort's passenger pool plus a share of a floor
signature, and ``n_jt`` the tumor's passenger calls only.
"""

import numpy as np
import pandas as pd
import pytest

import sigmutsel.spectrum_shrinkage as ss
from sigmutsel.estimate_mus import compute_mu_tau_per_tumor
from sigmutsel.models import MutationDataset
from sigmutsel.spectrum_shrinkage import (
    SpectrumShrinkage,
    floor_prior,
    floored_spectra,
    hybrid_spectra,
    select_kappas,
)

TYPES = pd.Index(["T1", "T2"], name="MutationType")


@pytest.fixture
def no_drivers(monkeypatch):
    """Only the gene DRIVER is a non-passenger."""
    import sigmutsel.estimate_presence as ep

    monkeypatch.setattr(
        ep, "non_passenger_symbols", lambda: {"DRIVER"}
    )
    monkeypatch.setattr(
        ep, "non_passenger_ensembl_ids", lambda **_: {"ENSG_DRIVER"}
    )


@pytest.fixture
def setup(tmp_path, no_drivers):
    # SIG_Z emits only T1; SIG5 emits both types.
    sigs = pd.DataFrame(
        {"SIG_Z": [1.0, 0.0], "SIG5": [0.5, 0.5]}, index=TYPES
    )
    path = tmp_path / "sigs.tsv"
    sigs.to_csv(path, sep="\t")
    # A: 3 calls, all T1, fitted to SIG_Z. B: 100 calls, 60 T1 and 40
    # T2, of which 20 T2 are in the driver gene.
    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["A"] * 3 + ["B"] * 100,
            "Variant_Classification": ["Missense_Mutation"] * 103,
            "type": ["T1"] * 3 + ["T1"] * 60 + ["T2"] * 40,
            "gene": ["P"] * 83 + ["DRIVER"] * 20,
        }
    )
    assignments = pd.DataFrame(
        {"SIG_Z": [3, 50], "SIG5": [0, 50]}, index=["A", "B"]
    )
    return db, path, assignments


def test_floor_formula():
    p = np.array([[1.0, 0.0], [0.75, 0.25]])
    e = np.array([0.4, 0.6])
    q = floored_spectra(p, np.array([3.0, 100.0]), e, 2.0)
    np.testing.assert_allclose(q[0], (3 * p[0] + 2 * e) / 5)
    np.testing.assert_allclose(q[1], (100 * p[1] + 2 * e) / 102)
    np.testing.assert_allclose(q.sum(axis=1), 1)


def test_hybrid_formula():
    c = np.array([[3.0, 0.0], [10.0, 30.0]])
    q = np.array([[0.5, 0.5], [0.9, 0.1]])
    pi = hybrid_spectra(c, q, 4.0)
    np.testing.assert_allclose(pi[0], (c[0] + 4 * q[0]) / 7)
    np.testing.assert_allclose(pi[1], (c[1] + 4 * q[1]) / 44)
    np.testing.assert_allclose(pi.sum(axis=1), 1)


def test_kappa_limits():
    p = np.array([[1.0, 0.0]])
    e = np.array([0.4, 0.6])
    c = np.array([[2.0, 6.0]])
    q = floored_spectra(p, np.array([3.0]), e, 2.0)
    # kappa_2 -> inf: the hybrid is the floored spectrum.
    np.testing.assert_allclose(
        hybrid_spectra(c, q, 1e12), q, atol=1e-9
    )
    # kappa_2 None: the floor alone, exactly.
    np.testing.assert_array_equal(hybrid_spectra(c, q, None), q)
    # kappa_2 -> 0: the tumor's own observed spectrum.
    np.testing.assert_allclose(
        hybrid_spectra(c, q, 1e-12), c / c.sum(), atol=1e-9
    )
    # kappa_1 -> inf: every tumor gets the prior.
    np.testing.assert_allclose(
        floored_spectra(p, np.array([3.0]), e, 1e12),
        e[None, :],
        atol=1e-9,
    )
    # kappa_1 -> 0: the signature spectrum.
    np.testing.assert_allclose(
        floored_spectra(p, np.array([3.0]), e, 1e-12), p, atol=1e-9
    )


def test_prior_mixes_pool_and_floor_signature():
    e = floor_prior(np.array([30.0, 10.0]), np.array([1.0, 3.0]), 0.1)
    np.testing.assert_allclose(
        e, 0.9 * np.array([0.75, 0.25]) + 0.1 * np.array([0.25, 0.75])
    )
    # No calls: the floor signature alone.
    np.testing.assert_allclose(
        floor_prior(np.zeros(2), np.array([1.0, 3.0]), 0.1),
        [0.25, 0.75],
    )
    with pytest.raises(ValueError, match="probability 0"):
        floor_prior(np.ones(2), np.array([1.0, 0.0]), 0.1)


def test_parameters_are_validated_and_round_trip():
    with pytest.raises(ValueError):
        SpectrumShrinkage(kappa_1=0)
    with pytest.raises(ValueError):
        SpectrumShrinkage(kappa_1=10, floor_share=0)
    s = SpectrumShrinkage(kappa_1=30, kappa_2=500, selection={"a": 1})
    assert SpectrumShrinkage.from_dict(s.to_dict()) == s
    assert SpectrumShrinkage.from_dict(None) is None


def test_mu_taus_shrunk_with_passenger_counts_only(setup):
    db, path, assignments = setup
    params = SpectrumShrinkage(
        kappa_1=1, kappa_2=2, floor_signature="SIG5"
    )
    mus = compute_mu_tau_per_tumor(
        db, path, assignments, spectrum_shrinkage=params
    )
    # Passenger calls: A (3, 0), B (60, 20); pool (63, 20) / 83.
    e = 0.9 * np.array([63, 20]) / 83 + 0.1 * np.array([0.5, 0.5])
    q_a = (3 * np.array([1.0, 0.0]) + 1 * e) / 4
    pi_a = (np.array([3.0, 0.0]) + 2 * q_a) / 5
    np.testing.assert_allclose(mus.loc["A"], 3 * pi_a)
    q_b = (100 * np.array([0.75, 0.25]) + e) / 101
    pi_b = (np.array([60.0, 20.0]) + 2 * q_b) / 82
    np.testing.assert_allclose(mus.loc["B"], 100 * pi_b)
    # The burden is untouched, and every type has a rate.
    np.testing.assert_allclose(mus.sum(axis=1), [3, 100])
    assert (mus.to_numpy() > 0).all()


def test_driver_calls_never_reach_the_rate(setup):
    # Doubling the driver's calls must not move any tumor's spectrum.
    db, path, assignments = setup
    extra = db[db["gene"] == "DRIVER"]
    params = SpectrumShrinkage(
        kappa_1=1, kappa_2=2, floor_signature="SIG5"
    )
    a = compute_mu_tau_per_tumor(
        db, path, assignments, spectrum_shrinkage=params
    )
    b = compute_mu_tau_per_tumor(
        pd.concat([db, extra], ignore_index=True),
        path,
        assignments,
        spectrum_shrinkage=params,
    )
    # B's burden grows by its 20 extra calls; its spectrum does not move.
    np.testing.assert_allclose(a.loc["A"], b.loc["A"])
    np.testing.assert_allclose(
        a.loc["B"] / a.loc["B"].sum(), b.loc["B"] / b.loc["B"].sum()
    )


def test_ensembl_ids_are_matched_without_version(setup):
    db = setup[0]
    db = db.assign(
        ensembl_gene_id=np.where(
            db["gene"] == "DRIVER", "ENSG_DRIVER.7", "ENSG_P.1"
        ),
        gene="whatever",
    )
    calls = ss.passenger_calls(db)
    assert len(calls) == 83


def test_floor_signature_must_be_in_the_matrix(setup):
    db, path, assignments = setup
    with pytest.raises(
        ValueError, match="not in the signature matrix"
    ):
        compute_mu_tau_per_tumor(
            db,
            path,
            assignments,
            spectrum_shrinkage=SpectrumShrinkage(kappa_1=1),
        )


def test_refused_with_separate_per_sigma(setup):
    db, path, assignments = setup
    with pytest.raises(ValueError, match="separate_per_sigma"):
        compute_mu_tau_per_tumor(
            db,
            path,
            assignments,
            separate_per_sigma=True,
            spectrum_shrinkage=SpectrumShrinkage(
                kappa_1=1, floor_signature="SIG5"
            ),
        )


def test_dataset_saves_and_loads_the_kappas(tmp_path):
    ds = MutationDataset(location_maf_files=str(tmp_path / "mafs"))
    assert ds.spectrum_shrinkage is None
    params = SpectrumShrinkage(
        kappa_1=30, kappa_2=500, selection={"gain_nested": 1.5}
    )
    ds.set_spectrum_shrinkage(params)
    ds.save_dataset(tmp_path / "ds", overwrite=True)
    back = MutationDataset.load_dataset(tmp_path / "ds")
    assert back.spectrum_shrinkage == params
    assert back.spectrum_shrinkage.selection == {"gain_nested": 1.5}
    back.set_spectrum_shrinkage(None)
    assert back.spectrum_shrinkage is None


def _fake_fit(matrix, signatures, workdir):
    """Exposures proportional to each tumor's T1 and T2 calls."""
    t = matrix.T
    return pd.DataFrame(
        {"SIG_Z": t["T1"] - t["T2"], "SIG5": 2 * t["T2"]},
        index=t.index,
    ).clip(lower=0)


def test_heldout_scores_and_selection(no_drivers):
    rng = np.random.default_rng(1)
    # 30 tumors with their own fixed spectra; the hybrid should prefer
    # leaning on each tumor's counts.
    rows = []
    for j in range(30):
        share = rng.uniform(0.05, 0.95)
        n = int(rng.integers(20, 200))
        types = np.where(rng.random(n) < share, "T1", "T2")
        rows += [(f"S{j:02d}", t) for t in types]
    db = pd.DataFrame(rows, columns=["Tumor_Sample_Barcode", "type"])
    db = db.assign(
        Variant_Classification="Missense_Mutation", gene="P"
    )
    sigs = pd.DataFrame(
        {"SIG_Z": [1.0, 0.0], "SIG5": [0.5, 0.5]}, index=TYPES
    )
    scores = ss.heldout_spectrum_scores(
        db,
        sigs,
        kappa_1_grid=(10.0, 100.0),
        kappa_2_grid=(1.0, 1000.0),
        floor_signature="SIG5",
        folds=3,
        fit=_fake_fit,
    )
    assert len(scores) == 30
    assert {"10|floor", "100|floor", "10|1", "100|1000"} <= set(
        scores.columns
    )
    assert scores["burden"].sum() == len(db)
    chosen = select_kappas(
        scores, outer_folds=3, floor_signature="SIG5"
    )
    sel = chosen.selection
    assert chosen.kappa_2 is not None
    assert sel["optimism"] == pytest.approx(
        sel["gain_in_sample"] - sel["gain_nested"]
    )
    assert sel["gain_in_sample"] >= sel["gain_nested"] - 1e-9
    assert len(sel["outer_fold_choices"]) == 3
    floor_only = select_kappas(scores, family="floor", outer_folds=3)
    assert floor_only.kappa_2 is None
