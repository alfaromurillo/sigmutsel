"""Spectrum uncertainty in the mu cut: each tumor's hybrid spectrum
drawn from the Dirichlet whose mean it is (``spectrum_draws``)."""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.constants import canonical_types_order
from sigmutsel.spectrum_shrinkage import SpectrumShrinkage

from .test_estimate_rg import _fitted_model_with_variants

KAPPA_2 = 50.0
# Passenger calls per tumor: T1 many, T2 few, T3 none.
N_CALLS = {"T1": 5000, "T2": 5, "T3": 0}
BURDEN = {"T1": 2000.0, "T2": 30.0, "T3": 10.0}


def _hybrid_model(tmp_path):
    """The fitted synthetic model, its mu_taus rebuilt as a genuine
    hybrid spectrum over typed passenger calls added to its table."""
    model = _fitted_model_with_variants(tmp_path)
    rng = np.random.default_rng(7)
    types = np.array(canonical_types_order)
    calls = pd.DataFrame(
        [
            {
                "ensembl_gene_id": "ENSG_C",
                "Tumor_Sample_Barcode": tumor,
                "Variant_Classification": "Missense_Mutation",
                "type": tau,
            }
            for tumor, n in N_CALLS.items()
            for tau in rng.choice(
                types, size=n, p=rng.dirichlet(np.full(96, 2.0))
            )
        ]
    )
    db = model.dataset.mutation_db
    model.dataset._mutation_db = pd.concat(
        [db, calls], ignore_index=True
    )

    counts = (
        pd.crosstab(calls["Tumor_Sample_Barcode"], calls["type"])
        .reindex(
            index=list(N_CALLS),
            columns=canonical_types_order,
            fill_value=0,
        )
        .astype(float)
    )
    q = pd.DataFrame(
        rng.dirichlet(np.full(96, 3.0), size=len(N_CALLS)),
        index=list(N_CALLS),
        columns=canonical_types_order,
    )
    pi = (counts + KAPPA_2 * q).div(
        counts.sum(axis=1) + KAPPA_2, axis=0
    )
    model._mu_taus = pi.mul(pd.Series(BURDEN), axis=0)
    model.compute_channel_base_mus()
    model._baseline_spectrum_shrinkage = SpectrumShrinkage(
        kappa_1=10.0, kappa_2=KAPPA_2
    ).to_dict()
    return model, pi


def _dirichlet_moments(model, pi, weights):
    """Mean and variance of sum_tau w_tau ell_j pi_jtau under each
    tumor's Dirichlet(alpha_j), alpha_j = pi_j (n_j. + kappa_2)."""
    w = weights.reindex(canonical_types_order).to_numpy()
    p = pi.to_numpy()
    mean = p @ w
    concentration = np.array([N_CALLS[t] for t in pi.index]) + KAPPA_2
    var = ((p * w**2).sum(axis=1) - mean**2) / (concentration + 1)
    ell = np.array([BURDEN[t] for t in pi.index])
    return ell * mean, ell**2 * var


def test_draws_follow_the_dirichlet(tmp_path):
    """Mean = the hybrid spectrum's baseline (to Monte Carlo error),
    variance = the Dirichlet's, for a gene's channel weights."""
    model, pi = _hybrid_model(tmp_path)
    weights = model._type_weights(
        model._channel_opportunity("nonsyn"), "ENSG_B"
    )
    baseline = model.base_mus_nonsyn.loc["ENSG_B"]
    mean, var = _dirichlet_moments(model, pi, weights)
    np.testing.assert_allclose(
        mean, baseline.loc[pi.index].to_numpy(), rtol=1e-10
    )

    n = 100_000
    draws = model._spectrum_baseline_draws(
        weights, n, np.random.default_rng(0)
    )[pi.index].to_numpy()
    se = np.sqrt(var / n)
    assert (np.abs(draws.mean(axis=0) - mean) < 5 * se).all()
    np.testing.assert_allclose(draws.var(axis=0), var, rtol=0.03)


def test_many_calls_barely_vary_few_vary_as_the_dirichlet_says(
    tmp_path,
):
    model, pi = _hybrid_model(tmp_path)
    weights = model._type_weights(
        model._channel_opportunity("nonsyn"), "ENSG_B"
    )
    draws = model._spectrum_baseline_draws(
        weights, 50_000, np.random.default_rng(1)
    )
    cv = draws.std(axis=0) / draws.mean(axis=0)
    # 5,000 calls: under 1%; 5 calls: kappa_2 + 5 pseudo-counts.
    assert cv["T1"] < 0.01
    assert cv["T2"] > 5 * cv["T1"]
    mean, var = _dirichlet_moments(model, pi, weights)
    expected_cv = np.sqrt(var) / mean
    np.testing.assert_allclose(
        cv[pi.index].to_numpy(), expected_cv, rtol=0.03
    )


def test_gene_draws_are_paired_and_centred(tmp_path):
    """On the full path the spectrum factor multiplies each covariate
    draw: on / off is the spectrum draw over the baseline, mean 1."""
    model, _ = _hybrid_model(tmp_path)
    off = model.compute_mu_g_posterior_draws(
        "ENSG_B", r_g_variant="none"
    )
    on = model.compute_mu_g_posterior_draws(
        "ENSG_B",
        r_g_variant="none",
        rng=np.random.default_rng(2),
        spectrum_draws=True,
    )
    assert on.shape == off.shape
    assert list(on.columns) == list(off.columns)
    ratio = (on / off).to_numpy()
    se = ratio.std(axis=0) / np.sqrt(ratio.shape[0])
    assert (np.abs(ratio.mean(axis=0) - 1) < 5 * se + 1e-12).all()
    assert ratio[:, list(off.columns).index("T2")].std() > 0


def test_variant_draws_are_centred_on_the_point_rate(tmp_path):
    """A multi-type variant: its route types are drawn jointly, and
    the draws average to the option-off draws."""
    model, _ = _hybrid_model(tmp_path)
    off = model.compute_mu_m_posterior_draws(
        "VAR2_multitype", r_g_variant="none"
    )
    on = model.compute_mu_m_posterior_draws(
        "VAR2_multitype",
        r_g_variant="none",
        rng=np.random.default_rng(3),
        spectrum_draws=True,
    )
    ratio = (on / off).to_numpy()
    se = ratio.std(axis=0) / np.sqrt(ratio.shape[0])
    assert (np.abs(ratio.mean(axis=0) - 1) < 5 * se + 1e-12).all()
    t1, t2 = (list(off.columns).index(t) for t in ("T1", "T2"))
    assert ratio[:, t2].std() > 5 * ratio[:, t1].std()


def test_option_off_is_unchanged_bit_for_bit(tmp_path):
    """Off (the default) is exactly the covariate draws times the
    point baseline, as before the option existed."""
    model, _ = _hybrid_model(tmp_path)
    default = model.compute_mu_g_posterior_draws(
        "ENSG_B", r_g_variant="none"
    )
    off = model.compute_mu_g_posterior_draws(
        "ENSG_B", r_g_variant="none", spectrum_draws=False
    )
    scale, _ = model._covariate_scale_draws("ENSG_B")
    baseline = model.base_mus_nonsyn.loc["ENSG_B"]
    assert np.array_equal(default.to_numpy(), off.to_numpy())
    assert np.array_equal(
        off.to_numpy(), scale[:, None] * baseline.to_numpy()[None, :]
    )
    variant_default = model.compute_mu_m_posterior_draws(
        "VAR1", r_g_variant="none"
    )
    variant_off = model.compute_mu_m_posterior_draws(
        "VAR1", r_g_variant="none", spectrum_draws=False
    )
    assert np.array_equal(
        variant_default.to_numpy(), variant_off.to_numpy()
    )


def test_estimate_gamma_passes_the_option(tmp_path, monkeypatch):
    model, _ = _hybrid_model(tmp_path)
    seen = {}
    original = model.compute_mu_g_posterior_draws

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "compute_mu_g_posterior_draws", spy)
    import sigmutsel.estimate_gammas as estimate_gammas_mod

    monkeypatch.setattr(
        estimate_gammas_mod,
        "estimate_gamma_from_mus",
        lambda mus_yes, mus_no, **kwargs: "fake_result",
    )
    model.estimate_gamma(
        "ENSG_B",
        level="gene",
        store=False,
        use_mu_posterior=True,
        spectrum_draws=True,
        hold_out_multi_base=False,
        hold_out_homdel=False,
        hold_out_indel=False,
    )
    assert seen["spectrum_draws"] is True


def test_needs_the_hybrid_spectrum(tmp_path):
    model, _ = _hybrid_model(tmp_path)
    model._baseline_spectrum_shrinkage = SpectrumShrinkage(
        kappa_1=10.0
    ).to_dict()
    with pytest.raises(ValueError, match="kappa_2"):
        model.compute_mu_g_posterior_draws(
            "ENSG_B", r_g_variant="none", spectrum_draws=True
        )


def test_refuses_calls_changed_after_the_baselines(tmp_path):
    """More passenger calls of a type than the hybrid's pseudo-counts
    allow: the dataset no longer matches mu_taus."""
    model, _ = _hybrid_model(tmp_path)
    db = model.dataset.mutation_db
    extra = db[db["Tumor_Sample_Barcode"] == "T2"].dropna(
        subset=["type"]
    )
    model.dataset._mutation_db = pd.concat(
        [db] + [extra] * 50, ignore_index=True
    )
    with pytest.raises(ValueError, match="calls changed"):
        model.compute_mu_g_posterior_draws(
            "ENSG_B", r_g_variant="none", spectrum_draws=True
        )
