"""Tests for the shared per-gene rate correction ``r_g``.

Two things have to be right here, and both fail silently if they are
not:

1. The **marginalization**. ``r_g`` is integrated out analytically, so
   the closed form has to actually equal the integral -- checked
   against numerical quadrature rather than re-derived.
2. The **production/evaluation separation**. The evaluation ``r_g``
   must not be able to see non-silent data, which is enforced
   structurally (no argument for it) and checked here.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson

from sigmutsel.estimate_rg import (
    channel_rg_log_likelihood,
    estimate_channel_rg_effect,
    r_g_production,
    r_g_silent_only_for_evaluation,
)
from sigmutsel.models import Model, MutationDataset

from .test_channel_cov_effects import (  # noqa: F401
    _GENES,
    _model_with_channels,
    _mutation_db,
    _synthetic_mu_taus,
    _synthetic_opportunities,
)


def _brute_force_log_marginal(counts, expected, theta):
    """log ∫ ∏_j Poisson(N_j; m_j r) Gamma(r; θ, 1/θ) dr, numerically."""
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(expected, dtype=float)

    def integrand(r):
        return np.prod(
            poisson.pmf(counts, expected * r)
        ) * gamma_dist.pdf(r, a=theta, scale=1 / theta)

    value, _ = quad(integrand, 0, np.inf, limit=400)
    return np.log(value)


@pytest.mark.parametrize("theta", [0.5, 3.0, 25.0])
def test_marginalization_matches_numerical_integration(theta):
    """The closed form must equal the integral it claims to be.

    The per-observation ``Σ_j [N log μ̄ - log N!]`` term is dropped by
    the implementation as a constant, so it is added back here before
    comparing.
    """
    from scipy.special import gammaln

    rng = np.random.default_rng(7)
    for _ in range(4):
        baseline = rng.uniform(0.05, 2.0, size=5)
        counts = rng.poisson(baseline * 1.3)
        eta = float(rng.uniform(-0.4, 0.4))
        expected = baseline * np.exp(eta)

        # implementation, as a single gene with no non-silent channel
        got = channel_rg_log_likelihood(
            eta_silent=np.array([eta]),
            theta=theta,
            counts_silent=np.array([counts.sum()], dtype=float),
            counts_non_silent=np.array([0.0]),
            baseline_silent=np.array([baseline.sum()]),
            baseline_non_silent=np.array([0.0]),
        )
        dropped = np.sum(counts * np.log(baseline)) - np.sum(
            gammaln(counts + 1)
        )
        want = _brute_force_log_marginal(counts, expected, theta)
        assert np.isclose(got + dropped, want, atol=1e-9)


def test_marginalization_shares_r_g_across_channels():
    """A gene's two channels must be integrated against **one** r_g.

    Treating them as separate integrals would give a different (and
    wrong) answer -- this pins the sharing that the whole mechanism
    rests on.
    """
    theta = 4.0
    eta = np.array([0.1])
    shared = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=theta,
        counts_silent=np.array([3.0]),
        counts_non_silent=np.array([5.0]),
        baseline_silent=np.array([1.2]),
        baseline_non_silent=np.array([4.0]),
    )
    separate = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=theta,
        counts_silent=np.array([3.0]),
        counts_non_silent=np.array([0.0]),
        baseline_silent=np.array([1.2]),
        baseline_non_silent=np.array([0.0]),
    ) + channel_rg_log_likelihood(
        eta_silent=eta,
        theta=theta,
        counts_silent=np.array([5.0]),
        counts_non_silent=np.array([0.0]),
        baseline_silent=np.array([4.0]),
        baseline_non_silent=np.array([0.0]),
    )
    assert not np.isclose(shared, separate)


def test_large_theta_approaches_plain_poisson():
    """θ → ∞ pins r_g at 1, so the likelihood must approach the
    plain two-channel Poisson one (up to the dropped constant)."""
    counts = np.array([4.0, 1.0])
    baseline = np.array([2.0, 3.0])
    eta = np.array([0.05, -0.2])

    expected = baseline * np.exp(eta)
    plain = np.sum(counts * eta - expected)

    previous = None
    for theta in (1e3, 1e5, 1e7):
        got = channel_rg_log_likelihood(
            eta_silent=eta,
            theta=theta,
            counts_silent=counts,
            counts_non_silent=np.zeros_like(counts),
            baseline_silent=baseline,
            baseline_non_silent=np.zeros_like(baseline),
        )
        gap = abs(got - plain)
        if previous is not None:
            assert gap < previous
        previous = gap
    assert previous < 1e-3


def test_r_g_variants_differ_and_are_shrunk_to_one():
    counts_silent = pd.Series([0.0, 5.0, 2.0], index=_GENES)
    counts_non_silent = pd.Series([1.0, 20.0, 0.0], index=_GENES)
    expected_silent = pd.Series([1.0, 2.0, 2.0], index=_GENES)
    expected_non_silent = pd.Series([4.0, 8.0, 8.0], index=_GENES)
    theta = 5.0

    production = r_g_production(
        counts_silent,
        counts_non_silent,
        expected_silent,
        expected_non_silent,
        theta,
    )
    evaluation = r_g_silent_only_for_evaluation(
        counts_silent, expected_silent, theta
    )

    # the gene with a big non-silent excess is pulled up only in the
    # production variant -- that is the leakage the evaluation one
    # exists to avoid
    assert production["ENSG_B"] > evaluation["ENSG_B"]
    # everything is shrunk toward the prior mean of 1
    assert (production > 0).all() and (evaluation > 0).all()
    assert abs(evaluation["ENSG_A"] - 1.0) < abs(
        0.0 / 1.0 - 1.0
    )  # 0 observed does not send r_g to 0


def test_evaluation_r_g_cannot_receive_non_silent_data():
    """The separation is structural: there is no argument for it."""
    import inspect

    params = set(
        inspect.signature(r_g_silent_only_for_evaluation).parameters
    )
    assert params == {"counts_silent", "expected_silent", "theta"}
    assert not any("non_silent" in p for p in params)


def test_theta_shrinks_more_when_small():
    """Smaller θ means weaker shrinkage toward 1 (more trust in the
    gene's own count) -- the empirical-Bayes knob behaving."""
    counts = pd.Series([10.0], index=["g"])
    expected = pd.Series([2.0], index=["g"])
    weak = r_g_silent_only_for_evaluation(counts, expected, 0.5)["g"]
    strong = r_g_silent_only_for_evaluation(counts, expected, 50.0)[
        "g"
    ]
    assert weak > strong > 1.0


def _rg_model(tmp_path, **kwargs):
    model = _model_with_channels(tmp_path, **kwargs)
    model.dataset.compute_gene_counts_channels()
    return model


def test_estimate_channel_rg_cov_effects_map(tmp_path):
    model = _rg_model(tmp_path)
    result = model.estimate_channel_rg_cov_effects(sample="MAP")
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))
    assert np.isfinite(model.rg_theta) and model.rg_theta > 0

    production = model.compute_r_g_production()
    evaluation = model.compute_r_g_for_evaluation()
    assert len(production) == len(evaluation)
    assert (production > 0).all() and (evaluation > 0).all()
    assert not np.allclose(production.values, evaluation.values)


def test_rg_requires_counts(tmp_path):
    model = _model_with_channels(tmp_path)
    with pytest.raises(ValueError, match="Channel count matrices"):
        model.estimate_channel_rg_cov_effects(sample="MAP")


def test_rg_theta_raises_before_fitting(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    model = Model(dataset)
    with pytest.raises(ValueError, match="theta not fitted"):
        _ = model.rg_theta


def test_r_g_before_fitting_raises(tmp_path):
    model = _rg_model(tmp_path)
    with pytest.raises(ValueError, match="No r_g fit available"):
        model.compute_r_g_production()


def test_drivers_toggle_changes_gene_set(tmp_path):
    """include_drivers=False restricts the silent channel to the
    passenger set, so fewer genes get an r_g."""
    on = _rg_model(tmp_path)
    on.estimate_channel_rg_cov_effects(
        sample="MAP", include_drivers=True
    )
    off = _rg_model(tmp_path)
    off.estimate_channel_rg_cov_effects(
        sample="MAP", include_drivers=False
    )
    assert len(on.compute_r_g_production()) >= len(
        off.compute_r_g_production()
    )


def test_rg_statistics_align_with_counts(tmp_path):
    """The per-gene sufficient statistics must be the actual row sums
    of the count matrices -- everything downstream trusts them."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    stats = model._rg_statistics

    expected_silent = model.dataset.genes_counts_silent.reindex(
        index=stats["genes"],
        columns=model.base_mus_syn.columns,
        fill_value=0,
    ).sum(axis=1)
    pd.testing.assert_series_equal(
        stats["counts_silent"], expected_silent, check_dtype=False
    )


def test_rg_gene_array_mismatch_raises():
    with pytest.raises(ValueError, match="must be aligned"):
        estimate_channel_rg_effect(
            counts_silent=np.zeros(3),
            baseline_silent=np.ones(2),
            counts_non_silent=np.zeros(3),
            baseline_non_silent=np.zeros(3),
            cov_matrix=np.zeros((3, 1)),
            draws=1,
        )


def test_gene_scaling_applies_r_g_and_is_not_stored(tmp_path):
    """A gene_scaling'd R² is returned but never stored -- it depends
    on which r_g the caller chose, so it must not be readable as the
    model's own number."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")

    plain = model.estimate_passenger_genes_r2(
        target="non_silent_counts"
    )
    stored = model.passenger_genes_r2_non_silent_counts
    scaled = model.estimate_passenger_genes_r2(
        target="non_silent_counts",
        gene_scaling=model.compute_r_g_for_evaluation(),
    )
    assert scaled != plain
    # the stored attribute still holds the unscaled number
    assert model.passenger_genes_r2_non_silent_counts == stored


def test_gene_scaling_of_ones_is_a_no_op(tmp_path):
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    ones = pd.Series(
        1.0, index=model.compute_r_g_for_evaluation().index
    )
    assert model.estimate_passenger_genes_r2(
        target="non_silent_counts"
    ) == model.estimate_passenger_genes_r2(
        target="non_silent_counts", gene_scaling=ones
    )


# ---------------------------------------------------------------
# Stage 5: separate c per channel
# ---------------------------------------------------------------


def test_separate_c_with_equal_vectors_equals_shared():
    """The nesting that makes the LR comparison valid: separate-c
    with c^(syn) == c^(nonsyn) must give exactly the shared-c
    likelihood, not merely a close one."""
    rng = np.random.default_rng(3)
    n = 6
    eta = rng.normal(0, 0.3, size=n)
    kwargs = {
        "theta": 4.0,
        "counts_silent": rng.poisson(2.0, size=n).astype(float),
        "counts_non_silent": rng.poisson(5.0, size=n).astype(float),
        "baseline_silent": rng.uniform(0.5, 3.0, size=n),
        "baseline_non_silent": rng.uniform(1.0, 6.0, size=n),
    }
    shared = channel_rg_log_likelihood(eta_silent=eta, **kwargs)
    separate = channel_rg_log_likelihood(
        eta_silent=eta, eta_non_silent=eta.copy(), **kwargs
    )
    assert separate == pytest.approx(shared, rel=0, abs=1e-12)


def test_separate_c_actually_differs_when_etas_differ():
    rng = np.random.default_rng(4)
    n = 6
    kwargs = {
        "theta": 4.0,
        "counts_silent": rng.poisson(2.0, size=n).astype(float),
        "counts_non_silent": rng.poisson(5.0, size=n).astype(float),
        "baseline_silent": rng.uniform(0.5, 3.0, size=n),
        "baseline_non_silent": rng.uniform(1.0, 6.0, size=n),
    }
    eta = rng.normal(0, 0.3, size=n)
    shared = channel_rg_log_likelihood(eta_silent=eta, **kwargs)
    separate = channel_rg_log_likelihood(
        eta_silent=eta, eta_non_silent=eta + 0.25, **kwargs
    )
    assert separate != shared


def test_separate_c_fit_shape_and_storage(tmp_path):
    """separate_c lands in channel_cov_effects, never in
    cov_effects -- nothing downstream may read a (2, n) array as
    the shared vector."""
    model = _rg_model(tmp_path)
    result = model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=True
    )
    assert result.shape == (2, 2)
    assert model.channel_cov_effects.shape == (2, 2)
    assert model.cov_effects is None
    assert np.isfinite(model.rg_theta)


def test_separate_c_beats_or_matches_shared_in_likelihood(tmp_path):
    """The separate fit maximises over a superset of the shared
    fit's parameter space, so its log-likelihood cannot be lower
    (up to optimiser tolerance)."""
    shared = _rg_model(tmp_path)
    shared.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )
    ll_shared = shared.channel_rg_log_likelihood_at_fit()

    separate = _rg_model(tmp_path)
    separate.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=True
    )
    ll_separate = separate.channel_rg_log_likelihood_at_fit()

    assert np.isfinite(ll_shared) and np.isfinite(ll_separate)
    assert ll_separate >= ll_shared - 1e-6


def test_channel_cov_effects_raises_without_separate_fit(tmp_path):
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )
    with pytest.raises(ValueError, match="No separate-c fit"):
        _ = model.channel_cov_effects


def test_log_likelihood_at_fit_requires_a_fit(tmp_path):
    model = _rg_model(tmp_path)
    with pytest.raises(ValueError, match="No r_g fit available"):
        model.channel_rg_log_likelihood_at_fit()


def test_intercept_mode_is_between_shared_and_separate(tmp_path):
    """The three c parameterisations are nested, so their maximised
    log-likelihoods must be ordered shared <= intercept <= separate.
    Testing separate against shared alone would conflate the
    calibration offset with the slopes."""
    lls = {}
    for mode in (False, "intercept", True):
        model = _rg_model(tmp_path)
        model.estimate_channel_rg_cov_effects(
            sample="MAP", separate_c=mode
        )
        lls[str(mode)] = model.channel_rg_log_likelihood_at_fit()

    assert lls["False"] <= lls["intercept"] + 1e-6
    assert lls["intercept"] <= lls["True"] + 1e-6


def test_intercept_mode_keeps_shared_c_shape(tmp_path):
    """`intercept` shares the slopes, so `c` stays 1-D and the
    downstream mu_gs recomputation is still meaningful."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c="intercept"
    )
    assert np.asarray(model.cov_effects).ndim == 1
    assert model._rg_delta_intercept is not None
    assert np.isfinite(model._rg_delta_intercept)


def test_unknown_separate_c_raises(tmp_path):
    model = _rg_model(tmp_path)
    with pytest.raises(ValueError, match="separate_c must be"):
        model.estimate_channel_rg_cov_effects(
            sample="MAP", separate_c="slopes"
        )


def test_intercept_delta_reaches_the_nonsyn_channel_rates(tmp_path):
    """The fitted per-channel intercept must actually be applied
    downstream. It lives outside `cov_effects`, so it is easy to
    drop silently -- and it is a 13-18% correction on real data."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c="intercept"
    )
    delta = model._rg_delta_intercept
    assert delta is not None

    nonsyn_with = model.compute_channel_mu_gs("nonsyn")
    syn = model.compute_channel_mu_gs("syn")

    # the syn channel is untouched by the delta; the nonsyn one is
    # scaled by exactly exp(delta)
    model._rg_delta_intercept = None
    nonsyn_without = model.compute_channel_mu_gs("nonsyn")
    syn_again = model.compute_channel_mu_gs("syn")

    pd.testing.assert_frame_equal(syn, syn_again)
    ratio = (nonsyn_with / nonsyn_without).values
    assert np.allclose(ratio, np.exp(delta))


def test_intercept_is_the_default(tmp_path):
    """The default carries a fitted per-channel intercept, because a
    shared one leaves the channels miscalibrated by 13-18%."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    assert model._rg_separate_c == "intercept"
    assert model._rg_delta_intercept is not None


def test_rg_fit_survives_save_and_load(tmp_path):
    """The r_g fit is not recoverable from the saved rate matrices,
    so it must be persisted explicitly. Without it a reloaded model
    silently drops the channel corrections rather than failing."""
    from sigmutsel.models import Model

    model = _rg_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    model.estimate_channel_rg_cov_effects(sample="MAP")

    before = {
        "theta": model.rg_theta,
        "delta": model._rg_delta_intercept,
        "mode": model._rg_separate_c,
        "nonsyn": model.compute_channel_mu_gs("nonsyn"),
        "r_g": model.compute_r_g_for_evaluation(),
    }

    out = tmp_path / "model"
    model.save_model(out)
    loaded = Model.load_model(out)

    assert loaded.rg_theta == pytest.approx(before["theta"])
    assert loaded._rg_delta_intercept == pytest.approx(
        before["delta"]
    )
    assert loaded._rg_separate_c == before["mode"]
    # the corrections actually reach the rates after a round trip
    pd.testing.assert_frame_equal(
        loaded.compute_channel_mu_gs("nonsyn"), before["nonsyn"]
    )
    pd.testing.assert_series_equal(
        loaded.compute_r_g_for_evaluation(), before["r_g"]
    )
