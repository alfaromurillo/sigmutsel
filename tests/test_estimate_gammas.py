"""Tests for estimate_gammas.estimate_gamma_from_mus's natural
gamma-ceiling cap.

Regression coverage for a real finding: a weakly-identified gamma
(e.g. a giant passenger gene whose mutation count carries little
selection signal) can produce a likelihood that goes numerically
flat past a data-dependent scale -- P(present) = 1 - exp(-gamma *
mu) is clip-saturated for every sample -- and MCMC can then drift
anywhere in that flat region with clean-looking diagnostics (chains
agree with each other because there is nothing left to disagree
about), giving wildly different, scientifically meaningless point
estimates from run to run. Capping the prior at the natural ceiling
where clip-saturation kicks in keeps the sampler in the genuinely
informative region instead.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel import constants
from sigmutsel.estimate_gammas import (
    _natural_gamma_ceiling,
    estimate_gamma_from_mus,
)
from sigmutsel.models import Model

# A small mu value among the "no" group forces a low natural
# ceiling: -log(1e-12) / 0.01 ~= 2763.
_MUS_YES = np.array([0.5, 0.6, 0.55, 0.52, 0.58] * 4)
_MUS_NO = np.array([0.05, 0.06, 0.01, 0.04, 0.03] * 20)


def test_natural_gamma_ceiling_matches_smallest_mu():
    ceiling = _natural_gamma_ceiling(
        np.concatenate([_MUS_YES, _MUS_NO])
    )
    expected = -np.log(1e-12) / _MUS_NO.min()
    assert ceiling == expected


def test_cap_keeps_estimate_stable_across_seeds():
    """Without a natural cap, a weakly-identified gamma can land on
    very different values depending on the random seed once the
    sampler wanders past the point where the likelihood is flat.
    With the cap, repeated fits at different seeds should land close
    together instead."""
    means = []
    for seed in (0, 1, 2):
        constants.random_seed = seed
        result = estimate_gamma_from_mus(
            _MUS_YES,
            _MUS_NO,
            draws=1000,
            burn=500,
            upper_bound_prior=1e6,
            auto_raise_target_accept=False,
            cap_at_natural_ceiling=True,
        )
        means.append(float(result.posterior["gamma"].values.mean()))
    constants.random_seed = None

    # All three seeds should agree to within a small relative
    # tolerance -- a flat, uncapped posterior would instead scatter
    # across many orders of magnitude (see the module docstring).
    assert max(means) / min(means) < 1.5


def test_saturated_flag_and_bound_recorded_in_attrs():
    constants.random_seed = 0
    result = estimate_gamma_from_mus(
        _MUS_YES,
        _MUS_NO,
        draws=1000,
        burn=500,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
        cap_at_natural_ceiling=True,
    )
    constants.random_seed = None

    attrs = result.posterior.attrs
    assert "likelihood_saturated" in attrs
    assert "natural_gamma_ceiling" in attrs
    # The initial 1e6 bound must have been capped down to the
    # natural ceiling, not left at the requested value.
    assert attrs["final_upper_bound_prior"] < 1e6
    assert (
        attrs["final_upper_bound_prior"]
        == attrs["natural_gamma_ceiling"]
    )


def test_well_identified_gamma_is_not_capped():
    """A variant/gene with a well-separated, tiny-scale mu (mirrors
    real per-tumor variant rates, e.g. KRAS p.G12D) should have a
    natural ceiling far above the default bound. Whatever expansion
    the existing bound-saturation logic does for a genuinely large
    true value (the case it was built for) must not be mistaken for
    -- or blocked by -- the natural-ceiling cap."""
    mus_yes = np.array([5e-5] * 20)
    mus_no = np.array([1e-8] * 400)

    ceiling = _natural_gamma_ceiling(
        np.concatenate([mus_yes, mus_no])
    )
    assert ceiling > 1e6

    constants.random_seed = 0
    result = estimate_gamma_from_mus(
        mus_yes,
        mus_no,
        draws=1000,
        burn=500,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
        cap_at_natural_ceiling=True,
    )
    constants.random_seed = None

    attrs = result.posterior.attrs
    assert attrs["likelihood_saturated"] == 0
    assert (
        attrs["final_upper_bound_prior"]
        < attrs["natural_gamma_ceiling"]
    )


# --- Model._estimate_gamma_variant/_estimate_gamma_gene's
# --- excluded_samples (part of the L_low low-burden-correction
# --- rework's dropping-only option; see the plan for why weighting
# --- wasn't built here -- PyMC's Bernoulli likelihood in
# --- estimate_gamma_from_mus has no native per-observation weight
# --- argument). Both wrapper methods just build present_mask/
# --- absent_mask and delegate to estimate_gamma_from_mus (tested
# --- above), so these tests intercept that call rather than run a
# --- real MCMC fit -- they check the masking, not gamma estimation.


def _model_for_gamma_variant():
    model = Model.__new__(Model)
    model.mu_ms = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4]],
        index=["VAR1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    dataset = type("FakeDataset", (), {})()
    dataset.variants_present = pd.DataFrame(
        [[1, 0, 1, 0]],
        index=["VAR1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model.dataset = dataset
    model.gammas = {}
    return model


def test_estimate_gamma_variant_excludes_samples_from_both_masks(
    monkeypatch,
):
    captured = {}

    def fake_estimate_gamma_from_mus(mus_yes, mus_no, **kwargs):
        captured["yes_index"] = list(mus_yes.index)
        captured["no_index"] = list(mus_no.index)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        fake_estimate_gamma_from_mus,
    )

    model = _model_for_gamma_variant()
    model._estimate_gamma_variant(
        "VAR1", store=False, excluded_samples=["T1", "T4"]
    )

    # T1 was in the "present" group, T4 in "absent" -- both excluded,
    # T2/T3 untouched.
    assert captured["yes_index"] == ["T3"]
    assert captured["no_index"] == ["T2"]


def test_estimate_gamma_variant_no_excluded_samples_unchanged(
    monkeypatch,
):
    captured = {}

    def fake_estimate_gamma_from_mus(mus_yes, mus_no, **kwargs):
        captured["yes_index"] = list(mus_yes.index)
        captured["no_index"] = list(mus_no.index)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        fake_estimate_gamma_from_mus,
    )

    model = _model_for_gamma_variant()
    model._estimate_gamma_variant("VAR1", store=False)

    assert captured["yes_index"] == ["T1", "T3"]
    assert captured["no_index"] == ["T2", "T4"]


# --- The mu posterior cut (2-D mus_yes/mus_no). With 1-D mus,
# --- gamma's interval reflects only the Bernoulli sampling term and
# --- treats mu as known exactly. These tests check that 2-D
# --- (draws, tumors) input actually widens gamma's interval with
# --- wider mu uncertainty, and that a near-point-mass 2-D input
# --- reproduces the 1-D result -- not just that the code runs.


def _lognormal_draws(point_mus, sigma, n_draws, rng):
    """(n_draws, len(point_mus)) draws with the given point means."""
    return np.exp(
        rng.normal(
            loc=np.log(point_mus),
            scale=sigma,
            size=(n_draws, len(point_mus)),
        )
    )


def test_2d_near_point_mass_matches_1d_estimate():
    """Near-zero mu uncertainty (2-D input, tiny spread) should
    reproduce roughly the same gamma estimate as the 1-D (mu-known)
    call on the same point values."""
    rng = np.random.default_rng(0)
    mus_yes_2d = _lognormal_draws(_MUS_YES, 1e-4, 200, rng)
    mus_no_2d = _lognormal_draws(_MUS_NO, 1e-4, 200, rng)

    constants.random_seed = 0
    result_1d = estimate_gamma_from_mus(
        _MUS_YES,
        _MUS_NO,
        draws=1000,
        burn=500,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
    )
    result_2d = estimate_gamma_from_mus(
        mus_yes_2d,
        mus_no_2d,
        draws=1000,
        burn=500,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
    )
    constants.random_seed = None

    mean_1d = float(result_1d.posterior["gamma"].values.mean())
    mean_2d = float(result_2d.posterior["gamma"].values.mean())
    assert max(mean_1d, mean_2d) / min(mean_1d, mean_2d) < 1.5


def test_wider_mu_uncertainty_widens_gamma_posterior():
    """More spread in the mu posterior draws should propagate into a
    wider gamma posterior, not just a shifted mean -- the whole point
    of the cut over today's mu-known-exactly behavior.

    Uses a small tumor count (unlike ``_MUS_YES``/``_MUS_NO``'s ~120):
    with many i.i.d.-ish tumors the per-tumor cut priors partly
    average out in their effect on the single shared gamma, which
    washed out the contrast at the module's usual scale."""
    mus_yes = np.array([0.5, 0.55, 0.52])
    mus_no = np.array([0.05, 0.06, 0.04, 0.05, 0.03])

    rng = np.random.default_rng(1)
    mus_yes_narrow = _lognormal_draws(mus_yes, 0.02, 300, rng)
    mus_no_narrow = _lognormal_draws(mus_no, 0.02, 300, rng)
    mus_yes_wide = _lognormal_draws(mus_yes, 1.0, 300, rng)
    mus_no_wide = _lognormal_draws(mus_no, 1.0, 300, rng)

    constants.random_seed = 1
    result_narrow = estimate_gamma_from_mus(
        mus_yes_narrow,
        mus_no_narrow,
        draws=2000,
        burn=1000,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
    )
    result_wide = estimate_gamma_from_mus(
        mus_yes_wide,
        mus_no_wide,
        draws=2000,
        burn=1000,
        upper_bound_prior=1e6,
        auto_raise_target_accept=False,
    )
    constants.random_seed = None

    std_narrow = float(result_narrow.posterior["gamma"].values.std())
    std_wide = float(result_wide.posterior["gamma"].values.std())
    assert std_wide > 1.2 * std_narrow


def test_mismatched_ndim_raises():
    with pytest.raises(ValueError, match="1-D"):
        estimate_gamma_from_mus(
            np.ones((10, 5)), np.ones(20), draws=1
        )


def test_mismatched_draw_counts_raises():
    with pytest.raises(ValueError, match="draws"):
        estimate_gamma_from_mus(
            np.ones((10, 5)), np.ones((20, 5)), draws=1
        )


def test_estimate_gamma_public_forwards_excluded_samples(monkeypatch):
    captured = {}

    def fake_estimate_gamma_from_mus(mus_yes, mus_no, **kwargs):
        captured["yes_index"] = list(mus_yes.index)
        captured["no_index"] = list(mus_no.index)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        fake_estimate_gamma_from_mus,
    )

    model = _model_for_gamma_variant()
    model.estimate_gamma(
        "VAR1",
        level="variant",
        store=False,
        excluded_samples=["T1", "T4"],
    )

    assert captured["yes_index"] == ["T3"]
    assert captured["no_index"] == ["T2"]


# --- _estimate_gamma_gene's channel-rate fix. Found while validating
# --- the mu-posterior cut (mutation_rates/TODO.md's TOP PRIORITY
# --- entry) on real COAD data: a channel-split model's non-silent
# --- presence was being scored against the merged mu_gs (wrong
# --- baseline, missing delta_intercept) instead of
# --- compute_channel_mu_gs("nonsyn") -- the same distinction
# --- estimate_passenger_genes_r2 already makes for its "non_silent"
# --- targets. Overstated mu biased every existing channel-model gene
# --- gamma downward (~12% on APC/COAD).


def _model_for_gamma_gene_channel():
    """A channel-split model where the merged mu_gs (0.5) and the
    non-synonymous channel's own rate (0.1) deliberately differ, so a
    test can tell which one _estimate_gamma_gene actually used."""
    model = Model.__new__(Model)
    model._mu_gs = pd.DataFrame(
        [[0.5, 0.5, 0.5, 0.5]],
        index=["GENE1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model._base_mus_nonsyn = pd.DataFrame(
        [[0.1, 0.1, 0.1, 0.1]],
        index=["GENE1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model._base_mus_syn = pd.DataFrame(
        [[0.05, 0.05, 0.05, 0.05]],
        index=["GENE1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model.cov_matrix = pd.DataFrame({"cov1": [0.0]}, index=["GENE1"])
    model.cov_effects = np.array([0.0, 0.0])
    model._rg_delta_intercept = None
    dataset = type("FakeDataset", (), {})()
    dataset.mutation_db = pd.DataFrame(
        columns=["gene", "ensembl_gene_id"]
    )
    dataset.genes_present_non_silent = pd.DataFrame(
        [[1, 0, 1, 0]],
        index=["GENE1"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model.dataset = dataset
    model.gammas = {}
    return model


def test_estimate_gamma_gene_uses_channel_rate_when_available(
    monkeypatch,
):
    captured = {}

    def fake_estimate_gamma_from_mus(mus_yes, mus_no, **kwargs):
        captured["yes"] = list(mus_yes)
        captured["no"] = list(mus_no)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        fake_estimate_gamma_from_mus,
    )

    model = _model_for_gamma_gene_channel()
    model._estimate_gamma_gene("GENE1", store=False)

    # present: T1, T3; absent: T2, T4 -- values must come from the
    # nonsyn channel baseline (0.1), not the merged mu_gs (0.5).
    assert captured["yes"] == [0.1, 0.1]
    assert captured["no"] == [0.1, 0.1]


def test_estimate_gamma_gene_falls_back_to_merged_mu_gs_without_channels(
    monkeypatch,
):
    """A non-channel model (no base_mus_syn/nonsyn) keeps using the
    merged mu_gs -- unchanged behavior, only channel models get the
    fix."""
    captured = {}

    def fake_estimate_gamma_from_mus(mus_yes, mus_no, **kwargs):
        captured["yes"] = list(mus_yes)
        captured["no"] = list(mus_no)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        fake_estimate_gamma_from_mus,
    )

    model = _model_for_gamma_gene_channel()
    model._base_mus_syn = None
    model._base_mus_nonsyn = None
    model._estimate_gamma_gene("GENE1", store=False)

    assert captured["yes"] == [0.5, 0.5]
    assert captured["no"] == [0.5, 0.5]


# --- Model.estimate_gamma: use_mu_posterior/r_g_variant forwarding
# --- to _estimate_gamma_gene/_estimate_gamma_variant. The draws
# --- themselves (masking, r_g scaling) are covered against a real
# --- fit in test_estimate_rg.py -- this only checks the public
# --- entry point actually threads the two new kwargs through to the
# --- right private method, for both levels.


def _model_stub_for_forwarding():
    model = Model.__new__(Model)
    model.mu_ms = None
    model._mu_gs = pd.DataFrame(
        [[0.5]], index=["GENE1"], columns=["T1"]
    )
    return model


def test_estimate_gamma_forwards_use_mu_posterior_to_gene(
    monkeypatch,
):
    model = _model_stub_for_forwarding()
    captured = {}

    def fake_estimate_gamma_gene(item, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        model, "_estimate_gamma_gene", fake_estimate_gamma_gene
    )
    model.estimate_gamma(
        "GENE1",
        level="gene",
        use_mu_posterior=True,
        r_g_variant="evaluation",
    )

    assert captured["use_mu_posterior"] is True
    assert captured["r_g_variant"] == "evaluation"


def test_estimate_gamma_forwards_use_mu_posterior_to_variant(
    monkeypatch,
):
    model = _model_stub_for_forwarding()
    captured = {}

    def fake_estimate_gamma_variant(item, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        model, "_estimate_gamma_variant", fake_estimate_gamma_variant
    )
    model.estimate_gamma(
        "VAR1",
        level="variant",
        use_mu_posterior=True,
        r_g_variant="evaluation",
    )

    assert captured["use_mu_posterior"] is True
    assert captured["r_g_variant"] == "evaluation"


def test_estimate_gamma_defaults_use_mu_posterior_off(monkeypatch):
    model = _model_stub_for_forwarding()
    captured = {}

    def fake_estimate_gamma_gene(item, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        model, "_estimate_gamma_gene", fake_estimate_gamma_gene
    )
    model.estimate_gamma("GENE1", level="gene")

    assert captured["use_mu_posterior"] is False
    assert captured["r_g_variant"] == "none"


# ---------------------------------------------------------------------
# Gene-tumor dispersion in the presence likelihood (``gene_tumor_dispersion``)
# ---------------------------------------------------------------------


def _dispersed_presence(seed, gamma, phi, n_tumors=1500):
    """Presence data simulated under the gene-tumor Gamma rate model."""
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0, 1.2, n_tumors))
    mu = 0.05 * mu / mu.mean()
    m = mu.sum()
    lam = rng.gamma(phi * mu / m, m / phi)
    present = rng.random(n_tumors) > np.exp(-gamma * lam)
    return mu, present


def _closed_form_mle(mu, present, phi):
    """Brute-force grid MLE of the same likelihood, for comparison."""
    m = mu.sum()
    grid = np.exp(np.linspace(np.log(0.5), np.log(100), 4000))
    lls = []
    for g in grid:
        log_absent = -phi * (mu / m) * np.log1p(g * m / phi)
        lls.append(
            log_absent[~present].sum()
            + np.log(-np.expm1(log_absent[present])).sum()
        )
    return grid[int(np.argmax(lls))]


def test_dispersed_ceiling_reduces_to_dispersion_free():
    mus = np.concatenate([_MUS_YES, _MUS_NO])
    from sigmutsel.estimate_gammas import (
        _natural_gamma_ceiling_dispersed,
    )

    assert _natural_gamma_ceiling_dispersed(
        mus, 1e9
    ) == pytest.approx(_natural_gamma_ceiling(mus), rel=1e-4)
    assert _natural_gamma_ceiling_dispersed(mus, 5.0) > (
        _natural_gamma_ceiling(mus)
    )


def test_gene_tumor_dispersion_rejects_nonpositive():
    with pytest.raises(ValueError, match="gene_tumor_dispersion"):
        estimate_gamma_from_mus(
            _MUS_YES, _MUS_NO, draws=1, gene_tumor_dispersion=0
        )


def test_gene_tumor_dispersion_map_matches_grid():
    """MAP under dispersion equals a brute-force grid MLE of the same
    closed form -- the check that the PyTensor expression is the
    likelihood the docstring states."""
    phi = 150.0
    mu, present = _dispersed_presence(3, 8.0, phi, n_tumors=280)
    constants.random_seed = 0
    disp = estimate_gamma_from_mus(
        mu[present], mu[~present], draws=1, gene_tumor_dispersion=phi
    )
    constants.random_seed = None
    order = np.concatenate(
        [np.flatnonzero(present), np.flatnonzero(~present)]
    )
    is_yes = np.arange(len(mu)) < present.sum()
    g_grid = _closed_form_mle(mu[order], is_yes, phi)
    assert float(disp["gamma"]) == pytest.approx(g_grid, rel=0.01)


def test_gene_tumor_dispersion_recovers_planted_gamma():
    """Across replicates simulated under the dispersion model, the
    dispersion-aware MLE recovers the planted gamma and the
    dispersion-free one is biased low. A single replicate is too noisy
    to test: with shape ``phi * p_j`` well below one, each tumor
    carries little information."""
    gamma, phi = 8.0, 150.0
    disp, plain = [], []
    for seed in range(25):
        mu, present = _dispersed_presence(100 + seed, gamma, phi, 280)
        disp.append(_closed_form_mle(mu, present, phi))
        plain.append(_closed_form_mle(mu, present, 1e12))
    assert abs(np.log(np.median(disp) / gamma)) < 0.12
    assert np.median(plain) < np.median(disp)


def test_gene_tumor_dispersion_large_phi_matches_default():
    constants.random_seed = 0
    a = estimate_gamma_from_mus(_MUS_YES, _MUS_NO, draws=1)
    b = estimate_gamma_from_mus(
        _MUS_YES, _MUS_NO, draws=1, gene_tumor_dispersion=1e9
    )
    constants.random_seed = None
    assert float(b["gamma"]) == pytest.approx(
        float(a["gamma"]), rel=1e-3
    )


def test_gene_tumor_dispersion_with_mu_posterior_cut_samples():
    """The production path: 2-D mu draws (the mu-posterior cut, so mu
    is a latent vector whose total M is a random variable) together
    with dispersion. It must build, sample, and land near the
    point-mu dispersion MLE when the draws are tight."""
    phi = 150.0
    mu, present = _dispersed_presence(7, 8.0, phi, n_tumors=280)
    rng = np.random.default_rng(0)
    draws = mu[None, :] * np.exp(rng.normal(0, 0.01, (200, len(mu))))
    constants.random_seed = 0
    res = estimate_gamma_from_mus(
        draws[:, present],
        draws[:, ~present],
        draws=800,
        burn=400,
        chains=2,
        gene_tumor_dispersion=phi,
        auto_raise_target_accept=False,
    )
    constants.random_seed = None
    order = np.concatenate(
        [np.flatnonzero(present), np.flatnonzero(~present)]
    )
    is_yes = np.arange(len(mu)) < present.sum()
    g_grid = _closed_form_mle(mu[order], is_yes, phi)
    g_post = float(np.median(res.posterior["gamma"].values))
    assert abs(np.log(g_post / g_grid)) < 0.25


def test_gene_tumor_shape_reproduces_gene_tumor_dispersion():
    """gene_tumor_dispersion is the special case k_j = phi mu_j / M."""
    phi = 150.0
    mu, present = _dispersed_presence(11, 8.0, phi, n_tumors=280)
    k = phi * mu / mu.sum()
    constants.random_seed = 0
    a = estimate_gamma_from_mus(
        mu[present], mu[~present], draws=1, gene_tumor_dispersion=phi
    )
    b = estimate_gamma_from_mus(
        mu[present],
        mu[~present],
        draws=1,
        gene_tumor_shape=(k[present], k[~present]),
    )
    constants.random_seed = None
    assert float(b["gamma"]) == pytest.approx(
        float(a["gamma"]), rel=1e-4
    )


def test_gene_tumor_shape_argument_errors():
    k = np.ones(len(_MUS_YES) + len(_MUS_NO))
    with pytest.raises(ValueError, match="not both"):
        estimate_gamma_from_mus(
            _MUS_YES,
            _MUS_NO,
            draws=1,
            gene_tumor_dispersion=10.0,
            gene_tumor_shape=(k[: len(_MUS_YES)], k[len(_MUS_YES) :]),
        )
    with pytest.raises(ValueError, match="one shape per tumor"):
        estimate_gamma_from_mus(
            _MUS_YES,
            _MUS_NO,
            draws=1,
            gene_tumor_shape=(k[:3], k[:3]),
        )
    with pytest.raises(ValueError, match="positive"):
        estimate_gamma_from_mus(
            _MUS_YES,
            _MUS_NO,
            draws=1,
            gene_tumor_shape=(
                0 * k[: len(_MUS_YES)],
                k[len(_MUS_YES) :],
            ),
        )


def test_variant_inheriting_gene_dispersion_recovers_gamma():
    """A variant sharing its gene's per-tumor multiplier: the shaped
    likelihood recovers gamma across replicates and the plain one is
    biased low. The variant's rate is a small, tumor-varying fraction
    of the gene's, so its own rates say nothing about the shape."""
    gamma, phi, n = 30.0, 60.0, 400

    def mle(mu_m, present, shapes):
        grid = np.exp(np.linspace(np.log(1), np.log(400), 3000))
        best, arg = -np.inf, None
        for g in grid:
            la = -shapes * np.log1p(g * mu_m / shapes)
            ll = (
                la[~present].sum()
                + np.log(-np.expm1(la[present])).sum()
            )
            if ll > best:
                best, arg = ll, g
        return arg

    shaped, plain = [], []
    for seed in range(25):
        rng = np.random.default_rng(200 + seed)
        mu_g = np.exp(rng.normal(0, 1.2, n))
        mu_g = 0.05 * mu_g / mu_g.mean()
        p = mu_g / mu_g.sum()
        shapes = phi * p
        eps = rng.gamma(shapes, 1.0 / shapes)
        mu_m = mu_g * rng.uniform(0.02, 0.08, n)
        present = rng.random(n) > np.exp(-gamma * mu_m * eps)
        shaped.append(mle(mu_m, present, shapes))
        plain.append(mle(mu_m, present, np.full(n, 1e12)))
    assert abs(np.log(np.median(shaped) / gamma)) < 0.15
    assert np.median(plain) < np.median(shaped)


def test_cut_drops_absent_tumors_with_zero_rate():
    """A tumor with rate 0 lacks the mutation with probability 1 for
    every gamma, so adding such tumors must leave the estimate alone
    rather than stop the sampler at log(0)."""
    rng = np.random.default_rng(2)
    mus_yes_2d = _lognormal_draws(_MUS_YES, 1e-4, 200, rng)
    mus_no_2d = _lognormal_draws(_MUS_NO, 1e-4, 200, rng)
    padded_no = np.concatenate(
        [mus_no_2d, np.zeros((200, 3))], axis=1
    )
    kwargs = {
        "draws": 1000,
        "burn": 500,
        "upper_bound_prior": 1e6,
        "auto_raise_target_accept": False,
    }

    constants.random_seed = 2
    base = estimate_gamma_from_mus(mus_yes_2d, mus_no_2d, **kwargs)
    constants.random_seed = 2
    padded = estimate_gamma_from_mus(mus_yes_2d, padded_no, **kwargs)
    constants.random_seed = None

    assert padded.posterior.attrs["n_zero_rate_absent_dropped"] == 3
    np.testing.assert_allclose(
        padded.posterior["gamma"].values,
        base.posterior["gamma"].values,
    )


def test_cut_rejects_a_carried_mutation_at_zero_rate():
    rng = np.random.default_rng(3)
    mus_yes_2d = _lognormal_draws(_MUS_YES, 1e-4, 50, rng)
    mus_yes_2d[:, 0] = 0.0
    mus_no_2d = _lognormal_draws(_MUS_NO, 1e-4, 50, rng)
    with pytest.raises(ValueError, match="positive rates"):
        estimate_gamma_from_mus(
            mus_yes_2d, mus_no_2d, draws=10, burn=10
        )


# --- Per-variant/per-gene sample accounting. A gamma is a claim
# --- about its denominator, and excluded_samples moves that
# --- denominator without leaving a trace in the posterior, so the
# --- counts are stamped into posterior.attrs beside it.


def _fake_posterior_result():
    """A minimal real InferenceData -- attrs must survive netCDF."""
    import arviz as az

    return az.from_dict({"gamma": np.ones((2, 5))})


def _patch_fit(monkeypatch, result):
    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        lambda mus_yes, mus_no, **kwargs: result,
    )


def test_variant_accounting_counts_the_excluded(monkeypatch):
    result = _fake_posterior_result()
    _patch_fit(monkeypatch, result)

    model = _model_for_gamma_variant()
    model._estimate_gamma_variant(
        "VAR1", store=False, excluded_samples=["T1", "T4"]
    )

    attrs = result.posterior.attrs
    # Of four tumors, T1 (present) and T4 (absent) were dropped.
    assert attrs["n_tumors_with"] == 1
    assert attrs["n_tumors_without"] == 1
    assert attrs["n_tumors_included"] == 2
    assert attrs["n_tumors_excluded"] == 2
    assert attrs["n_tumors_held_out"] == 0


def test_accounting_without_exclusions_covers_every_tumor(
    monkeypatch,
):
    result = _fake_posterior_result()
    _patch_fit(monkeypatch, result)

    model = _model_for_gamma_variant()
    model._estimate_gamma_variant("VAR1", store=False)

    attrs = result.posterior.attrs
    assert attrs["n_tumors_with"] == 2
    assert attrs["n_tumors_without"] == 2
    assert attrs["n_tumors_excluded"] == 0


def test_gene_accounting_is_recorded_too(monkeypatch):
    result = _fake_posterior_result()
    _patch_fit(monkeypatch, result)

    model = _model_for_gamma_gene_channel()
    model._estimate_gamma_gene("GENE1", store=False)

    assert result.posterior.attrs["n_tumors_included"] == len(
        model.dataset.genes_present_non_silent.columns
    )


def test_accounting_survives_the_netcdf_round_trip(
    monkeypatch, tmp_path
):
    """attrs is the storage claim: gammas are saved as .nc files."""
    import arviz as az

    result = _fake_posterior_result()
    _patch_fit(monkeypatch, result)
    model = _model_for_gamma_variant()
    model._estimate_gamma_variant(
        "VAR1", store=False, excluded_samples=["T1"]
    )

    path = tmp_path / "gamma.nc"
    result.to_netcdf(str(path))
    reloaded = az.from_netcdf(str(path))

    assert reloaded.posterior.attrs["n_tumors_with"] == 1
    assert reloaded.posterior.attrs["n_tumors_excluded"] == 1


def test_gamma_sample_accounting_shows_a_gamma_with_no_counts(
    monkeypatch,
):
    """A result fitted before this existed must be visible, not
    dropped: the table's job is to show what is unaccounted for."""
    result = _fake_posterior_result()
    _patch_fit(monkeypatch, result)

    model = _model_for_gamma_variant()
    model._estimate_gamma_variant("VAR1")
    model.gammas["OLD"] = "a result from before the accounting"

    table = model.gamma_sample_accounting()

    assert table.loc["VAR1", "n_tumors_included"] == 4
    assert np.isnan(table.loc["OLD", "n_tumors_included"])
    assert list(table.index) == ["VAR1", "OLD"]


# --- presence_probability, the extracted link, and the
# --- presence_model= seam that replaces it. The three branches are
# --- checked against closed forms and against each other, since the
# --- scalar-dispersion branch is meant to be the per-tumor-shape
# --- branch's special case k_j = phi mu_j / M.


def _p(gamma, mus, **kwargs):
    from sigmutsel.estimate_gammas import presence_probability

    return np.asarray(
        presence_probability(gamma, np.asarray(mus), **kwargs).eval()
    )


def test_plain_link_is_the_poisson_presence_probability():
    mus = np.array([0.01, 0.1, 1.0])
    assert np.allclose(_p(3.0, mus), 1 - np.exp(-3.0 * mus))


def test_shape_link_is_the_negative_binomial_form():
    mus = np.array([0.01, 0.1, 1.0])
    k = np.array([2.0, 5.0, 50.0])
    assert np.allclose(
        _p(3.0, mus, gene_tumor_shape=k),
        1 - (1 + 3.0 * mus / k) ** (-k),
    )


def test_scalar_dispersion_is_the_per_tumor_shape_special_case():
    mus = np.array([0.01, 0.1, 1.0])
    phi = 40.0
    k = phi * mus / mus.sum()
    assert np.allclose(
        _p(3.0, mus, gene_tumor_dispersion=phi),
        _p(3.0, mus, gene_tumor_shape=k),
    )


def test_dispersion_vanishes_as_phi_grows():
    mus = np.array([0.01, 0.1, 1.0])
    assert np.allclose(
        _p(3.0, mus, gene_tumor_dispersion=1e12),
        _p(3.0, mus),
        atol=1e-9,
    )


def test_link_is_clipped_off_zero_and_one():
    from sigmutsel.estimate_gammas import _CLIP_FLOOR

    mus = np.array([1e-300, 1.0])
    probabilities = _p(1e12, mus)
    assert probabilities[0] == pytest.approx(_CLIP_FLOOR)
    assert probabilities[1] == pytest.approx(1 - _CLIP_FLOOR)


def test_presence_model_replaces_the_link():
    """A link ten times weaker per mutation should need about ten
    times the gamma to explain the same presence pattern."""
    import pytensor.tensor as tt

    def weaker(gamma, mu, **kwargs):
        return tt.clip(1 - tt.exp(-gamma * mu / 10), 1e-12, 1 - 1e-12)

    constants.random_seed = 11
    default = estimate_gamma_from_mus(
        _MUS_YES, _MUS_NO, draws=1, upper_bound_prior=1e5
    )
    replaced = estimate_gamma_from_mus(
        _MUS_YES,
        _MUS_NO,
        draws=1,
        upper_bound_prior=1e5,
        presence_model=weaker,
    )

    assert float(replaced["gamma"]) == pytest.approx(
        10 * float(default["gamma"]), rel=0.05
    )
