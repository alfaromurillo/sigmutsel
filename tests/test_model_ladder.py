"""Tests for the nested-ladder restrictions of the channel likelihood.

The channel Poisson with ``r_g`` marginalised out is the *top* of a
nested ladder of rate models. Two switches reach the rungs below it,
and both are here because each one is a way to get a wrong number with
no error anywhere:

* ``fit_rg=False`` -- the plain two-channel Poisson. It has to be the
  ``theta -> inf`` limit of the marginal form, not merely something
  similar, or the two arms are not nested and the difference between
  them is not "what ``r_g`` buys".
* ``fit_intercept=False`` -- the intercept *pinned at 0* rather than
  fitted. A zero-column covariate matrix does not do this: the ones
  column is prepended unconditionally, so that gives a fitted
  intercept (arm 1), not "nothing fitted" (arm 0). The difference is
  the whole first rung.

Plus ``use_silent_channel=False``, the single-channel Poisson GLM with
offset ``log mu_bar^(nonsyn)`` over passengers, which is the ladder's
December-data arms.

The load-bearing test is
``test_arm_2a_matches_a_plain_combined_count_poisson_glm``: at
``delta = 0`` the consequence split vanishes algebraically and arms 1
and 2a *are* combined-count Poisson regressions, so an independent GLM
fit (statsmodels, nothing shared with the PyMC path) must reproduce
the fitted coefficients. If it does not, the whole ladder is void.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.estimate_rg import (
    channel_rg_log_likelihood,
    estimate_channel_rg_effect,
)

from .test_estimate_rg import _rg_model


def _two_channel_case(seed=0, n_genes=40):
    """Per-gene statistics for a synthetic two-channel cohort."""
    rng = np.random.default_rng(seed)
    cov = rng.normal(size=(n_genes, 2))
    baseline_silent = rng.gamma(2.0, 1.5, size=n_genes)
    baseline_non_silent = rng.gamma(2.0, 4.0, size=n_genes)
    truth = np.array([0.3, 0.45, -0.25])
    eta = truth[0] + cov @ truth[1:]
    counts_silent = rng.poisson(baseline_silent * np.exp(eta))
    counts_non_silent = rng.poisson(baseline_non_silent * np.exp(eta))
    return {
        "counts_silent": counts_silent.astype(float),
        "baseline_silent": baseline_silent,
        "counts_non_silent": counts_non_silent.astype(float),
        "baseline_non_silent": baseline_non_silent,
        "cov_matrix": cov,
    }


def _plain_two_channel_poisson(
    eta_silent,
    counts_silent,
    baseline_silent,
    eta_non_silent,
    counts_non_silent,
    baseline_non_silent,
):
    """The Poisson log-likelihood, minus the same dropped constant."""
    return float(
        np.sum(counts_silent * eta_silent)
        + np.sum(counts_non_silent * eta_non_silent)
        - np.sum(baseline_silent * np.exp(eta_silent))
        - np.sum(baseline_non_silent * np.exp(eta_non_silent))
    )


# --- the likelihood's two restrictions -------------------------------


def test_theta_none_is_the_plain_two_channel_poisson():
    case = _two_channel_case()
    eta = np.full(len(case["counts_silent"]), 0.11)

    got = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=None,
        counts_silent=case["counts_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_silent=case["baseline_silent"],
        baseline_non_silent=case["baseline_non_silent"],
    )
    want = _plain_two_channel_poisson(
        eta,
        case["counts_silent"],
        case["baseline_silent"],
        eta,
        case["counts_non_silent"],
        case["baseline_non_silent"],
    )
    assert got == pytest.approx(want, rel=1e-12)


def test_theta_none_is_the_large_theta_limit():
    """The no-r_g arm must be the same model the marginal form
    approaches, not a separately-written lookalike -- otherwise the
    arm-3-to-arm-4 step measures a code difference, not r_g."""
    case = _two_channel_case(seed=3)
    eta = np.full(len(case["counts_silent"]), -0.05)
    shared = {
        "eta_silent": eta,
        "counts_silent": case["counts_silent"],
        "counts_non_silent": case["counts_non_silent"],
        "baseline_silent": case["baseline_silent"],
        "baseline_non_silent": case["baseline_non_silent"],
    }
    limit = channel_rg_log_likelihood(theta=None, **shared)

    previous = None
    for theta in (1e4, 1e6, 1e8):
        gap = abs(
            channel_rg_log_likelihood(theta=theta, **shared) - limit
        )
        if previous is not None:
            assert gap < previous
        previous = gap
    assert previous < 1e-4


def test_use_silent_false_drops_the_silent_terms():
    case = _two_channel_case(seed=5)
    eta = np.linspace(-0.2, 0.2, len(case["counts_silent"]))

    got = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=None,
        use_silent=False,
        **{
            k: case[k]
            for k in (
                "counts_silent",
                "counts_non_silent",
                "baseline_silent",
                "baseline_non_silent",
            )
        },
    )
    want = float(
        np.sum(case["counts_non_silent"] * eta)
        - np.sum(case["baseline_non_silent"] * np.exp(eta))
    )
    assert got == pytest.approx(want, rel=1e-12)


def test_use_silent_false_ignores_the_silent_arrays_entirely():
    """Not "zeroed out" -- absent. estimate_channel_rg_effect clips
    baseline_silent away from 0 so its log is finite, so a silent
    channel switched off by zeroing would still leak a small term."""
    case = _two_channel_case(seed=7)
    eta = np.full(len(case["counts_silent"]), 0.02)
    shared = {
        "eta_silent": eta,
        "theta": None,
        "use_silent": False,
        "counts_non_silent": case["counts_non_silent"],
        "baseline_non_silent": case["baseline_non_silent"],
    }
    a = channel_rg_log_likelihood(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        **shared,
    )
    b = channel_rg_log_likelihood(
        counts_silent=np.zeros_like(case["counts_silent"]),
        baseline_silent=np.full_like(case["baseline_silent"], 1e-12),
        **shared,
    )
    assert a == b


def test_use_silent_false_with_r_g_is_the_single_channel_marginal():
    """The two switches are independent: dropping the silent channel
    must not quietly drop r_g with it."""
    case = _two_channel_case(seed=9)
    eta = np.full(len(case["counts_silent"]), 0.0)
    got = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=4.0,
        use_silent=False,
        counts_silent=case["counts_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_silent=case["baseline_silent"],
        baseline_non_silent=case["baseline_non_silent"],
    )
    want = channel_rg_log_likelihood(
        eta_silent=eta,
        theta=4.0,
        counts_silent=np.zeros_like(case["counts_silent"]),
        counts_non_silent=case["counts_non_silent"],
        baseline_silent=np.zeros_like(case["baseline_silent"]),
        baseline_non_silent=case["baseline_non_silent"],
    )
    assert got == pytest.approx(want, rel=1e-12)


# --- the fit ---------------------------------------------------------


def test_arm_zero_fits_nothing_and_never_calls_pymc(monkeypatch):
    """No covariates, no intercept, no delta, no r_g: there is not a
    single free parameter, and find_MAP raises on such a model. The
    fixed c = 0 vector must come back without PyMC being entered."""
    from sigmutsel import estimate_rg

    case = _two_channel_case(seed=11)

    def _fail(*args, **kwargs):
        raise AssertionError("PyMC must not be used for arm 0")

    monkeypatch.setattr(estimate_rg.pm, "Model", _fail)

    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=np.zeros((len(case["counts_silent"]), 0)),
        fit_rg=False,
        fit_intercept=False,
        draws=1,
    )
    assert isinstance(result, dict)
    assert "log_theta" not in result
    np.testing.assert_array_equal(result["c"], np.zeros(1))


def test_arm_one_intercept_matches_its_closed_form():
    """Arm 1 has an exact solution, so the fit can be checked against
    algebra rather than against another fit: maximising
    sum_g N_g c0 - e^{c0} sum_g B_g gives c0 = log(sum N / sum B)."""
    case = _two_channel_case(seed=13)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=np.zeros((len(case["counts_silent"]), 0)),
        fit_rg=False,
        draws=1,
    )
    total_counts = (
        case["counts_silent"].sum() + case["counts_non_silent"].sum()
    )
    total_baseline = (
        case["baseline_silent"].sum()
        + case["baseline_non_silent"].sum()
    )
    assert result["c"].shape == (1,)
    assert float(result["c"][0]) == pytest.approx(
        np.log(total_counts / total_baseline), abs=1e-4
    )


def test_arm_2a_matches_a_plain_combined_count_poisson_glm():
    """THE HARNESS GATE, as a unit test.

    At delta = 0 the channel-dependent part of the likelihood is
    sum(n_syn eta_syn) + sum(n_non eta_non), which collapses to
    sum(N_g eta_g): the split vanishes and arm 2a *is* a Poisson
    regression on combined counts with offset log(B_syn + B_non).
    Checked against statsmodels, which shares no code with the PyMC
    path.
    """
    sm = pytest.importorskip("statsmodels.api")

    case = _two_channel_case(seed=17, n_genes=200)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=case["cov_matrix"],
        fit_rg=False,
        draws=1,
    )

    combined_counts = (
        case["counts_silent"] + case["counts_non_silent"]
    )
    offset = np.log(
        case["baseline_silent"] + case["baseline_non_silent"]
    )
    design = sm.add_constant(case["cov_matrix"], prepend=True)
    glm = sm.GLM(
        combined_counts,
        design,
        family=sm.families.Poisson(),
        offset=offset,
    ).fit()

    np.testing.assert_allclose(
        result["c"], glm.params, rtol=0, atol=1e-4
    )


def test_fit_intercept_false_pins_it_at_zero_and_keeps_c_full_length():
    """c keeps its 1 + n_covariates length with a hard 0 in front, so
    every downstream consumer of cov_effects works unchanged."""
    sm = pytest.importorskip("statsmodels.api")

    case = _two_channel_case(seed=19, n_genes=200)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=case["cov_matrix"],
        fit_rg=False,
        fit_intercept=False,
        draws=1,
    )
    assert result["c"].shape == (3,)
    assert float(result["c"][0]) == 0.0

    glm = sm.GLM(
        case["counts_silent"] + case["counts_non_silent"],
        case["cov_matrix"],
        family=sm.families.Poisson(),
        offset=np.log(
            case["baseline_silent"] + case["baseline_non_silent"]
        ),
    ).fit()
    np.testing.assert_allclose(
        result["c"][1:], glm.params, rtol=0, atol=1e-4
    )


def test_single_channel_arm_matches_a_non_silent_only_glm():
    """Arms 1p/2ap: one channel, offset log(baseline_non_silent)."""
    sm = pytest.importorskip("statsmodels.api")

    case = _two_channel_case(seed=23, n_genes=200)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=case["cov_matrix"],
        fit_rg=False,
        use_silent_channel=False,
        draws=1,
    )
    glm = sm.GLM(
        case["counts_non_silent"],
        sm.add_constant(case["cov_matrix"], prepend=True),
        family=sm.families.Poisson(),
        offset=np.log(case["baseline_non_silent"]),
    ).fit()
    np.testing.assert_allclose(
        result["c"], glm.params, rtol=0, atol=1e-4
    )


def test_single_channel_rejects_a_per_channel_intercept():
    case = _two_channel_case(seed=29)
    with pytest.raises(ValueError, match="only one channel"):
        estimate_channel_rg_effect(
            counts_silent=case["counts_silent"],
            baseline_silent=case["baseline_silent"],
            counts_non_silent=case["counts_non_silent"],
            baseline_non_silent=case["baseline_non_silent"],
            cov_matrix=case["cov_matrix"],
            separate_c="intercept",
            use_silent_channel=False,
            draws=1,
        )


def test_per_coefficient_bounds_drop_their_intercept_entry():
    """An array bound is sized for the full c; pinning the intercept
    must trim it rather than misalign every slope's prior."""
    case = _two_channel_case(seed=31, n_genes=60)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=case["cov_matrix"],
        fit_rg=False,
        fit_intercept=False,
        lower_bounds_c=np.array([-2.0, -2.0, -2.0]),
        upper_bounds_c=np.array([2.0, 2.0, 2.0]),
        draws=1,
    )
    assert result["c"].shape == (3,)
    assert float(result["c"][0]) == 0.0


def test_mistaken_bound_length_raises():
    case = _two_channel_case(seed=37, n_genes=60)
    with pytest.raises(ValueError, match="intercept included"):
        estimate_channel_rg_effect(
            counts_silent=case["counts_silent"],
            baseline_silent=case["baseline_silent"],
            counts_non_silent=case["counts_non_silent"],
            baseline_non_silent=case["baseline_non_silent"],
            cov_matrix=case["cov_matrix"],
            fit_rg=False,
            fit_intercept=False,
            lower_bounds_c=np.array([-2.0, -2.0]),
            draws=1,
        )


def _arm_scorer(case):
    """Fit an arm on ``case`` and return its maximised log-likelihood."""
    n_genes = len(case["counts_silent"])
    zero_cov = np.zeros((n_genes, 0))

    def fit_and_score(cov, **kwargs):
        result = estimate_channel_rg_effect(
            counts_silent=case["counts_silent"],
            baseline_silent=case["baseline_silent"],
            counts_non_silent=case["counts_non_silent"],
            baseline_non_silent=case["baseline_non_silent"],
            cov_matrix=cov,
            draws=1,
            **kwargs,
        )
        cov_ext = np.concatenate([np.ones((n_genes, 1)), cov], axis=1)
        eta = cov_ext @ result["c"]
        delta = float(result.get("delta_intercept", 0.0))
        return float(
            channel_rg_log_likelihood(
                eta_silent=eta,
                eta_non_silent=eta + delta,
                theta=(
                    np.exp(float(result["log_theta"]))
                    if "log_theta" in result
                    else None
                ),
                counts_silent=case["counts_silent"],
                counts_non_silent=case["counts_non_silent"],
                baseline_silent=case["baseline_silent"],
                baseline_non_silent=case["baseline_non_silent"],
            )
        )

    return fit_and_score, zero_cov


def test_nested_arms_0_to_3_are_ordered_in_likelihood():
    """0 <= 1 <= {2a, 2b} <= 3 at their own maxima, on one data set.

    Each arm's parameter set contains the previous one's, so its
    maximised likelihood cannot be lower. This is the property that
    makes the ladder a ladder; it is also the cheapest possible
    detector of a restriction wired up wrongly. 2a and 2b are
    siblings, not nested in each other, so each is compared only with
    1 and with 3.
    """
    case = _two_channel_case(seed=41, n_genes=120)
    fit_and_score, zero_cov = _arm_scorer(case)

    arm_0 = fit_and_score(zero_cov, fit_rg=False, fit_intercept=False)
    arm_1 = fit_and_score(zero_cov, fit_rg=False)
    arm_2a = fit_and_score(case["cov_matrix"], fit_rg=False)
    arm_2b = fit_and_score(
        zero_cov, fit_rg=False, separate_c="intercept"
    )
    arm_3 = fit_and_score(
        case["cov_matrix"], fit_rg=False, separate_c="intercept"
    )

    tol = 1e-4
    assert arm_0 <= arm_1 + tol
    assert arm_1 <= arm_2a + tol
    assert arm_1 <= arm_2b + tol
    assert arm_2a <= arm_3 + tol
    assert arm_2b <= arm_3 + tol
    # and the ladder must not be flat, or it is measuring nothing
    assert arm_3 > arm_0 + 1.0


def test_arm_4_beats_arm_3_when_the_data_carry_real_r_g():
    """The r_g step is checked on dispersed data on purpose.

    Under counts drawn from the model *without* r_g, theta's maximum
    is at +infinity -- the boundary problem CLAUDE.md records for
    dispersion parameters in general -- so find_MAP stops on a flat
    ridge near the log_theta bound and arm 4 can land a few
    hundredths of a nat *below* arm 3. That is the optimiser on a
    ridge, not a broken nesting, and it would make a pure-Poisson
    ordering test flaky for a reason that has nothing to do with the
    ladder. With genuine per-gene dispersion in the data the step is
    unambiguous.
    """
    rng = np.random.default_rng(101)
    n_genes = 150
    cov = rng.normal(size=(n_genes, 2))
    baseline_silent = rng.gamma(2.0, 1.5, size=n_genes)
    baseline_non_silent = rng.gamma(2.0, 4.0, size=n_genes)
    eta = 0.3 + cov @ np.array([0.45, -0.25])
    r_g = rng.gamma(2.0, 0.5, size=n_genes)
    case = {
        "counts_silent": rng.poisson(
            baseline_silent * np.exp(eta) * r_g
        ).astype(float),
        "baseline_silent": baseline_silent,
        "counts_non_silent": rng.poisson(
            baseline_non_silent * np.exp(eta) * r_g
        ).astype(float),
        "baseline_non_silent": baseline_non_silent,
        "cov_matrix": cov,
    }
    fit_and_score, _ = _arm_scorer(case)

    arm_3 = fit_and_score(
        case["cov_matrix"], fit_rg=False, separate_c="intercept"
    )
    arm_4 = fit_and_score(case["cov_matrix"], separate_c="intercept")
    assert arm_4 > arm_3 + 1.0


# --- Model-level wiring ----------------------------------------------


def test_model_no_rg_arm_refuses_to_hand_out_an_r_g(tmp_path):
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", fit_rg=False, separate_c=False
    )
    assert model.rg_fitted is False
    with pytest.raises(ValueError, match="theta not fitted"):
        _ = model.rg_theta
    with pytest.raises(ValueError, match="No r_g fit available"):
        model.compute_r_g_for_evaluation()


def test_model_arm_zero_leaves_rates_at_the_baseline(tmp_path):
    """Arm 0 is mu = mu_bar: the non-synonymous channel's rates must
    come back as the untouched baseline, not merely close to it."""
    model = _rg_model(tmp_path)
    model.cov_matrix = model.cov_matrix.iloc[:, :0]
    model.estimate_channel_rg_cov_effects(
        sample="MAP",
        fit_rg=False,
        fit_intercept=False,
        separate_c=False,
    )
    np.testing.assert_array_equal(model.cov_effects, np.zeros(1))
    pd.testing.assert_frame_equal(
        model.compute_channel_mu_gs("nonsyn"),
        model.base_mus_nonsyn.loc[model.cov_matrix.index],
        check_exact=False,
    )


def test_model_zero_column_cov_matrix_survives_the_fit(tmp_path):
    """A zero-column covariate matrix is how arms 1 and 2b are built;
    it must reach the fit intact and give exactly one coefficient."""
    model = _rg_model(tmp_path)
    model.cov_matrix = model.cov_matrix.iloc[:, :0]
    assert model.cov_matrix.shape[1] == 0

    result = model.estimate_channel_rg_cov_effects(
        sample="MAP", fit_rg=False, separate_c=False
    )
    assert np.asarray(result).shape == (1,)
    assert np.isfinite(result).all()
    assert np.isfinite(
        model.estimate_passenger_genes_r2(target="non_silent_counts")
    )


def test_model_single_channel_arm_uses_passengers_only(tmp_path):
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP",
        fit_rg=False,
        use_silent_channel=False,
        separate_c=False,
    )
    stats = model._rg_statistics
    assert list(stats["genes"]) == list(stats["in_non_silent"])
    assert (stats["counts_silent"] == 0).all()
    assert (stats["baseline_silent"] == 0).all()
    assert model._rg_use_silent_channel is False


def test_model_log_likelihood_at_fit_tracks_the_arm(tmp_path):
    """channel_rg_log_likelihood_at_fit must evaluate the arm that was
    fitted -- a no-r_g arm scored with a theta it never had would make
    every ladder step wrong in the same direction."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", fit_rg=False, separate_c=False
    )
    no_rg = model.channel_rg_log_likelihood_at_fit()

    stats = model._rg_statistics
    cov = model.cov_matrix.loc[stats["genes"]].values
    cov_ext = np.concatenate(
        [np.ones((cov.shape[0], 1)), cov], axis=1
    )
    eta = cov_ext @ np.asarray(model.cov_effects)
    want = _plain_two_channel_poisson(
        eta,
        stats["counts_silent"].values,
        stats["baseline_silent"].values,
        eta,
        stats["counts_non_silent"].values,
        stats["baseline_non_silent"].values,
    )
    assert no_rg == pytest.approx(want, rel=1e-9)


def test_model_no_rg_arm_survives_save_and_load(tmp_path):
    """A no-r_g arm has no theta, and the channel state used to be
    persisted only when theta was present -- so without this the
    statistics and the split baselines went missing on reload and the
    arm came back unscoreable."""
    from sigmutsel.models import Model, MutationDataset

    model = _rg_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    model.estimate_channel_rg_cov_effects(
        sample="MAP", fit_rg=False, separate_c=False
    )
    before = model.channel_rg_log_likelihood_at_fit()

    model.save_model(tmp_path / "arm")
    reloaded = Model.load_model(tmp_path / "arm")

    assert reloaded.rg_fitted is False
    assert reloaded._rg_use_silent_channel is True
    assert reloaded.channel_rg_log_likelihood_at_fit() == (
        pytest.approx(before, rel=1e-9)
    )


def test_posterior_carries_the_pinned_intercept_as_c():
    """``sample="full"`` reads ``c`` back by name, so with the
    intercept pinned -- where ``c`` is a Deterministic rather than a
    free variable -- the sampler has to carry it into the posterior.
    If it does not, the MCMC arms fail at the first read rather than
    silently, but they fail; production fits with ``sample="full"``.
    """
    import arviz as az

    case = _two_channel_case(seed=43, n_genes=60)
    result = estimate_channel_rg_effect(
        counts_silent=case["counts_silent"],
        baseline_silent=case["baseline_silent"],
        counts_non_silent=case["counts_non_silent"],
        baseline_non_silent=case["baseline_non_silent"],
        cov_matrix=case["cov_matrix"],
        fit_rg=False,
        fit_intercept=False,
        draws=200,
        chains=2,
        burn=200,
    )
    drawn = az.extract(result, var_names=["c"])
    assert drawn.shape[0] == 3
    assert np.all(drawn.values[0, :] == 0.0)
    assert "log_theta" not in result.posterior


def test_map_fit_clears_a_previous_posterior(tmp_path):
    """Sweeping the arms in one process must not let a MAP arm be
    scored against the previous arm's posterior draws. Nothing warns
    if it does -- the numbers just come out wrong."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample=200, chains=2, burn=200, separate_c=False
    )
    assert model.cov_effects_posteriors is not None
    assert model.has_cov_effects_posteriors()

    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )
    assert model.cov_effects_posteriors is None
    assert not model.has_cov_effects_posteriors()


def test_integer_sample_is_the_number_of_draws(tmp_path):
    """An integer `sample` must give that many draws, not 4000.

    It used to be mapped to "full" silently, so a caller asking for
    fewer got a full-size fit with no indication -- which made an
    ELPD sweep cost four times what the argument suggested and left
    no way to make it cheaper.
    """
    import arviz as az

    model = _rg_model(tmp_path)
    idata = model.estimate_channel_rg_cov_effects(
        sample=200, chains=2, burn=200, separate_c=False
    )
    assert az.extract(idata, var_names=["c"]).sizes["sample"] == 200


def test_fewer_draws_than_chains_raises(tmp_path):
    """int(draws/chains) would be 0 per chain; say so rather than
    handing the sampler an empty request."""
    model = _rg_model(tmp_path)
    with pytest.raises(ValueError, match="fewer draws than chains"):
        model.estimate_channel_rg_cov_effects(
            sample=2, chains=4, separate_c=False
        )


def test_full_and_map_are_unchanged(tmp_path):
    """The two string modes production uses must not move."""
    import arviz as az

    model = _rg_model(tmp_path)
    idata = model.estimate_channel_rg_cov_effects(
        sample="full", chains=2, burn=200, separate_c=False
    )
    assert az.extract(idata, var_names=["c"]).sizes["sample"] == 4000

    result = model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )
    assert np.asarray(result).ndim == 1
    assert model.cov_effects_posteriors is None


def test_map_fit_records_optimiser_diagnostics(tmp_path):
    """A MAP fit has no chains, so nothing ever said whether the
    optimiser reached a mode -- which is the gap every nc was
    selected through. The gradient norm and scipy's success flag are
    the MAP analogue of r_hat and must travel with the fit."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )

    diag = model.map_diagnostics
    assert diag is not None
    assert diag["map_success"] is True
    assert np.isfinite(diag["map_grad_norm"])
    assert diag["map_grad_norm"] < 1.0
    assert diag["map_nit"] >= 0


def test_mcmc_fit_has_no_map_diagnostics(tmp_path):
    """They would be stale from an earlier MAP fit otherwise."""
    model = _rg_model(tmp_path)
    model.estimate_channel_rg_cov_effects(
        sample="MAP", separate_c=False
    )
    assert model.map_diagnostics is not None
    model.estimate_channel_rg_cov_effects(
        sample=200, chains=2, burn=200, separate_c=False
    )
    assert model.map_diagnostics is None
