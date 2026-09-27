"""Tests for the covariate fallback.

A gene without a complete covariate row used to vanish from every
covariate-scaled rate. It now keeps one. With a PCA-reduced matrix it
is projected onto the complete genes' PC basis from the covariates it
has (a missing standardized value is 0, the column mean) and shifted
by a fitted ``d_B`` per covariate block it misses; without PCA it
falls back to its baseline, ``nu = mu_bar``. Everything downstream of
the covariate term still applies to it -- the non-synonymous
channel's ``delta``, ``r_g`` from its own silent counts, and the
gene-tumor dispersion.

Four things fail silently if they are wrong, so each is pinned here:

1. ``delta`` reaches the fallback gene on every path (it used to be
   dropped on the variant path, which appended the baseline *after*
   applying ``delta``).
2. The fallback gene gets an ``r_g`` and its shifts without entering
   the fit that estimates ``c`` and ``theta``.
3. The projection places a complete row exactly where the PCA put it.
4. A gamma resting on a fallback says so.

The model-level tests below the PCA section use a raw (non-PCA)
covariate matrix, i.e. the baseline fallback.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sigmutsel.estimate_mus import (
    compute_mus_per_gene_per_sample,
    covariate_complete_genes,
)
from sigmutsel.estimate_rg import fit_fallback_shifts
from sigmutsel.models import Model, MutationDataset
from sigmutsel.utils import project_onto_pca, run_pca_on_covariates

from .test_channel_cov_effects import _model_with_channels

_FALLBACK = "ENSG_C"


def _fallback_model(tmp_path, how="absent"):
    """The synthetic channel model with ENSG_C's covariates removed.

    ``how="absent"`` drops its row, ``"nan"`` keeps it as NaN -- the
    two ways a gene arrives without covariates.
    """
    model = _model_with_channels(tmp_path)
    model.dataset.compute_gene_counts_channels()
    if how == "absent":
        model.cov_matrix = model.cov_matrix.drop(index=_FALLBACK)
    else:
        model.cov_matrix.loc[_FALLBACK] = np.nan
    return model


# --- the low-level function ---


def _toy():
    base = pd.DataFrame(
        {"T1": [1.0, 2.0, 3.0], "T2": [4.0, 5.0, 6.0]},
        index=pd.Index(["G1", "G2", "G3"], name="ensembl_gene_id"),
    )
    cov = pd.DataFrame({"x": [1.0, np.nan]}, index=["G1", "G2"])
    return base, cov


def test_without_a_fallback_the_gene_is_dropped():
    """The historical behaviour is kept when no fallback is asked for:
    G3 (absent) is dropped. G2 (a NaN row) comes back NaN, which is
    why a caller must never score such a matrix directly."""
    base, cov = _toy()
    out = compute_mus_per_gene_per_sample(None, base, [0.5, 2.0], cov)
    assert list(out.index) == ["G1", "G2"]
    assert out.loc["G2"].isna().all()


@pytest.mark.parametrize("log_scale", [0.0, -0.3])
def test_fallback_keeps_every_gene_at_its_scaled_baseline(log_scale):
    base, cov = _toy()
    c = [0.5, 2.0]
    out = compute_mus_per_gene_per_sample(
        None, base, c, cov, fallback_log_scale=log_scale
    )
    assert list(out.index) == list(base.index)
    assert out.index.name == base.index.name
    np.testing.assert_allclose(
        out.loc["G1"], base.loc["G1"] * np.exp(0.5 + 2.0 * 1.0)
    )
    for gene in ("G2", "G3"):  # a NaN row and an absent one
        np.testing.assert_allclose(
            out.loc[gene], base.loc[gene] * np.exp(log_scale)
        )


def test_fallback_in_signature_separated_mode():
    base, cov = _toy()
    base_mus = {"s1": base, "s2": 2 * base}
    c = np.array([[0.5, 2.0], [0.0, -1.0]])
    out = compute_mus_per_gene_per_sample(
        None, base_mus, c, cov, fallback_log_scale=0.0
    )
    assert list(out.index) == list(base.index)
    np.testing.assert_allclose(
        out.loc["G1"],
        base.loc["G1"] * np.exp(2.5)
        + 2 * base.loc["G1"] * np.exp(-1.0),
    )
    np.testing.assert_allclose(out.loc["G3"], 3 * base.loc["G3"])


def test_complete_genes_helper():
    _, cov = _toy()
    assert list(covariate_complete_genes(cov)) == ["G1"]


# --- the model ---


@pytest.mark.parametrize("how", ["absent", "nan"])
def test_fallback_genes_are_identified(tmp_path, how):
    model = _fallback_model(tmp_path, how)
    assert list(model.covariate_fallback_genes) == [_FALLBACK]


def test_no_covariates_means_no_fallback(tmp_path):
    model = _fallback_model(tmp_path)
    model.cov_matrix = None
    assert len(model.covariate_fallback_genes) == 0


def test_fallback_gene_is_not_in_the_fit_but_has_statistics(tmp_path):
    """Fit c and theta on complete cases, predict for everyone."""
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    assert _FALLBACK not in model._rg_statistics["genes"]
    assert list(model._rg_fallback_statistics["genes"]) == [_FALLBACK]
    assert model.n_in_cov_effects_estimation == 2


def test_fit_is_unchanged_by_the_fallback_gene(tmp_path):
    """The fallback gene's data must not reach c: fitting with it as a
    fallback gene equals fitting without the gene at all."""
    with_gene = _fallback_model(tmp_path / "a")
    with_gene.estimate_channel_rg_cov_effects(sample="MAP")

    without = _fallback_model(tmp_path / "b")
    keep = without.cov_matrix.index
    for name in ("_base_mus", "_base_mus_syn", "_base_mus_nonsyn"):
        setattr(without, name, getattr(without, name).loc[keep])
    without.estimate_channel_rg_cov_effects(sample="MAP")

    np.testing.assert_allclose(
        with_gene.cov_effects, without.cov_effects, rtol=1e-6
    )
    assert with_gene.rg_theta == pytest.approx(without.rg_theta)


def test_channel_rates_include_the_fallback_gene_with_delta(tmp_path):
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    delta = model._rg_delta_intercept
    assert delta is not None

    nonsyn = model.compute_channel_mu_gs("nonsyn")
    syn = model.compute_channel_mu_gs("syn")
    assert list(nonsyn.index) == list(model.base_mus_nonsyn.index)
    np.testing.assert_allclose(
        nonsyn.loc[_FALLBACK],
        model.base_mus_nonsyn.loc[_FALLBACK] * np.exp(delta),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        syn.loc[_FALLBACK],
        model.base_mus_syn.loc[_FALLBACK],
        rtol=1e-6,
    )


def test_variant_path_carries_delta_for_the_fallback_gene(tmp_path):
    """Regression: _compute_mu_g_taus appended covariate-less genes
    AFTER applying delta, so their non-synonymous variants had none.
    The per-type rates must sum to the channel total, delta included,
    for every gene."""
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")

    total = sum(model._compute_mu_g_taus().values()).sort_index(
        axis=1
    )
    nonsyn = model.compute_channel_mu_gs("nonsyn").sort_index(axis=1)
    pd.testing.assert_frame_equal(
        total.loc[nonsyn.index], nonsyn, check_exact=False
    )
    np.testing.assert_allclose(
        total.loc[_FALLBACK],
        model.base_mus_nonsyn.loc[_FALLBACK, total.columns]
        * np.exp(model._rg_delta_intercept),
        rtol=1e-5,
    )


def test_merged_rates_include_the_fallback_gene(tmp_path):
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    assert list(model.mu_gs.index) == list(model.base_mus.index)
    np.testing.assert_allclose(
        model.mu_gs.loc[_FALLBACK], model.base_mus.loc[_FALLBACK]
    )


def test_fallback_gene_gets_the_closed_form_r_g(tmp_path):
    """r_g for a fallback gene is the same conjugate posterior mean
    as anyone's, with its baseline as the expectation."""
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    theta = model.rg_theta
    stats = model._rg_fallback_statistics

    r_eval = model.compute_r_g_for_evaluation()
    expected = (theta + stats["counts_silent"][_FALLBACK]) / (
        theta + stats["baseline_silent"][_FALLBACK]
    )
    assert r_eval[_FALLBACK] == pytest.approx(expected)
    assert (
        stats["counts_silent"][_FALLBACK] == 1
    )  # the one Silent call
    assert set(r_eval.index) == set(model.base_mus_nonsyn.index)
    assert set(model.compute_r_g_production().index) == set(
        model.base_mus_nonsyn.index
    )


def test_fit_likelihood_is_over_the_fit_genes_only(tmp_path):
    """The fallback statistics are kept apart, so the fitted objective
    is still the one the fit maximised."""
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    ll = model.channel_rg_log_likelihood_at_fit()
    model._rg_fallback_statistics = None
    assert model.channel_rg_log_likelihood_at_fit() == pytest.approx(
        ll
    )


def test_posterior_draws_for_a_fallback_gene(tmp_path):
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="full")

    none = model.compute_mu_g_posterior_draws(
        _FALLBACK, r_g_variant="none"
    )
    import arviz as az

    delta = az.extract(
        model.cov_effects_posteriors, var_names=["delta_intercept"]
    ).values
    np.testing.assert_allclose(
        none.to_numpy(),
        np.exp(delta)[:, None]
        * model.base_mus_nonsyn.loc[_FALLBACK].to_numpy()[None, :],
        rtol=1e-6,
    )

    evaluation = model.compute_mu_g_posterior_draws(
        _FALLBACK,
        r_g_variant="evaluation",
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(evaluation.to_numpy()).all()
    assert (evaluation.to_numpy() > 0).all()


def test_old_model_without_fallback_statistics_explains_itself(
    tmp_path,
):
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="full")
    model._rg_fallback_statistics = None
    with pytest.raises(
        ValueError, match="compute_rg_fallback_statistics"
    ):
        model.compute_mu_g_posterior_draws(
            _FALLBACK, r_g_variant="evaluation"
        )
    model.compute_rg_fallback_statistics()
    model.compute_mu_g_posterior_draws(
        _FALLBACK, r_g_variant="evaluation"
    )


def _fake_gamma(monkeypatch, captured):
    import sigmutsel.estimate_gammas as estimate_gammas_mod

    def fake(mus_yes, mus_no, **kwargs):
        captured["yes"] = np.asarray(mus_yes)
        return SimpleNamespace(posterior=SimpleNamespace(attrs={}))

    monkeypatch.setattr(
        estimate_gammas_mod, "estimate_gamma_from_mus", fake
    )


@pytest.mark.parametrize(
    "gene, flag",
    [(_FALLBACK, 1), ("ENSG_A", 0)],
    ids=["fallback", "covered"],
)
def test_gene_gamma_is_marked(tmp_path, monkeypatch, gene, flag):
    model = _fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="full")
    model.dataset.genes_present_non_silent = pd.DataFrame(
        [[1, 0, 1]], index=[gene], columns=["T1", "T2", "T3"]
    )
    captured = {}
    _fake_gamma(monkeypatch, captured)

    result = model._estimate_gamma_gene(
        gene,
        store=False,
        use_mu_posterior=True,
        r_g_variant="evaluation",
    )
    assert result.posterior.attrs["covariate_fallback"] == flag
    assert np.isfinite(captured["yes"]).all()


def test_fallback_statistics_survive_save_and_load(tmp_path):
    model = _fallback_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    model.estimate_channel_rg_cov_effects(sample="MAP")
    before = model.compute_r_g_for_evaluation()

    model.save_model(tmp_path / "model")
    loaded = Model.load_model(tmp_path / "model")

    assert list(loaded._rg_fallback_statistics["genes"]) == [
        _FALLBACK
    ]
    pd.testing.assert_series_equal(
        loaded.compute_r_g_for_evaluation(), before
    )
    assert list(loaded.covariate_fallback_genes) == [_FALLBACK]


# --- PCA: projection and block shifts ---


def test_projection_reproduces_the_fit_and_zero_is_the_mean():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        rng.normal(size=(40, 4)) * [1, 3, 0.5, 2] + [0, 5, -1, 2],
        columns=list("abcd"),
    )
    pcs = run_pca_on_covariates(
        raw, n_components=2, svd_solver="full"
    )
    np.testing.assert_allclose(
        project_onto_pca(raw, pcs).to_numpy(),
        pcs.to_numpy(),
        atol=1e-10,
    )
    # A row missing everything is the mean: the origin of the basis.
    empty = pd.DataFrame(np.nan, index=["x"], columns=list("abcd"))
    np.testing.assert_allclose(
        project_onto_pca(empty, pcs), 0.0, atol=1e-12
    )
    # A row missing one column equals the full row with that column
    # set to its mean.
    row = raw.iloc[[0]].copy()
    at_mean = row.copy()
    at_mean["b"] = raw["b"].mean()
    row["b"] = np.nan
    np.testing.assert_allclose(
        project_onto_pca(row, pcs),
        project_onto_pca(at_mean, pcs),
        atol=1e-10,
    )


@pytest.mark.parametrize("theta", [None, 3.0])
def test_fit_fallback_shifts_recovers_planted_shifts(theta):
    rng = np.random.default_rng(1)
    n = 4000
    M = np.column_stack(
        [rng.random(n) < 0.4, rng.random(n) < 0.3]
    ).astype(float)
    d_true = np.array([-1.0, 0.4])
    eta = rng.normal(0, 0.3, n)
    b_s, b_ns = rng.uniform(0.5, 3, n), rng.uniform(1, 6, n)
    r = rng.gamma(theta, 1 / theta, n) if theta else np.ones(n)
    scale = np.exp(eta + M @ d_true) * r
    n_s = rng.poisson(b_s * scale)
    n_ns = rng.poisson(b_ns * scale * np.exp(0.15))
    d = fit_fallback_shifts(
        n_s, n_ns, b_s, b_ns, eta, M, delta=0.15, theta=theta
    )
    np.testing.assert_allclose(d, d_true, atol=0.08)


def test_fit_fallback_shifts_is_bounded():
    """A block whose genes never mutate would run to -inf."""
    d = fit_fallback_shifts(
        np.zeros(5), np.zeros(5), np.ones(5), np.ones(5),
        np.zeros(5), np.ones((5, 1)), bound=2.0,
    )  # fmt: skip
    assert d[0] == pytest.approx(-2.0)


def _pca_fallback_model(tmp_path):
    """The synthetic channel model with a PCA-reduced covariate matrix
    in which ENSG_C misses block "b"."""
    model = _model_with_channels(tmp_path)
    model.dataset.compute_gene_counts_channels()
    raw = pd.DataFrame(
        {
            "a1": [0.5, -0.5, 0.2],
            "a2": [1.0, 3.0, 2.5],
            "b1": [2.0, 1.0, np.nan],
        },
        index=["ENSG_A", "ENSG_B", "ENSG_C"],
    )
    model.assign_cov_matrix(
        raw,
        run_pca=True,
        pca_kwargs={"n_components": 1, "svd_solver": "full"},
        missing_blocks={"a1": "a", "a2": "a", "b1": "b"},
    )
    return model


def test_pca_fallback_projection_and_indicators(tmp_path):
    model = _pca_fallback_model(tmp_path)
    assert list(model.cov_matrix.index) == ["ENSG_A", "ENSG_B"]
    assert list(model.covariate_fallback_genes) == [_FALLBACK]
    assert list(model._fallback_cov.index) == [_FALLBACK]
    ind = model._fallback_indicators.loc[_FALLBACK]
    assert ind["b"] == 1.0 and ind["a"] == 0.0


def test_pca_fallback_rate_is_projection_plus_shift_plus_delta(
    tmp_path,
):
    model = _pca_fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    # One fallback gene is below min_genes, so no shift is fitted.
    assert model._fallback_shifts is None
    c = np.asarray(model.cov_effects)
    x = float(model._fallback_cov.loc[_FALLBACK, "PC1"])
    delta = model._rg_delta_intercept
    np.testing.assert_allclose(
        model.compute_channel_mu_gs("nonsyn").loc[_FALLBACK],
        model.base_mus_nonsyn.loc[_FALLBACK]
        * np.exp(c[0] + c[1] * x + delta),
        rtol=1e-5,
    )

    shifts = model.estimate_fallback_shifts(min_genes=1)
    assert list(shifts.index) == ["b"]
    np.testing.assert_allclose(
        model.compute_channel_mu_gs("nonsyn").loc[_FALLBACK],
        model.base_mus_nonsyn.loc[_FALLBACK]
        * np.exp(c[0] + c[1] * x + shifts["b"] + delta),
        rtol=1e-5,
    )
    # r_g's expectation for the fallback gene uses the same term.
    expected_silent, _ = model._rg_expectations(include_fallback=True)
    assert expected_silent[_FALLBACK] == pytest.approx(
        model._rg_fallback_statistics["baseline_silent"][_FALLBACK]
        * np.exp(c[0] + c[1] * x + shifts["b"])
    )


def test_pca_fallback_leaves_complete_genes_alone(tmp_path):
    """Fitting the shifts must not move a complete gene's rate."""
    model = _pca_fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    before = model.compute_channel_mu_gs("nonsyn").loc[
        ["ENSG_A", "ENSG_B"]
    ]
    model.estimate_fallback_shifts(min_genes=1)
    after = model.compute_channel_mu_gs("nonsyn").loc[
        ["ENSG_A", "ENSG_B"]
    ]
    pd.testing.assert_frame_equal(before, after)


def test_pca_fallback_zero_column_matrix_uses_intercept(tmp_path):
    """The nested ladder swaps in a zero-column matrix; the fallback
    gene then sits at the intercept (plus its shift), like everyone.
    """
    model = _pca_fallback_model(tmp_path)
    model.cov_matrix = model.cov_matrix.iloc[:, :0]
    model.estimate_channel_rg_cov_effects(
        sample="MAP", fit_rg=False, separate_c=False
    )
    eta = model._fallback_eta()
    assert eta[_FALLBACK] == pytest.approx(
        float(model.cov_effects[0])
    )


def test_pca_fallback_draws_match_the_point_rate(tmp_path):
    model = _pca_fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="full")
    model.estimate_fallback_shifts(min_genes=1)
    draws = model.compute_mu_g_posterior_draws(
        _FALLBACK, r_g_variant="none"
    )
    point = model.compute_channel_mu_gs("nonsyn").loc[_FALLBACK]
    # Centred on the point rate (posterior mean c) to within the spread
    # of the draws.
    ratio = np.log(draws.to_numpy() / point.to_numpy()[None, :])[:, 0]
    assert abs(ratio.mean()) < 3 * ratio.std() + 1e-6


def test_pca_fallback_survives_save_and_load(tmp_path):
    model = _pca_fallback_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    model.estimate_channel_rg_cov_effects(sample="MAP")
    model.estimate_fallback_shifts(min_genes=1)
    before = model.compute_channel_mu_gs("nonsyn")

    model.save_model(tmp_path / "model")
    loaded = Model.load_model(tmp_path / "model")
    pd.testing.assert_frame_equal(
        loaded._fallback_cov, model._fallback_cov
    )
    pd.testing.assert_series_equal(
        loaded._fallback_shifts,
        model._fallback_shifts,
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        loaded.compute_channel_mu_gs("nonsyn"), before
    )


def test_gamma_names_the_missing_blocks(tmp_path, monkeypatch):
    model = _pca_fallback_model(tmp_path)
    model.estimate_channel_rg_cov_effects(sample="MAP")
    model.dataset.genes_present_non_silent = pd.DataFrame(
        [[1, 0, 1]], index=[_FALLBACK], columns=["T1", "T2", "T3"]
    )
    _fake_gamma(monkeypatch, {})
    result = model._estimate_gamma_gene(_FALLBACK, store=False)
    assert result.posterior.attrs["covariate_fallback"] == 1
    assert result.posterior.attrs["covariate_fallback_blocks"] == "b"
