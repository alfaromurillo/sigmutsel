"""Tests for the gene selection call and the empirical-Bayes gamma.

The pieces are checked against direct computation: the tabulated
likelihood against the link summed tumor by tumor, the mid-p against
enumerating every outcome, BH against its definition, the prior's
gradient against finite differences, and the guard against a fit it
must correct. The model-level table is checked against the per-gene
gamma path and survives a save and load.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from sigmutsel import selection as sel
from sigmutsel.models import Model, MutationDataset

from .test_channel_cov_effects import _model_with_channels


# ---- the likelihood ----------------------------------------------------
def _direct_loglik(u, sig, p, total, present, phi, floor):
    t = u + sig
    m = np.exp(t) * total
    big_l = -phi * np.log1p(m / phi) if phi else -m
    z = p[None, :] * big_l[:, None]
    hz = (
        np.logaddexp(np.log(floor), np.log1p(-floor) + z)
        if floor
        else z
    )
    return np.where(
        present[None, :],
        np.log(-np.expm1(np.minimum(hz, -1e-300))),
        hz,
    ).sum(axis=1)


@pytest.mark.parametrize("phi", [None, 3.0, 50.0])
@pytest.mark.parametrize("floor", [1e-3, 0.0])
def test_tabulated_likelihood_is_the_link_summed_over_tumors(
    phi, floor
):
    rng = np.random.default_rng(1)
    b = rng.gamma(0.5, 1.0, 120)
    b[:5] = 0.0
    total = b.sum()
    p = b / total
    present = (rng.random(120) < 0.05) & (p > 0)
    sig = np.log(rng.gamma(2.0, 1.0)) - 4.0 + rng.normal(0, 0.2, 30)
    lik = sel.FactoredGeneLikelihood(
        p, total, present, sig, phi=phi, absence_floor=floor
    )
    ll = lik.loglik()
    grid = sel.flat_grid()
    for ui in (50, 300, 500, 650):
        np.testing.assert_allclose(
            ll[:, ui],
            _direct_loglik(
                grid[ui], sig, p, total, present, phi, floor
            ),
            rtol=2e-3,
            atol=2e-3,
        )


def test_marginal_averages_the_draws():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(40))
    lik = sel.FactoredGeneLikelihood(
        p, 0.5, p > 0.05, rng.normal(0, 0.3, 20), phi=10.0
    )
    mask = sel.eb_mask()
    ll = lik.loglik(mask)
    expected = np.log(np.exp(ll).mean(axis=0))
    np.testing.assert_allclose(
        lik.marginal(mask), expected, rtol=1e-10
    )


# ---- the null test -----------------------------------------------------
def test_mid_p_matches_enumeration():
    rng = np.random.default_rng(3)
    la = -rng.gamma(1.0, 0.4, (3, 9))
    w = np.array([0.2, 0.5, 0.3])
    for n in range(6):
        p_gt = p_eq = 0.0
        for d in range(3):
            pa = np.exp(la[d])
            for out in itertools.product([0, 1], repeat=9):
                o = np.array(out)
                pr = np.prod(np.where(o == 1, 1 - pa, pa))
                k = o.sum()
                p_gt += w[d] * pr * (k > n)
                p_eq += w[d] * pr * (k == n)
        np.testing.assert_allclose(
            sel.upper_mid_p(la, n, w), p_gt + 0.5 * p_eq, rtol=1e-10
        )


def test_r_nodes_integrate_the_gamma():
    r, w = sel.r_posterior_nodes(12.0, 10.0, 256)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(r @ w, 1.2, rtol=1e-3)


def test_dispersed_null_reduces_to_poisson():
    mu = np.array([[0.1, 0.2, 0.05]])
    np.testing.assert_allclose(
        sel.log_absent_null(mu, 1e12, mu / mu.sum()), -mu, rtol=1e-6
    )
    np.testing.assert_array_equal(sel.log_absent_null(mu, None), -mu)


def test_benjamini_hochberg():
    p = np.array([0.01, 0.04, np.nan, 0.03, 0.2])
    q = sel.benjamini_hochberg(p)
    assert np.isnan(q[2])
    # m = 4 tested: sorted 0.01, 0.03, 0.04, 0.2
    np.testing.assert_allclose(
        q[[0, 3, 1, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.2], rtol=1e-10
    )


# ---- the prior ---------------------------------------------------------
def _synthetic_logliks(rng, n, centre=0.0, spread=0.02, frac=0.9):
    """Gaussian log-likelihoods of genes: most near ``centre``."""
    grid = sel.eb_grid()
    truth = np.where(
        rng.random(n) < frac,
        rng.normal(centre, spread, n),
        rng.uniform(0.5, 5.0, n),
    )
    sd = rng.uniform(0.2, 1.5, n)
    obs = truth + rng.normal(0, sd)
    return -0.5 * ((grid[None, :] - obs[:, None]) / sd[:, None]) ** 2


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    grid = sel.eb_grid()
    q = sel.slab_basis(grid)
    lik = sel._as_lik(_synthetic_logliks(rng, 50))
    a = rng.normal(0, 1, q.shape[1])
    spike = np.array([0.7, 0.05, np.log(0.1)])
    _, ga, gs = sel._nll_and_grad(lik, grid, q, a, spike)
    eps = 1e-6
    for i in range(len(a)):
        e = np.zeros_like(a)
        e[i] = eps
        fd = (
            sel._nll_and_grad(lik, grid, q, a + e, spike)[0]
            - sel._nll_and_grad(lik, grid, q, a - e, spike)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(ga[i], fd, rtol=1e-4, atol=1e-6)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        fd = (
            sel._nll_and_grad(lik, grid, q, a, spike + e)[0]
            - sel._nll_and_grad(lik, grid, q, a, spike - e)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(gs[i], fd, rtol=1e-4, atol=1e-6)


def test_fit_puts_the_spike_where_the_neutral_genes_are():
    rng = np.random.default_rng(5)
    prior = sel.fit_spike_slab(
        _synthetic_logliks(rng, 3000, centre=0.1)
    )
    assert abs(prior.spike_centre - 0.1) < 0.05
    assert prior.spike_weight > 0.5
    np.testing.assert_allclose(np.exp(prior.log_density).sum(), 1.0)
    assert not prior.guard_triggered


def test_guard_refits_a_spike_that_excludes_one():
    # every gene sharply at u = 0.06: the free spike sits there, narrow
    grid = sel.eb_grid()
    ll = np.tile(-0.5 * ((grid - 0.06) / 0.01) ** 2, (400, 1))
    free = sel.fit_spike_slab(ll, guard=False)
    assert sel.prior_excludes_one(free.log_density, grid)
    guarded = sel.fit_spike_slab(ll)
    assert guarded.guard_triggered
    assert guarded.spike_centre == 0.0
    assert guarded.unguarded["spike_centre"] == pytest.approx(
        free.spike_centre
    )
    assert not sel.prior_excludes_one(guarded.log_density, grid)


def test_spike_trigger_catches_what_the_whole_prior_misses():
    # a narrow spike at e^0.047 whose interval excludes one, inside a
    # prior whose smooth component's tails stretch its interval over one
    grid = sel.eb_grid()
    lg = sel.spike_slab_log_density(
        grid, np.zeros(sel.SLAB_DF), 0.95, 0.047, 0.027
    )
    assert not sel.prior_excludes_one(lg, grid)
    assert sel.spike_excludes_one(0.047, 0.027, grid)
    assert not sel.spike_excludes_one(0.011, 0.02, grid)
    prior = sel.SpikeSlabPrior(
        grid, lg, 0.95, 0.047, 0.027, np.zeros(sel.SLAB_DF)
    )
    assert sel.guard_triggers(prior)


def test_guard_leaves_a_prior_covering_one_alone():
    rng = np.random.default_rng(6)
    ll = _synthetic_logliks(rng, 2000, centre=0.0, spread=0.05)
    a = sel.fit_spike_slab(ll, guard=False)
    b = sel.fit_spike_slab(ll)
    assert not b.guard_triggered
    np.testing.assert_allclose(a.log_density, b.log_density)


def test_pooled_slab_is_shared_and_spikes_are_not():
    rng = np.random.default_rng(7)
    lls = {
        "A": _synthetic_logliks(rng, 1500, centre=0.0),
        "B": _synthetic_logliks(rng, 1500, centre=0.15),
    }
    coef, priors = sel.fit_pooled_spike_slab(lls, guard=False)
    np.testing.assert_allclose(priors["A"].slab_coef, coef)
    np.testing.assert_allclose(priors["B"].slab_coef, coef)
    assert priors["B"].spike_centre > priors["A"].spike_centre + 0.08
    # B's narrow spike at e^0.15 excludes one: guarded, it is refitted
    # at one on the pooled slab, and A is left as it was
    _, guarded = sel.fit_pooled_spike_slab(lls)
    assert guarded["B"].guard_triggered
    assert guarded["B"].spike_centre == 0.0
    np.testing.assert_allclose(guarded["B"].slab_coef, coef)
    assert not guarded["A"].guard_triggered
    np.testing.assert_allclose(
        guarded["A"].log_density, priors["A"].log_density
    )
    # one cohort with the slab fixed refits only its spike
    again = sel.fit_spike_slab(lls["A"], slab_coef=coef)
    assert again.slab_pooled
    assert again.spike_centre == pytest.approx(
        priors["A"].spike_centre, abs=0.02
    )


def test_prior_round_trips():
    rng = np.random.default_rng(8)
    prior = sel.fit_spike_slab(_synthetic_logliks(rng, 500))
    back = sel.SpikeSlabPrior.from_dict(prior.to_dict())
    np.testing.assert_allclose(back.grid, prior.grid)
    np.testing.assert_allclose(
        back.log_density, prior.log_density, atol=1e-7
    )
    assert back.spike_sd == prior.spike_sd


def test_posterior_summary_of_a_normal():
    grid = sel.flat_grid()
    mass = np.exp(-0.5 * ((grid - 1.0) / 0.5) ** 2)
    s = sel.posterior_summary(mass, grid)
    assert s["median"] == pytest.approx(np.e, rel=1e-3)
    assert np.log(s["lower"]) == pytest.approx(
        1.0 - 1.598 * 0.5, abs=0.01
    )
    assert s["p_above"] == pytest.approx(
        0.5 * (1 + math.erf((1.0 - np.log(2)) / (0.5 * 2**0.5))),
        abs=5e-3,  # P(gamma > 2) counts grid points, half a cell either way
    )


# ---- the model's table -------------------------------------------------
@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sel")
    model = _model_with_channels(tmp / "work")
    model.dataset.compute_gene_counts_channels()
    model.dataset.save_dataset(tmp / "ds")
    model.dataset = MutationDataset.load_dataset(tmp / "ds")
    model.estimate_channel_rg_cov_effects(sample="full")
    return model, tmp


def test_table_likelihood_is_the_gene_gamma_likelihood(fitted):
    # with r_g left out and every draw used, the factored draws are the
    # gene gamma's own (compute_mu_g_posterior_draws), so the tabulated
    # likelihood must equal the link summed over the fit's tumors
    model, _ = fitted
    gene = pd.Index(model._rg_statistics["genes"])[0]
    n = int(
        model.cov_effects_posteriors.posterior.sizes["draw"]
    ) * int(model.cov_effects_posteriors.posterior.sizes["chain"])
    liks = model.gene_selection_likelihoods(
        [gene],
        n_draws=n,
        gene_tumor_dispersion=None,
        r_g_variant="none",
    )
    lik = liks.likelihoods[0]
    draws = model.compute_mu_g_posterior_draws(
        gene, r_g_variant="none"
    )
    pres = (
        model.dataset.genes_present_non_silent.reindex(
            index=[gene], columns=draws.columns, fill_value=0
        ).iloc[0]
        == 1
    ).to_numpy()
    mu = draws.to_numpy()
    grid = sel.flat_grid()
    for ui in (200, 380, 420, 500):
        z = -np.exp(grid[ui]) * mu
        hz = np.logaddexp(np.log(1e-3), np.log1p(-1e-3) + z)
        direct = np.where(
            pres[None, :], np.log(-np.expm1(hz)), hz
        ).sum(axis=1)
        np.testing.assert_allclose(
            lik.loglik()[:, ui], direct, rtol=1e-3, atol=1e-3
        )


def test_null_test_matches_the_direct_poisson_binomial(fitted):
    model, _ = fitted
    genes = pd.Index(model._rg_statistics["genes"])
    test = model.gene_selection_null_test(
        genes, gene_tumor_dispersion=None
    )
    gene = genes[0]
    nu = model.compute_channel_mu_gs("nonsyn").loc[gene]
    pres = model.dataset.genes_present_non_silent.reindex(
        index=[gene], columns=nu.index, fill_value=0
    ).iloc[0]
    stats = model._rg_statistics_with_fallback()
    es, _ = model._rg_expectations(include_fallback=True)
    th = float(model.rg_theta)
    r, w = sel.r_posterior_nodes(
        th + stats["counts_silent"].loc[gene], th + es.loc[gene], 256
    )
    la = -r[:, None] * nu.to_numpy()[None, :]
    assert test.loc[gene, "p_mid"] == pytest.approx(
        sel.upper_mid_p(la, int((pres == 1).sum()), w), rel=1e-10
    )


def test_table_is_saved_and_loaded(fitted):
    model, tmp = fitted
    tab = model.gene_selection_table(gene_tumor_dispersion=None)
    assert set(tab.columns) >= {"p_mid", "q", "called", "eb_median"}
    assert model.selection_prior["fdr"] == 0.1
    model.save_model(tmp / "model")
    loaded = Model.load_model(tmp / "model")
    pd.testing.assert_frame_equal(loaded.selection_table, tab)
    assert loaded.selection_prior == model.selection_prior
    # a model saved without a table loads with none
    model.selection_table = None
    model.selection_prior = None
    model.save_model(tmp / "model2")
    assert Model.load_model(tmp / "model2").selection_table is None


# ---- in parallel ---------------------------------------------------------
def test_table_is_the_same_in_parallel(fitted):
    # the per-gene work is deterministic and r_g's draw order is taken
    # in gene order in the parent, so the table cannot depend on n_jobs
    model, _ = fitted
    kwargs = {"gene_tumor_dispersion": None, "store": False}
    serial = model.gene_selection_table(n_jobs=1, **kwargs)
    forked = model.gene_selection_table(n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(serial, forked, check_exact=True)


def test_gammas_are_the_same_in_parallel(fitted):
    # seeded per item, a gamma does not depend on the processes or on
    # the order the items ran in
    model, _ = fitted
    genes = list(pd.Index(model._rg_statistics["genes"])[:3])
    requests = [
        {
            "item": g,
            "level": "gene",
            "use_mu_posterior": True,
            "mu_uncertainty": "draws",
            "r_g_variant": "evaluation",
        }
        for g in genes
    ]

    def values(results):
        return [
            np.asarray(r.posterior["gamma"].values).ravel()
            for _, r in results
        ]

    serial = model.estimate_gammas(requests, seed=11, n_jobs=1)
    keys = [k for k, _ in serial]
    assert keys == genes
    forked = model.estimate_gammas(requests[::-1], seed=11, n_jobs=2)
    assert [k for k, _ in forked] == keys[::-1]
    for a, b in zip(values(serial), values(forked)[::-1]):
        np.testing.assert_array_equal(a, b)
    for k, (_, r) in zip(keys[::-1], forked):
        assert model.gammas[k] is r
    # a failing item comes back as its exception, the rest are fitted
    bad = model.estimate_gammas(
        [{"item": "NOT_A_GENE", "level": "gene"}, *requests[:1]],
        seed=11,
        n_jobs=2,
    )
    assert bad[0][0] is None and isinstance(bad[0][1], Exception)
    assert bad[1][0] == keys[0]


def test_hold_out_lookup_matches_a_filter():
    db = pd.DataFrame(
        {
            "ensembl_gene_id": ["G1", "G2", "G1", "G1", "G3"],
            "Tumor_Sample_Barcode": ["t2", "t1", "t1", "t2", "t3"],
        }
    )
    ds = MutationDataset.__new__(MutationDataset)
    ds._homdel_db = db
    for g in ("G1", "G2", "G3", "G9"):
        direct = pd.Index(
            db.loc[
                db["ensembl_gene_id"] == g, "Tumor_Sample_Barcode"
            ].unique()
        )
        assert list(ds.homdel_hits(g)) == list(direct)
    # a new table object rebuilds the index
    ds._homdel_db = db.iloc[:2]
    assert list(ds.homdel_hits("G1")) == ["t2"]
