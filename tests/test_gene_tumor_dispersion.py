"""Tests for gene-tumor dispersion: the allocation likelihood, the
fitted trend in ``phi``, and how ``Model`` stores and uses it.

The likelihood is checked against its own limits and against
enumeration; the fitters against planted values; the ``Model`` wiring
for persistence and argument handling. The gamma likelihood it feeds
is tested in ``test_estimate_gammas.py``.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import special as spc

from sigmutsel.gene_tumor_dispersion import (
    dirichlet_multinomial_logpmf,
    fit_gene_tumor_dispersion_trend,
    fit_phi,
    phi_for_count,
)
from sigmutsel.models import Model, MutationDataset

from .test_channel_cov_effects import _model_with_channels


def _multinomial_logpmf(counts, p):
    return (
        spc.gammaln(counts.sum(axis=1) + 1)
        - spc.gammaln(counts + 1).sum(axis=1)
        + (counts * np.log(p)).sum(axis=1)
    )


def test_infinite_and_huge_phi_give_the_multinomial():
    rng = np.random.default_rng(0)
    w = rng.gamma(1.0, 1.0, (40, 12))
    p = w / w.sum(axis=1, keepdims=True)
    counts = rng.multinomial(rng.poisson(6, 40), p).astype(float)
    want = _multinomial_logpmf(counts, p)
    got_inf = dirichlet_multinomial_logpmf(counts, np.log(w), np.inf)
    got_big = dirichlet_multinomial_logpmf(counts, np.log(w), 1e12)
    np.testing.assert_allclose(got_inf, want, atol=1e-10)
    np.testing.assert_allclose(got_big, want, atol=1e-8)


def test_masses_sum_to_one_over_the_support():
    p = np.array([0.5, 0.3, 0.2])
    n, phi = 5, 3.7
    support = [
        c
        for c in itertools.product(range(n + 1), repeat=3)
        if sum(c) == n
    ]
    counts = np.array(support, dtype=float)
    mass = np.exp(
        dirichlet_multinomial_logpmf(
            counts, np.log(np.broadcast_to(p, counts.shape)), phi
        )
    ).sum()
    assert mass == pytest.approx(1.0, abs=1e-12)


def _dm_draws(rng, weights, totals, phi_per_gene):
    p = weights / weights.sum(axis=1, keepdims=True)
    out = np.zeros_like(weights)
    for g in range(len(totals)):
        q = rng.dirichlet(phi_per_gene[g] * p[g])
        out[g] = rng.multinomial(totals[g], q)
    return out


def test_fit_phi_recovers_a_planted_value():
    rng = np.random.default_rng(1)
    w = rng.gamma(1.0, 1.0, (3000, 60))
    totals = rng.poisson(8, 3000)
    counts = _dm_draws(rng, w, totals, np.full(3000, 40.0))
    phi, ceiling = fit_phi(counts, np.log(w))
    assert not ceiling
    assert 25 < phi < 65


def test_trend_recovers_a_planted_slope():
    """phi rising with a gene's count is recovered as a slope."""
    rng = np.random.default_rng(2)
    n_genes = 12000
    w = rng.gamma(1.0, 1.0, (n_genes, 80))
    totals = np.maximum(
        2, np.rint(np.exp(rng.normal(1.8, 0.9, n_genes)))
    ).astype(int)
    a, b = np.log(30.0), 0.6
    phi = np.exp(a + b * np.log(totals))
    counts = _dm_draws(rng, w, totals, phi)
    trend = fit_gene_tumor_dispersion_trend(counts, w)
    assert trend["method"] == "trend"
    assert trend["slope"] == pytest.approx(b, abs=0.25)
    assert phi_for_count(trend, 20) == pytest.approx(
        np.exp(a + b * np.log(20)), rel=0.5
    )


def test_trend_falls_back_to_pooled_with_few_strata():
    rng = np.random.default_rng(3)
    w = rng.gamma(1.0, 1.0, (40, 10))
    counts = _dm_draws(rng, w, np.full(40, 3), np.full(40, 20.0))
    trend = fit_gene_tumor_dispersion_trend(counts, w, folds=0)
    assert trend["method"] == "pooled"
    assert trend["slope"] == 0.0
    assert phi_for_count(trend, 100) == pytest.approx(
        trend["pooled_phi"]
    )


def test_per_gene_phi_array_matches_scalar_calls():
    rng = np.random.default_rng(4)
    w = rng.gamma(1.0, 1.0, (30, 9))
    counts = rng.multinomial(
        rng.poisson(5, 30), w / w.sum(1, keepdims=True)
    )
    phis = np.where(
        np.arange(30) % 3 == 0, np.inf, 10.0 + np.arange(30)
    )
    got = dirichlet_multinomial_logpmf(counts, np.log(w), phis)
    want = np.array(
        [
            dirichlet_multinomial_logpmf(
                counts[i : i + 1], np.log(w[i : i + 1]), phis[i]
            )[0]
            for i in range(30)
        ]
    )
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_negative_slope_is_floored():
    """phi falling with count, as sparse high-count strata produce."""
    rng = np.random.default_rng(5)
    n_genes = 12000
    w = rng.gamma(1.0, 1.0, (n_genes, 80))
    totals = np.maximum(
        2, np.rint(np.exp(rng.normal(1.8, 0.9, n_genes)))
    ).astype(int)
    phi = np.exp(np.log(300.0) - 0.8 * np.log(totals))
    counts = _dm_draws(rng, w, totals, phi)
    trend = fit_gene_tumor_dispersion_trend(counts, w, folds=0)
    assert trend["slope_floored"]
    assert trend["slope"] == 0.0


def test_no_dispersion_switches_itself_off():
    """Multinomial data: the held-out check must not keep a finite phi,
    so gamma falls back to the dispersion-free likelihood."""
    rng = np.random.default_rng(6)
    n_genes = 8000
    w = rng.gamma(1.0, 1.0, (n_genes, 60))
    totals = np.maximum(
        2, np.rint(np.exp(rng.normal(1.6, 0.8, n_genes)))
    ).astype(int)
    counts = rng.multinomial(totals, w / w.sum(axis=1, keepdims=True))
    trend = fit_gene_tumor_dispersion_trend(counts.astype(float), w)
    assert trend["method"] == "none"
    assert phi_for_count(trend, 10) == np.inf


def test_real_dispersion_passes_the_held_out_check():
    rng = np.random.default_rng(7)
    n_genes = 8000
    w = rng.gamma(1.0, 1.0, (n_genes, 60))
    totals = np.maximum(
        2, np.rint(np.exp(rng.normal(1.6, 0.8, n_genes)))
    ).astype(int)
    counts = _dm_draws(rng, w, totals, np.full(n_genes, 30.0))
    trend = fit_gene_tumor_dispersion_trend(counts, w)
    assert trend["method"] in ("trend", "pooled")
    assert trend["cv_gain"] > 0


# ---------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------


def _fitted_model(tmp_path):
    model = _model_with_channels(tmp_path)
    model.dataset.compute_gene_counts_channels()
    model.estimate_gene_tumor_dispersion(min_genes=1)
    return model


def test_estimate_gene_tumor_dispersion_stores_a_trend(tmp_path):
    model = _fitted_model(tmp_path)
    trend = model.gene_tumor_dispersion_trend
    assert set(trend) >= {"intercept", "slope", "method", "strata"}
    assert np.isfinite(trend["intercept"])


def test_gene_tumor_dispersion_trend_survives_save_and_load(tmp_path):
    model = _fitted_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    out = tmp_path / "model"
    model.save_model(out)
    loaded = Model.load_model(out)
    assert (
        loaded.gene_tumor_dispersion_trend
        == model.gene_tumor_dispersion_trend
    )


def test_model_saved_before_the_rename_still_loads(tmp_path):
    import json

    model = _fitted_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    out = tmp_path / "model"
    model.save_model(out)
    manifest_path = next(
        p
        for p in out.glob("*.json")
        if "gene_tumor_dispersion_trend" in p.read_text()
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["cell_dispersion_trend"] = manifest.pop(
        "gene_tumor_dispersion_trend"
    )
    manifest_path.write_text(json.dumps(manifest))
    loaded = Model.load_model(out)
    assert (
        loaded.gene_tumor_dispersion_trend
        == model.gene_tumor_dispersion_trend
    )
    with pytest.warns(DeprecationWarning):
        assert (
            loaded.cell_dispersion_trend
            == model.gene_tumor_dispersion_trend
        )


def test_fitted_phi_uses_the_genes_own_count(tmp_path):
    model = _fitted_model(tmp_path)
    model.gene_tumor_dispersion_trend = {
        "intercept": np.log(50.0),
        "slope": 0.5,
        "method": "trend",
    }
    counts = model.dataset.genes_counts_non_silent
    kept = pd.Series(True, index=counts.columns)
    phi = model._resolve_gene_tumor_dispersion(
        "fitted", "ENSG_B", kept
    )
    n = float(counts.loc["ENSG_B"].sum())
    assert phi == pytest.approx(50.0 * max(n, 2.0) ** 0.5)
    assert (
        model._resolve_gene_tumor_dispersion(None, "ENSG_B", kept)
        is None
    )
    assert (
        model._resolve_gene_tumor_dispersion(7, "ENSG_B", kept) == 7.0
    )
    model.gene_tumor_dispersion_trend = {
        **model.gene_tumor_dispersion_trend,
        "method": "none",
    }
    assert (
        model._resolve_gene_tumor_dispersion("fitted", "ENSG_B", kept)
        is None
    )


def test_gene_tumor_dispersion_argument_errors(tmp_path):
    model = _model_with_channels(tmp_path)
    kept = pd.Series(True, index=["T1", "T2", "T3"])
    with pytest.raises(
        ValueError, match="estimate_gene_tumor_dispersion"
    ):
        model._resolve_gene_tumor_dispersion("fitted", "ENSG_B", kept)
    with pytest.raises(ValueError, match="'fitted'"):
        model._resolve_gene_tumor_dispersion(
            "sometimes", "ENSG_B", kept
        )
    with pytest.raises(ValueError, match="non_silent"):
        model.estimate_gamma(
            "ENSG_B",
            level="gene",
            non_silent=False,
            gene_tumor_dispersion=10.0,
        )


def test_gene_tumor_shapes_sum_to_phi(tmp_path):
    model = _fitted_model(tmp_path)
    shapes = model._gene_tumor_shapes(
        "ENSG_A", 40.0, pd.Index(["T1", "T2"])
    )
    assert shapes.sum() == pytest.approx(40.0)
    assert (shapes > 0).all()


def test_dispersion_baseline_falls_back_to_merged(tmp_path):
    """A model without the channel split (the no_cov models) fits phi
    against its merged baseline."""
    model = _model_with_channels(tmp_path)
    model._base_mus_nonsyn = None
    model.dataset.compute_gene_counts_channels()
    assert model._dispersion_baseline() is model._base_mus
    model.estimate_gene_tumor_dispersion(min_genes=1)
    assert model.gene_tumor_dispersion_trend is not None


def test_variant_gene_lookup(tmp_path):
    model = _model_with_channels(tmp_path)
    assert model._variant_gene_id("X p.A1B") in {
        "ENSG_A",
        "ENSG_B",
        "ENSG_C",
    }
    with pytest.raises(ValueError, match="gene of variant"):
        model._variant_gene_id("NOPE p.Z9Z")


def test_pre_rename_names_still_work():
    """``cell_*`` names are deprecated aliases, not removed."""
    import importlib
    import sys

    sys.modules.pop("sigmutsel.cell_dispersion", None)
    with pytest.warns(DeprecationWarning):
        old = importlib.import_module("sigmutsel.cell_dispersion")
    assert old.phi_for_count is phi_for_count
    assert (
        old.fit_cell_dispersion_trend
        is fit_gene_tumor_dispersion_trend
    )
