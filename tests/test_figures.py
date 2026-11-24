"""Tests for the SBS96 profile plot."""

import itertools

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sigmutsel.figures import (
    SBS96_SUBSTITUTIONS,
    plot_signature_profile,
    sbs96_order,
)

BASES = "ACGT"


def _all_types():
    return [
        f"{a}[{s}]{b}"
        for s in SBS96_SUBSTITUTIONS
        for a, b in itertools.product(BASES, BASES)
    ]


def test_sbs96_order_groups_by_substitution_then_context():
    types = _all_types()
    shuffled = list(np.random.default_rng(0).permutation(types))
    assert sbs96_order(shuffled) == types


def test_sbs96_order_rejects_malformed_types():
    with pytest.raises(ValueError):
        sbs96_order(["ACG"])
    with pytest.raises(ValueError):
        sbs96_order(["A[G>T]C"])


def test_plot_draws_96_bars_in_order_without_title():
    import matplotlib.pyplot as plt

    types = _all_types()
    values = pd.Series(np.arange(96) / 4560.0, index=types)
    ax = plot_signature_profile(values.sample(frac=1, random_state=1))
    heights = [p.get_height() for p in ax.containers[0]]
    assert np.allclose(heights, values.to_numpy())
    assert ax.get_title() == ""
    assert ax.get_xticklabels()[0].get_text() == "ACA"
    plt.close("all")


def test_plot_sets_the_given_ylabel():
    import matplotlib.pyplot as plt

    values = pd.Series(1 / 96, index=_all_types())
    ax = plot_signature_profile(
        values, ylabel="Probabilidad", context_labels=False
    )
    assert ax.get_ylabel() == "Probabilidad"
    assert len(ax.get_xticks()) == 0
    plt.close("all")


def test_plot_rejects_an_incomplete_profile():
    values = pd.Series(1 / 95, index=_all_types()[:95])
    with pytest.raises(ValueError):
        plot_signature_profile(values)


# --- the other figures: they run on the pinned versions, save where
# --- asked and return their figure


def test_comparison_with_observed_saves_where_asked(tmp_path):
    from matplotlib.figure import Figure

    from sigmutsel.figures import comparison_with_observed

    genes = [f"G{i}" for i in range(20)]
    expected = pd.Series(np.linspace(1, 20, 20), index=genes)
    observed = expected + np.random.default_rng(0).normal(0, 1, 20)
    fig = comparison_with_observed(
        expected, observed, save=tmp_path / "obs.png", show=False
    )
    assert isinstance(fig, Figure)
    assert (tmp_path / "obs.png").stat().st_size > 0


def test_plot_posteriors_vs_counts_takes_the_interval_mass():
    import xarray as xr

    from sigmutsel.figures import plot_posteriors_vs_counts

    rng = np.random.default_rng(0)
    results = {
        g: xr.Dataset(
            {"gamma": (("chain", "draw"), rng.gamma(5, 1, (2, 500)))}
        )
        for g in ("TP53", "TTN")
    }
    # an object with .posterior, as estimate_gamma's results have
    results = {
        g: type("R", (), {"posterior": ds})()
        for g, ds in results.items()
    }
    fig = plot_posteriors_vs_counts(
        results, {"TP53": 30, "TTN": 12}, {"TTN"}, prob=0.89
    )
    lines = fig.axes[0].collections  # the interval segments
    assert len(lines) == 2


def test_model_plot_signature_correlations_with_given_covariates(
    tmp_path,
):
    from sigmutsel.models import Model
    from tests.test_implicit_baselines import _full_context_dataset

    model = Model(_full_context_dataset(tmp_path))
    covariates = pd.DataFrame(
        {"late": [0.9, 0.1, 0.5], "expressed": [1.0, 3.0, 2.0]},
        index=["ENSG_A", "ENSG_B", "ENSG_C"],
    )
    _, ax1, _ = model.plot_signature_correlations(
        covariates=covariates, show=False
    )
    labels = [t.get_text() for t in ax1.get_legend().get_texts()]
    assert labels == ["late", "expressed", "Mutations (right axis)"]
    assert {t.get_text() for t in ax1.get_xticklabels()} <= {
        "SBS1",
        "SBS5",
    }


def test_model_plot_signature_correlations_needs_covariates(tmp_path):
    from sigmutsel.models import Model
    from tests.test_implicit_baselines import _full_context_dataset

    model = Model(_full_context_dataset(tmp_path))
    with pytest.raises(ValueError, match="covariates"):
        model.plot_signature_correlations(show=False)


def test_profile_percent_ticks_are_distinct_for_small_values():
    """A profile peaking near 2.5% keeps distinct tick labels (whole
    percents would read 0%, 0%, 1%, 1%, ...)."""
    import matplotlib.pyplot as plt

    values = pd.Series(
        np.linspace(0.0, 0.025, 96), index=_all_types()
    )
    ax = plot_signature_profile(values, context_labels=False)
    ax.figure.canvas.draw()
    labels = [
        t.get_text() for t in ax.get_yticklabels() if t.get_text()
    ]
    assert len(labels) == len(set(labels))
    plt.close(ax.figure)


def test_model_plot_gamma_results_returns_the_figure(tmp_path):
    """Stored gene gammas, plotted through the Model: labels are gene
    symbols and the interval mass is passed on."""
    import xarray as xr
    from matplotlib.figure import Figure

    from sigmutsel.models import Model
    from tests.test_implicit_baselines import _full_context_dataset

    dataset = _full_context_dataset(tmp_path)
    dataset._mutation_db["gene"] = dataset._mutation_db[
        "ensembl_gene_id"
    ].str.replace("ENSG_", "GENE")
    model = Model(dataset)
    rng = np.random.default_rng(0)
    for gene_id in ("ENSG_A", "ENSG_B"):
        posterior = xr.Dataset(
            {"gamma": (("chain", "draw"), rng.gamma(4, 1, (2, 300)))}
        )
        model.gammas[gene_id] = type(
            "R", (), {"posterior": posterior}
        )()
    fig = model.plot_gamma_results(level="gene", prob=0.89)
    assert isinstance(fig, Figure)
    legend = fig.axes[0].get_legend()
    assert {t.get_text() for t in legend.get_texts()} == {
        "GENEA",
        "GENEB",
    }
