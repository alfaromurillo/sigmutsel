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
