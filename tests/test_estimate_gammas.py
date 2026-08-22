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
