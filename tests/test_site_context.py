"""Tests for pentanucleotide context factors (sigmutsel.site_context).

1. The factor function: a context with no pentamer, and a signature
   in no group, take factor 1; one group at full share is that group's
   normalised flank factor.
2. The opportunity mode: the cohort table averages 1 per type over its
   territory; a table of ones changes nothing; on the toy annotation
   with a context table the site table rebuilds the opportunity and
   every variant gets the rate ``compute_mu_m_per_tumor`` gives it
   (rtol 1e-9), as for the other site weights.
3. The placement mode: within a group of sites the multipliers keep
   the group's total in every tumor.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel import site_weights as sw_module
from sigmutsel.constants import canonical_types_order
from sigmutsel.site_context import (
    NO_CONTEXT,
    ContextFactors,
    cohort_context_table,
    context_factor,
    context_opportunity,
    group_mix,
    normalised_flank,
    placement_multipliers,
)
from sigmutsel.site_weights import SiteWeights, weights_at
from tests import test_channel_universe as _toy
from tests import test_site_rates as _sr

models = _toy.models

SIGS = ["SBS1", "SBS2", "SBS13", "SBS5", "SBS99"]


def _factors(seed=0):
    rng = np.random.default_rng(seed)
    return ContextFactors(
        groups=["APOBEC", "SBS1", "flat"],
        log_flank=rng.normal(0, 0.4, size=(3, 96, 16)),
        signature_groups={
            "SBS2": "APOBEC",
            "SBS13": "APOBEC",
            "SBS5": "flat",
        },
    )


def _exposures(seed=1, n=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.gamma(1.0, 50.0, size=(n, len(SIGS))),
        index=[f"t{i}" for i in range(n)],
        columns=SIGS,
    )


def _emissions(seed=2):
    rng = np.random.default_rng(seed)
    e = rng.dirichlet(np.ones(96), size=len(SIGS)).T
    return pd.DataFrame(e, index=canonical_types_order, columns=SIGS)


def _territory(seed=3):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 10.0, size=(96, 17))


def test_factor_is_one_without_a_pentamer_or_a_group():
    f = _factors()
    psi = normalised_flank(f, _territory())
    mix = group_mix(f, _exposures(), _emissions())  # (J, 96, K + 1)
    out = context_factor(psi, mix, np.arange(96), NO_CONTEXT)
    np.testing.assert_array_equal(out, 1.0)
    # SBS99 is in no group: its share takes factor 1
    only99 = pd.DataFrame(
        [[0, 0, 0, 0, 10.0]], columns=SIGS, index=["x"]
    )
    mix99 = group_mix(f, only99, _emissions())
    out = context_factor(
        psi, mix99, np.arange(96)[:, None], np.arange(16)
    )
    np.testing.assert_allclose(out, 1.0)


def test_one_group_at_full_share_is_its_flank_factor():
    f = _factors()
    psi = normalised_flank(f, _territory())
    only2 = pd.DataFrame(
        [[0, 7.0, 0, 0, 0]], columns=SIGS, index=["x"]
    )
    mix = group_mix(f, only2, _emissions())
    tt, cc = np.meshgrid(np.arange(96), np.arange(16), indexing="ij")
    np.testing.assert_allclose(
        context_factor(psi, mix, tt, cc)[0], psi[0], rtol=1e-12
    )
    # each group's psi averages 1 per type over the territory
    W = _territory()[:, :16]
    mean = np.einsum("tc,ktc->kt", W, psi) / W.sum(axis=1)
    np.testing.assert_allclose(mean, 1.0, rtol=1e-12)


def test_cohort_table_averages_one_over_its_territory():
    W = _territory()
    table = cohort_context_table(
        _factors(), _exposures(), _emissions(), W
    )
    assert table.shape == (96, 16)
    np.testing.assert_allclose(
        (W[:, :16] * table).sum(axis=1) / W[:, :16].sum(axis=1),
        1.0,
        rtol=1e-12,
    )


def test_placement_keeps_each_groups_total():
    """Sum over a group's sites of weight x multiplier is the group's
    total weight, in every tumor."""
    rng = np.random.default_rng(5)
    f = _factors()
    psi = normalised_flank(f, _territory())
    mix = group_mix(f, _exposures(n=6), _emissions())
    for tau in (0, 17, 60):
        n = 40
        c = rng.integers(0, 17, size=n)  # some with no pentamer
        w = rng.gamma(2.0, 0.5, size=n)
        W = context_opportunity(np.full(n, tau), w, c)[tau]
        m = placement_multipliers(
            psi, mix, np.full(n, tau), c, np.tile(W, (n, 1))
        )  # (n, J)
        np.testing.assert_allclose(
            (w[:, None] * m).sum(axis=0), w.sum(), rtol=1e-12
        )
        assert np.ptp(m) > 0.05  # the factor does move rates


def test_site_weights_round_trip_with_a_context_table():
    sw = _toy._planted()
    sw.context = np.random.default_rng(0).uniform(0.5, 2.0, (96, 16))
    back = SiteWeights.from_dict(sw.to_dict())
    np.testing.assert_allclose(back.context, sw.context, rtol=1e-12)
    assert back.label() == sw.label() != _toy._planted().label()
    assert (
        SiteWeights.from_dict(_toy._planted().to_dict()).context
        is None
    )


def test_weights_at_needs_and_applies_the_contexts():
    sw = _toy._planted()
    rng = np.random.default_rng(1)
    dbin = rng.integers(0, 8, size=20)
    orient = rng.integers(0, 2, size=20)
    types = rng.integers(0, 96, size=(20, 3))
    c = rng.integers(0, 17, size=20)
    base = weights_at(dbin, orient, types, sw)
    sw.context = rng.uniform(0.5, 2.0, (96, 16))
    with pytest.raises(ValueError):
        weights_at(dbin, orient, types, sw)
    got = weights_at(dbin, orient, types, sw, c=c)
    want = base * np.where(
        c[:, None] < 16,
        sw.context[types, np.minimum(c, 15)[:, None]],
        1,
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)


# --- the opportunity mode end to end, on the toy annotation ----------


def _fake_contexts(models, genome_dir="default"):
    """Deterministic outer bases per flat position, some unreadable."""
    return ((np.arange(len(models.codes)) * 7) % 17).astype(np.int8)


def _ctx_weights(seed=0, ones=False):
    sw = _toy._planted()
    rng = np.random.default_rng(seed)
    sw.context = (
        np.ones((96, 16)) if ones else rng.uniform(0.4, 2.2, (96, 16))
    )
    return sw


def _toy_tables(models, sw):
    from sigmutsel.channel_universe import compute_channel_opportunity

    return compute_channel_opportunity(
        models,
        masked_keys=_toy._masked_keys(_toy.MASK),
        site_weights=sw,
    )


def test_a_context_table_of_ones_changes_nothing(models, monkeypatch):
    monkeypatch.setattr(
        sw_module, "position_context5", _fake_contexts
    )
    plain = _toy_tables(models, _toy._planted())
    ones = _toy_tables(models, _ctx_weights(ones=True))
    for k in plain:
        if not isinstance(plain[k], pd.DataFrame):
            continue
        np.testing.assert_allclose(
            np.asarray(ones[k], float),
            np.asarray(plain[k], float),
            rtol=1e-12,
        )


def test_opportunity_mode_site_rates_match_mu_ms(models, monkeypatch):
    """The context table enters the opportunity and the route weights
    alike: the site table rebuilds the tables, and every variant's
    site rate is its compute_mu_m_per_tumor rate (rtol 1e-9)."""
    from sigmutsel.channel_universe import CHANNELS
    from sigmutsel.estimate_mus import type_denominators
    from sigmutsel.site_rates import (
        SiteRates,
        SiteTable,
        tumor_type_rates,
    )
    from sigmutsel.site_weights import variant_route_weights

    monkeypatch.setattr(
        sw_module, "position_context5", _fake_contexts
    )
    sw = _ctx_weights()
    tables = _toy_tables(models, sw)
    plain = _toy_tables(models, _toy._planted())
    # the table does change the opportunity
    assert not np.allclose(
        np.asarray(tables["mis"], float),
        np.asarray(plain["mis"], float),
    )
    table = SiteTable.build(
        models,
        _sr.GENES,
        masked_keys=_toy._masked_keys(_toy.MASK),
        site_weights=sw,
    )
    mu_taus = _sr._mu_taus()
    type_opp = sum(tables[c] for c in CHANNELS)
    M = tumor_type_rates(
        mu_taus, type_denominators(tables["contexts"], type_opp)
    )
    rates = SiteRates(table, M, _sr.MULT)
    rates.verify_opportunity(tables)

    db = _sr._every_call(models)
    variants = db.drop_duplicates("variant").set_index("variant")
    variants = variants[
        ["ensembl_gene_id", "gene", "channel", "routes"]
    ]
    variants["mut_types"] = variants["routes"].str.split(";")
    weights = variant_route_weights(
        db, models, None, sw, masked_keys=_toy._masked_keys(_toy.MASK)
    )
    variants["route_weights"] = weights.reindex(
        variants.index
    ).to_numpy()
    mu_ms, _, _ = _sr._mu_ms(tables, mu_taus, variants)
    info, _, observable = rates.variant_rates(
        list(variants.index), genes=list(variants["ensembl_gene_id"])
    )
    assert info["error"].isna().all()
    np.testing.assert_allclose(
        observable.to_numpy(), mu_ms.to_numpy(), rtol=1e-9, atol=0
    )
