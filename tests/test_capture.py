"""Tests for the capture territory (sigmutsel.capture).

On the toy two-gene annotation of ``test_channel_universe`` (GRCh38
positions read as they are: ``build="hg38"``), with its germline mask
and planted site weights:

1. a BED covering everything, without a GC curve, leaves every
   opportunity table unchanged (rtol 1e-12);
2. the weights are ``g(GC of the exon) * (1 inside, s_out outside)``,
   splice sites against the padded BED, by their anchor exon's GC;
3. the overall scale of ``T`` cancels in the rates;
4. with a capture, the site table rebuilds the captured opportunity
   and every variant's rate is the model's (``compute_mu_m_per_tumor``
   with the captured route weights), at rtol 1e-9, splice variants
   included.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.capture import (
    CaptureTerritory,
    capture_weights,
    exon_gc,
    exon_index,
    splice_flat_index,
)
from sigmutsel.channel_universe import (
    CHANNELS,
    compute_channel_opportunity,
)
from sigmutsel.estimate_mus import (
    compute_mu_g_channel_per_tumor,
    type_denominators,
)
from sigmutsel.site_rates import (
    SiteRates,
    SiteTable,
    tumor_type_rates,
)
from sigmutsel.site_weights import variant_route_weights
from tests import test_channel_universe as _toy
from tests import test_site_rates as _sr
from tests.test_channel_universe import MASK, _masked_keys, _planted

models = _toy.models  # the toy two-gene annotation, as a fixture


def _bed(tmp_path, rows, name="targets.bed"):
    path = tmp_path / name
    path.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows))
    return str(path)


def _tables(models, capture=None):
    return compute_channel_opportunity(
        models,
        masked_keys=_masked_keys(MASK),
        site_weights=_planted(),
        capture_w=(
            None
            if capture is None
            else capture_weights(models, capture)
        ),
    )


def _partial(tmp_path, outside=0.25, shift=0.0):
    # Gene A's first exon (chr1 101-109) only; nothing on chr2.
    return CaptureTerritory(
        _bed(tmp_path, [("chr1", 100, 109)], f"p{shift}.bed"),
        outside=outside,
        gc=(0.0, 1.0),
        log_g=(0.0 + shift, -2.0 + shift),
        build="hg38",
    )


def test_full_bed_without_gc_is_the_identity(models, tmp_path):
    cap = CaptureTerritory(
        _bed(tmp_path, [("chr1", 0, 10**7), ("chr2", 0, 10**7)]),
        outside=0.3,
        build="hg38",
    )
    base = _tables(models)
    got = _tables(models, cap)
    for name in list(CHANNELS) + ["contexts"]:
        np.testing.assert_allclose(
            got[name].to_numpy(), base[name].to_numpy(), rtol=1e-12
        )


def test_weights_follow_the_definition(models, tmp_path):
    cap = _partial(tmp_path)
    cw, sw = capture_weights(models, cap)
    exon = exon_index(models)
    gc = exon_gc(models, exon)
    coding = np.flatnonzero(models.gpos >= 0)
    chrom = models.chrom_of[coding]
    pos = models.gpos[coding]
    inside = (chrom == "chr1") & (pos >= 101) & (pos <= 109)
    assert inside.any() and (~inside).any()
    expected = np.exp(-2.0 * gc[exon[coding]]) * np.where(
        inside, 1.0, 0.25
    )
    np.testing.assert_allclose(cw[coding], expected, rtol=1e-12)
    assert (cw[models.gpos < 0] == 0).all()
    sp = models.splice
    s_chrom = models.selection["chrom"].to_numpy()[
        sp["gene"].to_numpy()
    ]
    s_pos = sp["gpos"].to_numpy()
    pad = cap.splice_padding
    s_in = (
        (s_chrom == "chr1")
        & (s_pos >= 101 - pad)
        & (s_pos <= 109 + pad)
    )
    anchor_gc = gc[exon[splice_flat_index(models)]]
    np.testing.assert_allclose(
        sw,
        np.exp(-2.0 * anchor_gc) * np.where(s_in, 1.0, 0.25),
        rtol=1e-12,
    )
    # Each splice site hangs off a coding base of its own gene.
    flat = splice_flat_index(models)
    assert (models.gene_of[flat] == sp["gene"].to_numpy()).all()
    assert (models.gpos[flat] >= 0).all()


def test_the_scale_of_the_capture_cancels(models, tmp_path):
    mu_taus = _sr._mu_taus()
    rates = []
    for shift in (0.0, 0.7):
        tables = _tables(models, _partial(tmp_path, shift=shift))
        type_opp = sum(tables[c] for c in CHANNELS)
        rates.append(
            compute_mu_g_channel_per_tumor(
                mu_taus,
                tables["mis"] + tables["non"] + tables["spl"],
                tables["contexts"],
                type_opportunity=type_opp,
            )
        )
    np.testing.assert_allclose(
        rates[0].to_numpy(), rates[1].to_numpy(), rtol=1e-12
    )


def test_variant_rates_carry_the_capture(models, tmp_path):
    cap = _partial(tmp_path)
    tables = _tables(models, cap)
    base = _tables(models)
    # The capture changes the opportunity (else the test is empty).
    assert not np.allclose(
        tables["mis"].to_numpy(), base["mis"].to_numpy()
    )
    mu_taus = _sr._mu_taus()
    table = SiteTable.build(
        models,
        _sr.GENES,
        masked_keys=_masked_keys(MASK),
        site_weights=_planted(),
        capture=cap,
    )
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
        db,
        models,
        None,
        _planted(),
        masked_keys=_masked_keys(MASK),
        capture_w=capture_weights(models, cap),
    )
    variants["route_weights"] = weights.reindex(
        variants.index
    ).to_numpy()
    assert (variants["channel"] == "spl").any()
    assert (
        variants.loc[variants["channel"] == "spl", "route_weights"]
        .map(lambda w: isinstance(w, list))
        .all()
    )
    mu_ms, _, _ = _sr._mu_ms(tables, mu_taus, variants)
    info, _, observable = rates.variant_rates(
        list(variants.index), genes=list(variants["ensembl_gene_id"])
    )
    assert info["error"].isna().all()
    np.testing.assert_allclose(
        observable.to_numpy(), mu_ms.to_numpy(), rtol=1e-9, atol=0
    )


def test_spec_round_trip_and_validation(tmp_path):
    cap = _partial(tmp_path)
    again = CaptureTerritory.from_dict(cap.to_dict())
    assert again.label() == cap.label()
    assert CaptureTerritory.from_dict(None) is None
    other = _partial(tmp_path, outside=0.5)
    assert other.label() != cap.label()
    with pytest.raises(ValueError):
        CaptureTerritory(cap.bed_path, outside=1.5)
    with pytest.raises(ValueError):
        CaptureTerritory(cap.bed_path, gc=(0.0, 1.0))
    with pytest.raises(ValueError):
        CaptureTerritory(
            cap.bed_path, gc=(1.0, 0.0), log_g=(0.0, 0.0)
        )
    with pytest.raises(ValueError):
        CaptureTerritory(cap.bed_path, build="hg18")
    np.testing.assert_allclose(
        cap.g_at([0.0, 0.5, 1.0, 2.0]),
        np.exp([0.0, -1.0, -2.0, -2.0]),
    )
    pd.testing.assert_series_equal(
        pd.Series(cap.to_dict()["gc"]), pd.Series([0.0, 1.0])
    )


def test_cpg_term_of_the_site_weights():
    from sigmutsel.constants import canonical_types_order as order
    from sigmutsel.site_weights import (
        CPG_CT,
        SiteWeights,
        fit_site_weights,
        weights_at,
    )

    plain = SiteWeights()
    sw = SiteWeights(cpg=np.array([-1.0, -3.0]))
    assert not plain.has_cpg() and sw.has_cpg()
    assert plain.label() != sw.label()
    assert SiteWeights.from_dict(sw.to_dict()).label() == sw.label()
    # An old dict, without the term, loads with no term.
    d = plain.to_dict()
    d.pop("cpg")
    assert not SiteWeights.from_dict(d).has_cpg()
    ct = order.index("A[C>T]G")
    other = order.index("A[C>A]G")
    types = np.array([[ct, other, other]])
    w = weights_at(
        np.zeros(1, int),
        np.zeros(1, int),
        types,
        sw,
        x=np.array([0.2]),
    )
    np.testing.assert_allclose(
        w[0], [np.exp(-1.0 * 0.2 - 3.0 * 0.04), 1.0, 1.0], rtol=1e-12
    )
    with pytest.raises(ValueError):
        weights_at(np.zeros(1, int), np.zeros(1, int), types, sw)

    # The fit recovers a planted term within genes, and the prior acts.
    rng = np.random.default_rng(5)
    n_gene, per_exon, n_exon = 400, 20, 6
    gene = np.repeat(np.arange(n_gene), n_exon * per_exon)
    exon = np.repeat(np.arange(n_gene * n_exon), per_exon)
    x_of_exon = rng.uniform(-0.15, 0.2, n_gene * n_exon)
    n = len(gene)
    ct_types = np.flatnonzero(CPG_CT)
    ot_types = np.flatnonzero(~CPG_CT)
    types = np.stack(
        [
            rng.choice(ct_types, n),
            rng.choice(ot_types, n),
            rng.choice(ot_types, n),
        ],
        axis=1,
    )
    planted = np.array([-1.0, -3.0])
    x = x_of_exon[exon]
    h = np.exp(planted[0] * x + planted[1] * x**2)
    p = np.stack([h, np.ones(n), np.ones(n)], axis=1)
    p = p / p.sum()
    draw = rng.multinomial(400_000, p.ravel()).reshape(p.shape)
    rows, alts = np.nonzero(draw)
    rows = np.repeat(rows, draw[rows, alts])
    alts = np.repeat(alts, draw[draw > 0])
    # Per-type rates calibrated to the calls, as the real expectation is.
    per_type = np.bincount(
        types[rows, alts], minlength=96
    ) / np.maximum(np.bincount(types.ravel(), minlength=96), 1)
    expected = per_type[types]
    dbin, orient = np.zeros(n, int), np.zeros(n, int)
    design = (exon, gene, x_of_exon)
    flat = fit_site_weights(
        dbin, orient, types, expected, rows, alts, cpg=design
    )
    np.testing.assert_allclose(flat.cpg, planted, atol=0.15)
    pooled = fit_site_weights(
        dbin,
        orient,
        types,
        expected,
        rows,
        alts,
        cpg=design,
        cpg_prior=(np.array([0.5, 0.5]), np.zeros((2, 2))),
    )
    np.testing.assert_allclose(pooled.cpg, [0.5, 0.5], rtol=1e-12)
    shrunk = fit_site_weights(
        dbin,
        orient,
        types,
        expected,
        rows,
        alts,
        cpg=design,
        cpg_prior=(np.array([0.5, 0.5]), np.eye(2) * 1e-3),
    )
    assert abs(shrunk.cpg[0] - 0.5) < abs(flat.cpg[0] - 0.5)
    # No term asked for: unchanged behaviour.
    assert not fit_site_weights(
        dbin, orient, types, expected, rows, alts
    ).has_cpg()


def test_cpg_term_reaches_the_opportunity_and_variants(
    models, tmp_path
):
    from sigmutsel.site_weights import SiteWeights, position_gc

    base = _planted()
    sw = SiteWeights(
        distance=base.distance,
        strand=base.strand,
        cpg=np.array([-1.0, -3.0]),
    )
    tables = compute_channel_opportunity(
        models, masked_keys=_masked_keys(MASK), site_weights=sw
    )
    plain = compute_channel_opportunity(
        models, masked_keys=_masked_keys(MASK), site_weights=base
    )
    # Only CpG C>T columns can move.
    moved = [
        c
        for c in plain["mis"].columns
        if not np.allclose(tables["mis"][c], plain["mis"][c])
    ]
    assert all(c[2:5] == "C>T" and c[-1] == "G" for c in moved)
    # SiteTable and the opportunity agree with the term on.
    mu_taus = _sr._mu_taus()
    table = SiteTable.build(
        models,
        _sr.GENES,
        masked_keys=_masked_keys(MASK),
        site_weights=sw,
    )
    type_opp = sum(tables[c] for c in CHANNELS)
    M = tumor_type_rates(
        mu_taus, type_denominators(tables["contexts"], type_opp)
    )
    SiteRates(table, M, _sr.MULT).verify_opportunity(tables)
    x = position_gc(models)
    assert np.abs(x).max() <= 0.5
    assert (x[models.gpos < 0] == 0).all()
