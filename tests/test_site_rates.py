"""Tests for site rates (sigmutsel.site_rates).

On the toy two-gene annotation of ``test_channel_universe``, with a
germline mask and non-trivial site weights, every possible coding SNV
and exact-context splice change is classified as production classifies
calls, and its rate is checked against the model's own path:

1. per gene, channel and type, the site table rebuilds the opportunity
   and its summed rates are the gene rates;
2. every variant gets the rate ``compute_mu_m_per_tumor`` gives it
   (rtol 1e-9), directly and through ``SiteRates.from_model``;
3. a protein change's routes are ``classify_calls``' routes, in order;
4. masked routes count in the mutation rate and not in the observable
   rate.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.channel_universe import (
    _BASES,
    _TYPE_TABLE,
    CHANNELS,
    classify_calls,
    compute_channel_opportunity,
)
from sigmutsel.constants import canonical_types_order
from sigmutsel.estimate_mus import (
    compute_mu_g_channel_per_tumor,
    compute_mu_m_per_tumor,
    type_denominators,
)
from sigmutsel.site_rates import (
    SiteRates,
    SiteTable,
    tumor_type_rates,
)
from sigmutsel.site_weights import variant_route_weights
from tests import test_channel_universe as _toy
from tests.test_channel_universe import MASK, _masked_keys, _planted

models = _toy.models  # the toy two-gene annotation, as a fixture

TUMORS = ["t1", "t2", "t3"]
GENES = ["ENSGA", "ENSGB"]
MULT = pd.DataFrame(
    {"syn": [1.3, 0.6], "nonsyn": [1.3 * 1.2, 0.6 * 1.2]}, index=GENES
)


def _mu_taus(seed=0):
    return pd.DataFrame(
        np.random.default_rng(seed).gamma(2.0, 1.0, size=(3, 96)),
        index=TUMORS,
        columns=canonical_types_order,
    )


def _tables(models):
    return compute_channel_opportunity(
        models,
        masked_keys=_masked_keys(MASK),
        site_weights=_planted(),
    )


def _site_rates(models, tables, mu_taus, genome_dir=None):
    table = SiteTable.build(
        models,
        GENES,
        masked_keys=_masked_keys(MASK),
        site_weights=_planted(),
    )
    type_opp = sum(tables[c] for c in CHANNELS)
    M = tumor_type_rates(
        mu_taus, type_denominators(tables["contexts"], type_opp)
    )
    return SiteRates(table, M, MULT, genome_dir=genome_dir)


def _every_call(models):
    """One call per (coding position, alt), plus d1/a1 splice calls.

    CDS ends are left out: they have no element (tested apart).

    Donor +2 and acceptor -2 are left out: without a genome their
    context is a composition mixture, not one type (tested apart).
    """
    rows = []
    minus = models.selection["strand"].to_numpy() == "-"
    names = models.selection["gene_name"].to_numpy()
    for i in np.flatnonzero(models.gpos >= 0):
        g = models.gene_of[i]
        if models.local[i] in (0, models.lengths[g] - 1):
            continue  # a CDS end: no element (test_cds_ends_use_...)
        ref = int(models.codes[i])
        for k in (1, 2, 3):
            alt = (ref + k) % 4
            gref, galt = (
                (3 - ref, 3 - alt) if minus[g] else (ref, alt)
            )
            rows.append(
                (
                    models.gene_ids[g],
                    names[g],
                    models.chrom_of[i],
                    int(models.gpos[i]),
                    _BASES[gref],
                    _BASES[galt],
                    "",
                )
            )
    sp = models.splice
    for s in range(len(sp)):
        kind = sp["kind"].iat[s]
        if kind not in ("d1", "a1"):
            continue
        g = int(sp["gene"].iat[s])
        exon = int(sp["exon_base"].iat[s])
        left, ref, right = (
            (exon, 2, 3) if kind == "d1" else (0, 2, exon)
        )
        for k in (1, 2, 3):
            alt = (ref + k) % 4
            tau = canonical_types_order[
                _TYPE_TABLE[left, ref, right, alt]
            ]
            gref, galt = (
                (3 - ref, 3 - alt) if minus[g] else (ref, alt)
            )
            rows.append(
                (
                    models.gene_ids[g],
                    names[g],
                    models.selection["chrom"].iat[g],
                    int(sp["gpos"].iat[s]),
                    _BASES[gref],
                    _BASES[galt],
                    tau,
                )
            )
    db = pd.DataFrame(
        rows,
        columns=[
            "ensembl_gene_id",
            "gene",
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "type",
        ],
    )
    out = classify_calls(db, models)
    db = pd.concat([db, out], axis=1)
    db = db[db["in_universe"]].copy()
    db["variant"] = db["variant_label"]
    return db


def _variant_db(models, db):
    variants = db.drop_duplicates("variant").set_index("variant")
    variants = variants[
        ["ensembl_gene_id", "gene", "channel", "routes"]
    ]
    variants["mut_types"] = variants["routes"].str.split(";")
    weights = variant_route_weights(db, models, None, _planted())
    variants["route_weights"] = weights.reindex(
        variants.index
    ).to_numpy()
    return variants


def _mu_ms(tables, mu_taus, variants):
    """compute_mu_ms's path, with the multipliers as the gene scale."""
    contexts = tables["contexts"]
    type_opp = sum(tables[c] for c in CHANNELS)
    syn_tab = tables["syn"]
    nonsyn_tab = tables["mis"] + tables["non"] + tables["spl"]
    parts = []
    for silent, tab, col in (
        (False, nonsyn_tab, "nonsyn"),
        (True, syn_tab, "syn"),
    ):
        part = variants[(variants["channel"] == "syn") == silent]
        rates = compute_mu_g_channel_per_tumor(
            mu_taus,
            tab,
            contexts,
            separate_per_tau=True,
            type_opportunity=type_opp,
        )
        rates = {
            tau: frame.mul(MULT[col], axis=0)
            for tau, frame in rates.items()
        }
        parts.append(
            compute_mu_m_per_tumor(
                part, rates, contexts, opportunity_by_type=tab
            )
        )
    return pd.concat(parts).loc[variants.index], nonsyn_tab, syn_tab


def _covered(variants, syn_tab, nonsyn_tab):
    """Variants whose every route's type has opportunity in its channel.

    mu_ms spreads a type rate over the channel's elements of the type;
    where the mask removed a gene's last one, it drops the route (0/0)
    while the site rate keeps it as mutation rate.
    """

    def covered(row):
        tab = syn_tab if row["channel"] == "syn" else nonsyn_tab
        return all(
            tab.at[row["ensembl_gene_id"], t] > 0
            for t in row["mut_types"]
        )

    return variants.apply(covered, axis=1).to_numpy()


def test_site_table_rebuilds_the_opportunity(models):
    tables = _tables(models)
    rates = _site_rates(models, tables, _mu_taus())
    rates.verify_opportunity(tables)
    with pytest.raises(AssertionError):
        bad = dict(tables)
        bad["mis"] = tables["mis"] * 1.01
        rates.verify_opportunity(bad)


def test_site_rates_sum_to_the_gene_rates(models):
    tables = _tables(models)
    mu_taus = _mu_taus()
    rates = _site_rates(models, tables, mu_taus)
    got = rates.aggregate_rates().droplevel("label")
    type_opp = sum(tables[c] for c in CHANNELS)
    for channel in CHANNELS:
        col = "syn" if channel == "syn" else "nonsyn"
        # sum_tau n^h_{g tau} mu_bar^j_tau / D_tau: the model's gene rate
        # (compute_mu_g_channel_per_tumor), written out so that a type
        # absent from the toy universe (D_tau = 0) adds 0, not 0/0.
        D = type_denominators(tables["contexts"], type_opp)
        keep = D > 0
        expected = (
            tables[channel].loc[:, keep] / D[keep]
        ) @ mu_taus.loc[:, keep].T
        expected = expected.mul(MULT[col], axis=0)
        frame = got.xs(channel, level="channel").reindex(GENES)
        np.testing.assert_allclose(
            frame.to_numpy(),
            expected.loc[GENES].to_numpy(),
            rtol=1e-12,
        )
    totals = rates.aggregate_rates(per_tumor=False).droplevel("label")
    np.testing.assert_allclose(
        totals.to_numpy(), got.sum(axis=1).to_numpy(), rtol=1e-12
    )
    pooled = rates.aggregate_rates(by_gene=False)
    np.testing.assert_allclose(
        pooled.groupby(level="channel").sum().loc[list(CHANNELS)],
        got.groupby(level="channel").sum().loc[list(CHANNELS)],
        rtol=1e-12,
    )


def test_every_variant_gets_the_models_rate(models):
    tables = _tables(models)
    mu_taus = _mu_taus()
    rates = _site_rates(models, tables, mu_taus)
    db = _every_call(models)
    variants = _variant_db(models, db)
    mu_ms, nonsyn_tab, syn_tab = _mu_ms(tables, mu_taus, variants)
    info, got, _ = rates.variant_rates(
        list(variants.index), genes=list(variants["ensembl_gene_id"])
    )
    assert info["error"].isna().all()
    # Test 3: the same routes, in the same order.
    assert (info["routes"] == variants["routes"]).all()
    ok = _covered(variants, syn_tab, nonsyn_tab)
    assert ok.sum() > 0.95 * len(ok)
    np.testing.assert_allclose(
        got.to_numpy()[ok], mu_ms.to_numpy()[ok], rtol=1e-9
    )
    assert (mu_ms.sum(axis=1).to_numpy()[ok] > 0).all()


def test_masked_routes_are_mutation_not_observable(models):
    tables = _tables(models)
    rates = _site_rates(models, tables, _mu_taus())
    # GA position 105 T>C is masked: p.F2S's only route.
    info, got, observable = rates.variant_rates(["GA p.F2S"])
    assert info.at["GA p.F2S", "n_masked_routes"] == 1
    assert info.at["GA p.F2S", "rate_total"] > 0
    assert info.at["GA p.F2S", "observable_rate_total"] == 0
    assert (observable.loc["GA p.F2S"] == 0).all()
    snv, per = rates.snv_rates(["chr1"], [105], ["C"])
    assert not snv["observable"].iat[0]
    np.testing.assert_allclose(per.to_numpy()[0], got.to_numpy()[0])


def test_genomic_snvs_on_both_strands(models):
    tables = _tables(models)
    rates = _site_rates(models, tables, _mu_taus())
    # GB codon 1 ATG, second base T: genomic A at 1107 -> G is coding
    # T>C, ACG, p.M1T.
    snv, per = rates.snv_rates(
        ["chr2", "chr1"], [1107, 150], ["G", "G"]
    )
    assert len(snv) == 1  # the intron position is no element
    _, got, _ = rates.variant_rates(["GB p.M1T"])
    assert snv["channel"].iat[0] == "mis"
    np.testing.assert_allclose(per.to_numpy()[0], got.to_numpy()[0])
    # the reference base is no substitution
    empty, _ = rates.snv_rates(["chr2"], [1107], ["A"])
    assert len(empty) == 0


def test_cds_ends_use_the_genomic_type(models, tmp_path):
    """A change reachable only at a CDS end gets its genomic type.

    GB's first coding base (A of ATG, genomic T at 1108) has no
    element; p.M1V is reachable only there. Like mu_ms (which falls
    back to the call's type), it takes that site's type at weight 1.
    """
    tables = _tables(models)
    mu_taus = _mu_taus()
    plain = _site_rates(models, tables, mu_taus)
    info, got, _ = plain.variant_rates(["GB p.M1V"])
    assert info.at["GB p.M1V", "n_routes"] == 0
    assert info.at["GB p.M1V", "context"] == "edge (no genome)"
    # genomic 1107 A, 1108 T, 1109 G: on the coding strand C A T, a
    # coding A>G, i.e. A[T>C]G.
    genome = np.zeros(2000, dtype=np.uint8)
    genome[1106], genome[1107], genome[1108] = 0, 3, 2
    (tmp_path / "2.txt").write_bytes(genome.tobytes())
    exact = _site_rates(models, tables, mu_taus, genome_dir=tmp_path)
    info, got, _ = exact.variant_rates(["GB p.M1V"])
    assert info.at["GB p.M1V", "context"] == "edge"
    assert info.at["GB p.M1V", "routes"] == "A[T>C]G"
    tau = canonical_types_order.index("A[T>C]G")
    expected = exact.M.to_numpy()[:, tau] * MULT.at["ENSGB", "nonsyn"]
    np.testing.assert_allclose(
        got.to_numpy()[0], expected, rtol=1e-12
    )
    snv, per = exact.snv_rates(["chr2"], [1108], ["C"])
    assert snv["context"].iat[0] == "edge"
    np.testing.assert_allclose(
        per.to_numpy()[0], expected, rtol=1e-12
    )


def test_splice_context_from_a_genome(models, tmp_path):
    """Donor +2 takes the genome's +3 base when a genome is given."""
    tables = _tables(models)
    mu_taus = _mu_taus()
    plain = _site_rates(models, tables, mu_taus)
    # GA donor +2 at 111 (T on the + strand); make +3 (112) a C and
    # +1 (110) the canonical G.
    genome = np.zeros(2000, dtype=np.uint8)
    genome[109], genome[110], genome[111] = 2, 3, 1
    (tmp_path / "1.txt").write_bytes(genome.tobytes())
    exact = _site_rates(models, tables, mu_taus, genome_dir=tmp_path)
    info, got, _ = exact.variant_rates(["GA c.9+2T>A"])
    assert info.at["GA c.9+2T>A", "context"] == "genome"
    tau = canonical_types_order.index("G[T>A]C")
    expected = exact.M.to_numpy()[:, tau] * MULT.at["ENSGA", "nonsyn"]
    np.testing.assert_allclose(
        got.to_numpy()[0], expected, rtol=1e-12
    )
    info, got, _ = plain.variant_rates(["GA c.9+2T>A"])
    assert info.at["GA c.9+2T>A", "context"] == "composition"
    assert info.at["GA c.9+2T>A", "n_routes"] == 4


def test_from_model_reproduces_mu_ms(models, tmp_path):
    from sigmutsel.models import Model, MutationDataset

    tables = _tables(models)
    db = _every_call(models)
    db["Tumor_Sample_Barcode"] = np.resize(TUMORS, len(db))
    db["Variant_Classification"] = np.where(
        db["channel"] == "syn", "Silent", "Missense_Mutation"
    )
    dataset = MutationDataset(str(tmp_path / "mafs"))
    dataset._mutation_db = db.reset_index(drop=True)
    dataset._channel_universe = {
        "territory": None,
        "splice_padding": 0,
        "germline_mask_af": 1e-4,
        "site_weights": _planted().to_dict(),
    }
    dataset._contexts_by_gene = tables["contexts"]
    for ch in CHANNELS:
        setattr(dataset, f"_contexts_by_gene_{ch}", tables[ch])
    dataset._contexts_by_gene_nonsyn = (
        tables["mis"] + tables["non"] + tables["spl"]
    )
    dataset._variant_db = _variant_db(models, db)
    cov = pd.DataFrame({"x1": [0.4, -0.7]}, index=GENES)
    model = Model(dataset, cov)
    model._mu_taus = _mu_taus()
    model.compute_base_mus(prob_g_tau_tau_independent=False)
    model.compute_channel_base_mus()
    model.cov_effects = np.array([0.2, 0.5])
    model._rg_delta_intercept = 0.15
    model.compute_mu_ms()

    rates = SiteRates.from_model(
        model,
        r_g="none",
        transcript_models=models,
        coding_in=None,
        splice_in=None,
        masked_keys=_masked_keys(MASK),
        genome_dir=None,
    )
    variants = dataset.variant_db
    _, got, _ = rates.variant_rates(
        list(variants.index), genes=list(variants["ensembl_gene_id"])
    )
    full = _covered(
        variants, tables["syn"], dataset._contexts_by_gene_nonsyn
    )
    assert full.sum() > 0.95 * len(full)
    np.testing.assert_allclose(
        got.to_numpy()[full],
        model.mu_ms.loc[variants.index].to_numpy()[full],
        rtol=1e-9,
    )
    with pytest.raises(ValueError, match="production"):
        SiteRates.from_model(model, r_g="production")
