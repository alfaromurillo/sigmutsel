"""Tests for compound variants.

Two things to get right. The grouping: `by` splits, distance merges
transitively within a split, and nothing is silently dropped on the
way. The fitting: rates add across members and presence is an OR,
which is what makes one gamma over the group the right estimate --
so both are checked against explicit expected values rather than
against the implementation's own arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.compound_variants import (
    compound_presence,
    compound_rates,
    define_compound_variants,
    summarize_compounds,
)
from sigmutsel.models import Model

_VARIANT_DB = pd.DataFrame(
    {
        "gene": ["KRAS", "KRAS", "KRAS", "TP53", "TP53", "BRAF"],
        "ensembl_gene_id": [
            "ENSG_KRAS",
            "ENSG_KRAS",
            "ENSG_KRAS",
            "ENSG_TP53",
            "ENSG_TP53",
            "ENSG_BRAF",
        ],
        "Chromosome": [
            "chr12",
            "chr12",
            "chr12",
            "chr17",
            "chr17",
            "chr7",
        ],
        "Start_Position": [
            25245350.0,
            25245351.0,
            25245347.0,
            7675088.0,
            7674220.0,
            140753336.0,
        ],
    },
    index=[
        "KRAS p.G12D",
        "KRAS p.G12V",
        "KRAS p.G13D",
        "TP53 p.R175H",
        "TP53 p.R248Q",
        "BRAF p.V600E",
    ],
)


def test_by_gene_with_zero_distance_keeps_sites_apart():
    compounds = define_compound_variants(_VARIANT_DB, by="gene")

    # Nothing shares a position, so every variant is its own
    # compound -- and singletons are reported, not dropped.
    assert len(compounds) == 6
    assert all(len(m) == 1 for m in compounds.values())
    assert sorted(compounds) == [
        "BRAF.1",
        "KRAS.1",
        "KRAS.2",
        "KRAS.3",
        "TP53.1",
        "TP53.2",
    ]


def test_distance_merges_transitively_within_a_gene():
    """G13D-G12D-G12V chain at 25245347/50/51: a gap of 3 then 1."""
    compounds = define_compound_variants(
        _VARIANT_DB, by="gene", merge_distance=3
    )

    kras = {
        name: members
        for name, members in compounds.items()
        if name.startswith("KRAS")
    }
    assert len(kras) == 1
    # Ends 4 bp apart, merged through the variant between them.
    assert sorted(next(iter(kras.values()))) == [
        "KRAS p.G12D",
        "KRAS p.G12V",
        "KRAS p.G13D",
    ]
    # TP53's two sites are 868 bp apart and stay separate.
    assert sum(name.startswith("TP53") for name in compounds) == 2


def test_infinite_distance_merges_a_whole_group():
    compounds = define_compound_variants(
        _VARIANT_DB, by="gene", merge_distance=np.inf
    )
    assert len(compounds) == 3
    assert len(compounds["KRAS.1"]) == 3


def test_without_by_everything_is_one_pool():
    compounds = define_compound_variants(
        _VARIANT_DB, merge_distance=np.inf
    )
    assert list(compounds) == ["compound.1"]
    assert len(compounds["compound.1"]) == 6


def test_distance_does_not_bridge_chromosomes():
    """Positions on different chromosomes are not comparable."""
    compounds = define_compound_variants(
        _VARIANT_DB, merge_distance=10**9
    )
    assert len(compounds) == 3
    assert all(
        len({_VARIANT_DB.loc[v, "Chromosome"] for v in members}) == 1
        for members in compounds.values()
    )


def test_variants_restriction_and_unknown_variant():
    compounds = define_compound_variants(
        _VARIANT_DB,
        variants=["KRAS p.G12D", "BRAF p.V600E"],
        by="gene",
    )
    assert len(compounds) == 2

    with pytest.raises(ValueError, match="not in variant_db"):
        define_compound_variants(_VARIANT_DB, variants=["NOPE p.A1B"])


def test_a_missing_annotation_becomes_its_own_group():
    table = _VARIANT_DB.copy()
    table.loc["BRAF p.V600E", "gene"] = None

    compounds = define_compound_variants(table, by="gene")

    assert "gene.NA.1" in compounds
    assert compounds["gene.NA.1"] == ["BRAF p.V600E"]


def test_a_placeless_variant_is_isolated_not_dropped(caplog):
    """Real tables carry multi-site splice annotations with no
    single position: they cannot be merged by distance, but losing
    them would quietly shrink the variant set."""
    import logging

    table = _VARIANT_DB.copy()
    table.loc["KRAS p.G12D", "Start_Position"] = np.nan

    with caplog.at_level(logging.WARNING):
        compounds = define_compound_variants(
            table, by="gene", merge_distance=5
        )

    members = [v for m in compounds.values() for v in m]
    assert sorted(members) == sorted(table.index)
    assert ["KRAS p.G12D"] in compounds.values()
    # G12V and G13D are 4 bp apart and still merge with each other.
    assert ["KRAS p.G13D", "KRAS p.G12V"] in compounds.values()
    assert "no Start_Position" in caplog.text

    # With no distance to compute, the positions do not matter.
    assert (
        len(
            define_compound_variants(
                table, by="gene", merge_distance=np.inf
            )
        )
        == 3
    )


def test_summarize_flags_a_compound_that_spans_genes():
    compounds = define_compound_variants(
        _VARIANT_DB, merge_distance=np.inf
    )
    summary = summarize_compounds(compounds, _VARIANT_DB)

    assert summary.loc["compound.1", "n_variants"] == 6
    assert summary.loc["compound.1", "n_genes"] == 3


# --- The fitting half --------------------------------------------

_MEMBERS = ["VAR1", "VAR2"]


def _model_for_compound():
    model = Model.__new__(Model)
    model.mu_ms = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4], [0.01, 0.02, 0.03, 0.04]],
        index=["VAR1", "VAR2"],
        columns=["T1", "T2", "T3", "T4"],
    )
    dataset = type("FakeDataset", (), {})()
    dataset.variants_present = pd.DataFrame(
        [[1, 0, 0, 0], [0, 0, 1, 0]],
        index=["VAR1", "VAR2"],
        columns=["T1", "T2", "T3", "T4"],
    )
    model.dataset = dataset
    model.gammas = {}
    model._run_history = []
    return model


def test_rates_add_and_presence_is_an_or():
    model = _model_for_compound()

    rates = compound_rates(model.mu_ms, _MEMBERS)
    presence = compound_presence(
        model.dataset.variants_present, _MEMBERS
    )

    assert np.allclose(rates.values, [0.11, 0.22, 0.33, 0.44])
    assert list(presence) == [True, False, True, False]


def test_compound_fit_uses_the_summed_rate_and_ored_presence(
    monkeypatch,
):
    captured = {}

    def fake_fit(mus_yes, mus_no, **kwargs):
        captured["yes"] = dict(mus_yes)
        captured["no"] = dict(mus_no)
        return "fake_result"

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus", fake_fit
    )

    model = _model_for_compound()
    model.estimate_gamma_compound(_MEMBERS, name="KRAS codon 12")

    # T1 carries VAR1 and T3 carries VAR2, so both are "with".
    assert sorted(captured["yes"]) == ["T1", "T3"]
    assert sorted(captured["no"]) == ["T2", "T4"]
    assert np.isclose(captured["yes"]["T1"], 0.11)
    assert np.isclose(captured["no"]["T4"], 0.44)
    assert "KRAS codon 12" in model.gammas


def test_compound_records_its_members_and_accounting(monkeypatch):
    import json

    import arviz as az

    result = az.from_dict({"gamma": np.ones((2, 5))})
    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        lambda mus_yes, mus_no, **kwargs: result,
    )

    model = _model_for_compound()
    model.estimate_gamma_compound(
        _MEMBERS, excluded_samples=["T4"], store=False
    )

    attrs = result.posterior.attrs
    assert json.loads(attrs["compound_members"]) == _MEMBERS
    assert attrs["n_tumors_with"] == 2
    assert attrs["n_tumors_without"] == 1
    assert attrs["n_tumors_excluded"] == 1


def test_compound_default_name_lists_its_members(monkeypatch):
    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus",
        lambda mus_yes, mus_no, **kwargs: "fake_result",
    )
    model = _model_for_compound()
    model.estimate_gamma_compound(_MEMBERS)
    assert "VAR1 + VAR2" in model.gammas


def test_compound_rejects_empty_repeated_and_unknown_members():
    model = _model_for_compound()

    with pytest.raises(ValueError, match="needs members"):
        model.estimate_gamma_compound([])
    with pytest.raises(ValueError, match="double-count"):
        model.estimate_gamma_compound(["VAR1", "VAR1"])
    with pytest.raises(ValueError, match="not found in mutation"):
        model.estimate_gamma_compound(["VAR1", "NOPE"])
