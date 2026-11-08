"""One gene per position (channel_universe.position_owners).

On the toy annotation with gene B moved so that three of its coding
positions coincide with gene A's: a contested position has one owner,
the loser's opportunity drops by exactly those sites, and a call on a
contested position labelled with the loser moves to the owner.
"""

import copy

import numpy as np
import pandas as pd

from sigmutsel.channel_universe import (
    compute_channel_opportunity,
    credit_calls_to_owners,
    gene_priority,
    position_owners,
)
from tests import test_channel_universe as _toy

models = _toy.models  # the toy two-gene annotation, as a fixture
N_SHARED = 3


def _overlapping(models, name_a=None, name_b=None):
    m = copy.deepcopy(models)
    m.directory = None
    a, b = m.gene_index["ENSGA"], m.gene_index["ENSGB"]
    m.selection = m.selection.copy()
    m.selection.loc["ENSGB", "chrom"] = "chr1"
    if name_a:
        m.selection.loc["ENSGA", "gene_name"] = name_a
    if name_b:
        m.selection.loc["ENSGB", "gene_name"] = name_b
    m.chrom_of = m.selection["chrom"].to_numpy()[m.gene_of]
    ia = np.flatnonzero((m.gene_of == a) & (m.gpos >= 0))
    ib = np.flatnonzero((m.gene_of == b) & (m.gpos >= 0))
    m.gpos[ib] = 50_000 + np.arange(len(ib))
    m.gpos[ib[1 : 1 + N_SHARED]] = m.gpos[ia[1 : 1 + N_SHARED]]
    sp = m.splice.copy()
    sp.loc[sp["gene"] == b, "gpos"] += 70_000
    m.splice = sp
    return m, a, b, ia, ib


def test_priority_order(models):
    m, a, b, *_ = _overlapping(
        models, name_a="AC013489.1", name_b="DET1"
    )
    r = gene_priority(m)
    assert r[b] < r[a]  # a symbol beats a clone-based name
    # A readthrough: every part of the hyphenated name is a gene.
    m, a, b, *_ = _overlapping(models, name_a="GB-GB", name_b="GB")
    r = gene_priority(m)
    assert r[b] < r[a]
    # Not a readthrough: "A" is no gene, so the rule does not apply and
    # the clone-based name loses to it.
    m, a, b, *_ = _overlapping(
        models, name_a="HLA-A", name_b="AL121899.2"
    )
    r = gene_priority(m)
    assert r[a] < r[b]


def test_old_style_clone_names_lose_to_a_symbol(models):
    m, a, b, *_ = _overlapping(
        models, name_a="RP11-2C24.9", name_b="GB"
    )
    r = gene_priority(m)
    assert r[b] < r[a]


def test_one_owner_per_contested_position(models, tmp_path):
    m, _a, b, ia, ib = _overlapping(
        models, name_a="AC013489.1", name_b="GB"
    )
    own = position_owners(m)
    # B (a proper symbol) wins over A (clone-based).
    shared_a, shared_b = ia[1 : 1 + N_SHARED], ib[1 : 1 + N_SHARED]
    assert not own["coding_owned"][shared_a].any()
    assert own["coding_owned"][shared_b].all()
    rest = np.setdiff1d(ia, shared_a)
    assert own["coding_owned"][rest].all()
    assert own["splice_owned"].all()
    assert len(own["keys"]) == 2 * N_SHARED
    assert (own["owner"] == b).all()


def test_the_loser_loses_exactly_those_sites(models):
    m, _a, _b, _ia, _ib = _overlapping(
        models, name_a="AC013489.1", name_b="GB"
    )
    own = position_owners(m)
    full = compute_channel_opportunity(m)
    kept = compute_channel_opportunity(
        m, coding_in=own["coding_owned"]
    )
    id_a, id_b = "ENSGA", "ENSGB"
    assert full["contexts"].loc[id_a].sum() - kept["contexts"].loc[
        id_a
    ].sum() == float(N_SHARED)
    np.testing.assert_allclose(
        full["contexts"].loc[id_b].to_numpy(),
        kept["contexts"].loc[id_b].to_numpy(),
    )


def test_calls_move_to_the_owner(models):
    m, _a, _b, ia, _ib = _overlapping(
        models, name_a="AC013489.1", name_b="GB"
    )
    pos_shared = int(m.gpos[ia[1]])
    pos_a_only = int(m.gpos[ia[-1]])
    db = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "1"],
            "Start_Position": [pos_shared, pos_a_only, pos_shared],
            "ensembl_gene_id": ["ENSGA", "ENSGA", "ENSGB"],
            "gene": ["AC013489.1", "AC013489.1", "GB"],
        }
    )
    got, moved = credit_calls_to_owners(db, m)
    assert moved == 1
    assert list(got["ensembl_gene_id"]) == ["ENSGB", "ENSGA", "ENSGB"]
    assert list(got["gene"]) == ["GB", "AC013489.1", "GB"]
    assert list(got["ensembl_gene_id_maf"]) == [
        "ENSGA",
        "ENSGA",
        "ENSGB",
    ]
    # Idempotent: it starts again from the MAF's labels.
    again, moved_again = credit_calls_to_owners(got, m)
    assert moved_again == 1
    pd.testing.assert_series_equal(
        again["ensembl_gene_id"], got["ensembl_gene_id"]
    )
