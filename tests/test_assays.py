"""Per-assay territories (MutationDataset.set_assays, Model._by_assay).

On the toy two-gene annotation, two capture territories (the identity
and a partial BED with a GC curve) and three tumors, two on one assay
and one on the other:

1. each tumor's channel baselines are the ones its own assay's tables
   give it (rtol 1e-12), joined in the tumors' original order;
2. assays that share one territory reproduce the plain computation;
3. the variant draws of every assay replay the same random stream, so
   a draw is one stage-1 sample shared by all tumors;
4. ``assay_groups`` sends a tumor without an assay to the primary one.
"""

import numpy as np
import pandas as pd

from sigmutsel.capture import CaptureTerritory, capture_weights
from sigmutsel.channel_universe import (
    CHANNELS,
    compute_channel_opportunity,
)
from sigmutsel.constants import canonical_types_order
from sigmutsel.estimate_mus import compute_mu_g_channel_per_tumor
from sigmutsel.models import AssayView, Model, MutationDataset
from tests import test_channel_universe as _toy
from tests.test_channel_universe import MASK, _masked_keys, _planted

models = _toy.models  # the toy two-gene annotation, as a fixture

TUMORS = ["t1", "t2", "t3"]
GROUPS = {"A": ["t1", "t3"], "B": ["t2"]}


def _bed(tmp_path, rows, name):
    path = tmp_path / name
    path.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows))
    return str(path)


def _tables(models, capture):
    t = compute_channel_opportunity(
        models,
        masked_keys=_masked_keys(MASK),
        site_weights=_planted(),
        capture_w=capture_weights(models, capture),
    )
    t["nonsyn"] = t["mis"] + t["non"] + t["spl"]
    t["type"] = sum(t[c] for c in CHANNELS)
    return t


class _Dataset:
    """What compute_channel_base_mus reads, plus the assay hooks."""

    def __init__(self, tables, groups):
        self._t = tables
        self.groups = groups
        primary = next(iter(tables))
        for h in ("syn", "mis", "non", "spl", "esyn", "nonsyn"):
            setattr(self, f"contexts_by_gene_{h}", tables[primary][h])
        self.contexts_by_gene = tables[primary]["contexts"]
        self._contexts_by_gene = self.contexts_by_gene
        self.type_opportunity = tables[primary]["type"]
        self.germline_mask_af = None
        self.channel_universe = {}
        self.calls_masked = False

    def has_consequence_contexts_by_gene(self):
        return True

    def assay_groups(self, tumors):
        if self.groups is None:
            return None
        return {k: pd.Index(v) for k, v in self.groups.items()}

    def assay_view(self, name):
        view = AssayView(self, name)
        t = self._t[name]
        view._tables = {h: t[h] for h in AssayView._OWN}
        return view


def _model(dataset, seed=0):
    model = Model.__new__(Model)
    model.dataset = dataset
    model._mu_taus = pd.DataFrame(
        np.random.default_rng(seed).gamma(2.0, 1.0, size=(3, 96)),
        index=TUMORS,
        columns=canonical_types_order,
    )
    model._prob_g_tau_tau_independent = False
    return model


def _captures(tmp_path):
    full = CaptureTerritory(
        _bed(
            tmp_path,
            [("chr1", 0, 10**7), ("chr2", 0, 10**7)],
            "f.bed",
        ),
        outside=1.0,
        build="hg38",
    )
    partial = CaptureTerritory(
        _bed(tmp_path, [("chr1", 100, 109)], "p.bed"),
        outside=0.2,
        gc=(0.0, 1.0),
        log_g=(0.0, -2.0),
        build="hg38",
    )
    return full, partial


def test_each_tumor_gets_its_own_assays_rates(models, tmp_path):
    full, partial = _captures(tmp_path)
    tables = {
        "A": _tables(models, full),
        "B": _tables(models, partial),
    }
    model = _model(_Dataset(tables, GROUPS))
    syn, nonsyn = model.compute_channel_base_mus(False)
    assert list(nonsyn.columns) == TUMORS
    for name, tumors in GROUPS.items():
        t = tables[name]
        for got, tab in ((syn, t["syn"]), (nonsyn, t["nonsyn"])):
            want = compute_mu_g_channel_per_tumor(
                model._mu_taus.loc[tumors],
                tab,
                t["contexts"],
                type_opportunity=t["type"],
            )
            np.testing.assert_allclose(
                got[tumors].to_numpy(), want.to_numpy(), rtol=1e-12
            )
    # The two territories do give different rates (else this is empty).
    assert not np.allclose(
        nonsyn["t2"].to_numpy(),
        compute_mu_g_channel_per_tumor(
            model._mu_taus.loc[["t2"]],
            tables["A"]["nonsyn"],
            tables["A"]["contexts"],
            type_opportunity=tables["A"]["type"],
        )["t2"].to_numpy(),
    )


def test_one_shared_territory_is_the_plain_computation(
    models, tmp_path
):
    _, partial = _captures(tmp_path)
    t = _tables(models, partial)
    split = _model(_Dataset({"A": t, "B": t}, GROUPS))
    plain = _model(_Dataset({"A": t}, None))
    for a, b in zip(
        split.compute_channel_base_mus(False),
        plain.compute_channel_base_mus(False),
    ):
        np.testing.assert_allclose(
            a.to_numpy(), b.to_numpy(), rtol=1e-12
        )
        assert list(a.columns) == list(b.columns)


def test_variant_draws_share_one_stream(
    models, tmp_path, monkeypatch
):
    full, partial = _captures(tmp_path)
    tables = {
        "A": _tables(models, full),
        "B": _tables(models, partial),
    }
    model = _model(_Dataset(tables, GROUPS))

    def single(variant, r_g_variant, n_draws, rng, spectrum_draws):
        z = rng.normal(size=(n_draws, 1))
        cols = model._mu_taus.index
        return pd.DataFrame(z * np.ones((1, len(cols))), columns=cols)

    monkeypatch.setattr(
        model, "_compute_mu_m_posterior_draws_single", single
    )
    draws = model.compute_mu_m_posterior_draws(
        "v", n_draws=7, rng=np.random.default_rng(3)
    )
    assert list(draws.columns) == TUMORS
    for col in TUMORS[1:]:
        np.testing.assert_array_equal(draws[col], draws["t1"])
    np.testing.assert_array_equal(
        draws["t1"].to_numpy(),
        np.random.default_rng(3).normal(size=7),
    )


def test_assay_groups_fall_back_to_the_primary(tmp_path):
    ds = MutationDataset(location_maf_files=tmp_path)
    ds._channel_universe = {
        "assays": {
            "A": {"bed_path": __file__},
            "B": {"bed_path": __file__},
        },
        "primary_assay": "A",
    }
    ds._assay_table = pd.DataFrame(
        {"assay": ["B", "A"]}, index=pd.Index(["t2", "t1"])
    )
    groups = ds.assay_groups(["t1", "t2", "t9"])
    assert list(groups["A"]) == ["t1", "t9"]
    assert list(groups["B"]) == ["t2"]
    assert (
        MutationDataset(location_maf_files=tmp_path).assay_groups(
            ["t1"]
        )
        is None
    )


def test_a_view_shows_the_datasets_private_tables(tmp_path):
    # Model code reads private attributes (``_variant_db``,
    # ``_contexts_by_gene``) through the view: they must be the
    # dataset's, never an attribute of the view that shadows them.
    ds = MutationDataset(location_maf_files=tmp_path)
    ds._variant_db = pd.DataFrame({"x": [1]})
    view = AssayView(ds, "A")
    assert view._variant_db is ds._variant_db
    assert view._contexts_by_gene is ds._contexts_by_gene
