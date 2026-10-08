"""Multi-base events kept aside, and the gamma hold-out they feed.

A doublet's (or any multi-base event's) single-base rows are not SBS
calls: they leave every SBS input but are kept in
``MutationDataset.multi_base_db``, so gamma can hold out of a gene's
absent set the tumors whose hit in it was a multi-base event.
"""

import pandas as pd

from sigmutsel.models import Model, MutationDataset


def _dataset(tmp_path):
    return MutationDataset(location_maf_files=str(tmp_path))


def _calls(**columns):
    base = {
        "Tumor_Sample_Barcode": ["T1", "T2", "T3"],
        "ensembl_gene_id": ["G1", "G1", "G1"],
        "gene": ["A", "A", "A"],
        "Variant_Classification": ["Missense_Mutation"] * 3,
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_split_moves_multi_base_rows_out_of_the_mutation_table(
    tmp_path,
):
    ds = _dataset(tmp_path)
    ds._mutation_db = _calls(multi_base=[None, "dbs", "mnv"])
    ds._split_multi_base()
    assert list(ds.mutation_db["Tumor_Sample_Barcode"]) == ["T1"]
    assert "multi_base" not in ds.mutation_db.columns
    assert list(ds.multi_base_db["multi_base"]) == ["dbs", "mnv"]


def test_split_is_a_no_op_without_the_column(tmp_path):
    ds = _dataset(tmp_path)
    ds._mutation_db = _calls()
    ds._split_multi_base()
    assert len(ds.mutation_db) == 3
    assert ds.multi_base_db is None


def test_hits_count_only_non_silent_in_universe_components(tmp_path):
    ds = _dataset(tmp_path)
    ds.multi_base_db = _calls(
        Tumor_Sample_Barcode=["T1", "T2", "T3"],
        channel=["mis", "syn", "non"],
        in_universe=[True, True, False],
        multi_base=["dbs"] * 3,
    )
    assert list(ds.multi_base_hits("G1")) == ["T1"]
    assert list(ds.multi_base_hits("G2")) == []


def test_hits_without_the_table_are_empty(tmp_path):
    assert len(_dataset(tmp_path).multi_base_hits("G1")) == 0


def test_multi_base_table_survives_save_and_load(tmp_path):
    ds = _dataset(tmp_path / "mafs")
    ds._mutation_db = _calls(multi_base=[None, "dbs", "dbs"])
    ds._split_multi_base()
    ds.save_dataset(tmp_path / "ds", overwrite=True)
    back = MutationDataset.load_dataset(tmp_path / "ds")
    assert list(back.multi_base_db["Tumor_Sample_Barcode"]) == [
        "T2",
        "T3",
    ]


# --- the hold-out -------------------------------------------------------


class _FakeDataset:
    def __init__(self, hits):
        self._hits = hits

    def multi_base_hits(self, gene_id):
        return pd.Index(self._hits.get(gene_id, []))


def _fake_result():
    from tests.test_estimate_gammas import _fake_posterior_result

    return _fake_posterior_result()


def _capture(monkeypatch):
    captured = {}

    def fake_fit(mus_yes, mus_no, **kwargs):
        captured["yes"] = list(mus_yes.index)
        captured["no"] = list(mus_no.index)
        captured["result"] = _fake_result()
        return captured["result"]

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus", fake_fit
    )
    return captured


def _variant_model(hits):
    tumors = ["T1", "T2", "T3", "T4"]
    model = Model.__new__(Model)
    model.mu_ms = pd.DataFrame(
        [[0.1] * 4], index=["VAR1"], columns=tumors
    )
    model.cov_matrix = None
    ds = _FakeDataset(hits)
    ds.variants_present = pd.DataFrame(
        [[1, 0, 0, 0]], index=["VAR1"], columns=tumors
    )
    ds._variant_db = pd.DataFrame(
        {"ensembl_gene_id": ["G1"]}, index=["VAR1"]
    )
    ds.genes_present_non_silent = pd.DataFrame(
        [[1, 0, 0, 0]], index=["G1"], columns=tumors
    )
    model.dataset = ds
    model.gammas = {}
    return model


def test_variant_absent_set_loses_the_multi_base_tumors(monkeypatch):
    captured = _capture(monkeypatch)
    model = _variant_model({"G1": ["T3"], "G2": ["T4"]})
    model._estimate_gamma_variant("VAR1", store=False)
    assert captured["yes"] == ["T1"]
    assert captured["no"] == [
        "T2",
        "T4",
    ]  # T4's event is in another gene
    attrs = captured["result"].posterior.attrs
    assert attrs["n_tumors_held_out_multi_base"] == 1
    assert attrs["n_tumors_held_out"] == 1


def test_a_carrier_is_never_held_out(monkeypatch):
    captured = _capture(monkeypatch)
    model = _variant_model({"G1": ["T1"]})
    model._estimate_gamma_variant("VAR1", store=False)
    assert captured["yes"] == ["T1"]
    assert captured["no"] == ["T2", "T3", "T4"]


def test_same_gene_hold_out_counts_first(monkeypatch):
    captured = _capture(monkeypatch)
    model = _variant_model({"G1": ["T2"]})
    model.dataset.genes_present_non_silent.loc["G1", "T2"] = 1
    model._estimate_gamma_variant("VAR1", store=False)
    attrs = captured["result"].posterior.attrs
    assert captured["no"] == ["T3", "T4"]
    assert attrs["n_tumors_held_out"] == 1
    assert attrs["n_tumors_held_out_multi_base"] == 0


def test_the_hold_out_can_be_switched_off(monkeypatch):
    captured = _capture(monkeypatch)
    model = _variant_model({"G1": ["T3"]})
    model._estimate_gamma_variant(
        "VAR1", store=False, hold_out_multi_base=False
    )
    assert captured["no"] == ["T2", "T3", "T4"]


def test_gene_absent_set_loses_the_multi_base_tumors(monkeypatch):
    captured = _capture(monkeypatch)
    tumors = ["T1", "T2", "T3", "T4"]
    model = Model.__new__(Model)
    model._mu_gs = pd.DataFrame(
        [[0.5] * 4], index=["G1"], columns=tumors
    )
    model._base_mus = None
    model._base_mus_syn = None
    model._base_mus_nonsyn = None
    model.cov_matrix = None
    ds = _FakeDataset({"G1": ["T2", "T3"]})
    ds.mutation_db = pd.DataFrame(columns=["gene", "ensembl_gene_id"])
    ds.genes_present_non_silent = pd.DataFrame(
        [[1, 0, 0, 0]], index=["G1"], columns=tumors
    )
    model.dataset = ds
    model.gammas = {}
    model._estimate_gamma_gene("G1", store=False)
    assert captured["yes"] == ["T1"]
    assert captured["no"] == ["T4"]
    attrs = captured["result"].posterior.attrs
    assert attrs["n_tumors_held_out_multi_base"] == 2
    assert attrs["n_tumors_excluded"] == 0
