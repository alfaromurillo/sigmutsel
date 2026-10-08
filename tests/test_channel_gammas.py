"""Per-channel gene gammas: missense, truncating, synonymous control."""

import pandas as pd

from sigmutsel.models import GAMMA_CHANNELS, Model

TUMORS = ["T1", "T2", "T3", "T4"]


def _fake_result():
    from tests.test_estimate_gammas import _fake_posterior_result

    return _fake_posterior_result()


def _capture(monkeypatch):
    captured = {}

    def fake_fit(mus_yes, mus_no, **kwargs):
        captured["yes"] = list(mus_yes.index)
        captured["no"] = list(mus_no.index)
        captured["kwargs"] = kwargs
        captured["result"] = _fake_result()
        return captured["result"]

    monkeypatch.setattr(
        "sigmutsel.estimate_gammas.estimate_gamma_from_mus", fake_fit
    )
    return captured


class _Dataset:
    def __init__(self):
        self.mutation_db = pd.DataFrame(
            columns=["gene", "ensembl_gene_id"]
        )
        self.genes_present_non_silent = pd.DataFrame(
            [[1, 1, 0, 0]], index=["G1"], columns=TUMORS
        )
        # T1 has a missense call, T2 a nonsense one; T3 a synonymous one.
        self._by_channel = {
            "mis": [1, 0, 0, 0],
            "trunc": [0, 1, 0, 0],
            "syn": [0, 0, 1, 0],
        }

    def genes_present_channel(self, channel):
        return pd.DataFrame(
            [self._by_channel[channel]], index=["G1"], columns=TUMORS
        )

    def multi_base_hits(self, gene_id):
        return pd.Index(["T4"])

    def homdel_hits(self, gene_id):
        return pd.Index([])


def _model():
    model = Model.__new__(Model)
    model._mu_gs = pd.DataFrame(
        [[0.5] * 4], index=["G1"], columns=TUMORS
    )
    model._base_mus_nonsyn = pd.DataFrame(
        [[0.1] * 4], index=["G1"], columns=TUMORS
    )
    model._base_mus_syn = pd.DataFrame(
        [[0.05] * 4], index=["G1"], columns=TUMORS
    )
    model.cov_matrix = None
    model.dataset = _Dataset()
    model.gammas = {}
    model.compute_channel_mu_gs = lambda channel: pd.DataFrame(
        [[0.01] * 4], index=["G1"], columns=TUMORS
    )
    return model


def test_channel_presence_counts_only_that_channel(monkeypatch):
    captured = _capture(monkeypatch)
    model = _model()
    model._estimate_gamma_gene("G1", store=True, channel="trunc")
    assert captured["yes"] == ["T2"]
    # T1's missense hit is not held out; T4's doublet is.
    assert captured["no"] == ["T1", "T3"]
    assert "G1__trunc" in model.gammas
    attrs = captured["result"].posterior.attrs
    assert attrs["channel"] == "trunc"
    assert attrs["n_tumors_held_out_multi_base"] == 1


def test_synonymous_control_holds_nobody_out(monkeypatch):
    captured = _capture(monkeypatch)
    model = _model()
    model._estimate_gamma_gene("G1", store=False, channel="syn")
    assert captured["yes"] == ["T3"]
    assert captured["no"] == ["T1", "T2", "T4"]


def test_a_channel_inherits_the_gene_dispersion_as_shapes(
    monkeypatch,
):
    captured = _capture(monkeypatch)
    model = _model()
    model._resolve_gene_tumor_dispersion = lambda *a: 8.0
    model._estimate_gamma_gene(
        "G1", store=False, channel="mis", gene_tumor_dispersion=8.0
    )
    yes, no = captured["kwargs"]["gene_tumor_shape"]
    assert "gene_tumor_dispersion" not in captured["kwargs"]
    # phi * p_gj with an even allocation over the 3 tumors kept.
    assert list(yes) == [8.0 / 3] and list(no) == [8.0 / 3, 8.0 / 3]


def test_channel_baselines_sum_parts_and_cache(monkeypatch):
    calls = []

    def fake_channel_rates(**kwargs):
        part = kwargs["channel_contexts_by_gene"]
        calls.append(part)
        return pd.DataFrame(
            [[{"mis": 3.0, "non": 1.0, "spl": 0.5}[part]] * 4],
            index=["G1"],
            columns=TUMORS,
        )

    monkeypatch.setattr(
        "sigmutsel.estimate_mus.compute_mu_g_channel_per_tumor",
        fake_channel_rates,
    )
    model = _model()
    model._mu_taus = object()
    model._prob_g_tau_tau_independent = False
    model.dataset.contexts_by_gene = None
    model.dataset.contexts_by_gene_mis = "mis"
    model.dataset.contexts_by_gene_non = "non"
    model.dataset.contexts_by_gene_spl = "spl"
    trunc = model._channel_baseline("trunc")
    assert trunc.loc["G1", "T1"] == 1.5
    model._channel_baseline("non")  # cached, not recomputed
    assert calls == ["non", "spl"]
    assert set(GAMMA_CHANNELS["nonsyn"]) == {"mis", "non", "spl"}


def test_variant_level_refuses_a_channel():
    import pytest

    model = _model()
    model.mu_ms = pd.DataFrame(
        [[0.1] * 4], index=["V1"], columns=TUMORS
    )
    with pytest.raises(ValueError, match="gene gammas only"):
        model.estimate_gamma("V1", level="variant", channel="mis")
