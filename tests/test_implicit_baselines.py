"""The recommended model in the tutorial's calls, with nothing chosen.

``Model(dataset)``, ``assign_cov_matrix``, ``compute_channel_base_mus()``
and ``estimate_channel_rg_cov_effects`` must give the model the
explicit calls give (``prob_g_tau_tau_independent=False``, type rates
and merged baseline computed first), keep an earlier explicit choice,
reach a variant rate, and save and load without recomputing anything.
On the toy two-gene annotation of ``test_channel_universe`` with
planted site weights.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from sigmutsel.channel_universe import CHANNELS
from sigmutsel.constants import canonical_types_order
from sigmutsel.models import Model, MutationDataset
from tests import test_site_rates as _sr
from tests.test_channel_universe import _planted

models = _sr.models  # the toy two-gene annotation, as a fixture

COV = pd.DataFrame({"x1": [0.4, -0.7]}, index=_sr.GENES)


def _dataset(models, path):
    """A channel-universe dataset that compute_mu_taus can run on."""
    tables = _sr._tables(models)
    db = _sr._every_call(models)
    db["Tumor_Sample_Barcode"] = np.resize(_sr.TUMORS, len(db))
    db["Variant_Classification"] = np.where(
        db["channel"] == "syn", "Silent", "Missense_Mutation"
    )
    dataset = MutationDataset(str(path / "mafs"))
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
    dataset._variant_db = _sr._variant_db(models, db)
    dataset.compute_gene_presence()
    dataset.compute_gene_presence_non_silent()
    dataset.compute_gene_presence_silent()
    dataset.compute_gene_counts_channels()

    rng = np.random.default_rng(0)
    emissions = rng.random((96, 2))
    emissions /= emissions.sum(axis=0)
    dataset._signature_matrix = pd.DataFrame(
        emissions,
        index=pd.Index(canonical_types_order, name="MutationType"),
        columns=["SBS1", "SBS5"],
    )
    burden = db.groupby("Tumor_Sample_Barcode").size().astype(float)
    dataset._sig_assignments = pd.DataFrame(
        {"SBS1": 0.6 * burden, "SBS5": 0.4 * burden}
    )
    return dataset


def _implicit(models, path):
    model = Model(_dataset(models, path))
    model.assign_cov_matrix(COV)
    model.compute_channel_base_mus()
    return model


def test_nothing_chosen_gives_the_explicit_model(
    models, tmp_path, caplog
):
    with caplog.at_level(logging.INFO, logger="sigmutsel.models"):
        implicit = _implicit(models, tmp_path / "a")

    explicit = Model(
        _dataset(models, tmp_path / "b"),
        None,
        cov_effects_per_sigma=False,
        prob_g_tau_tau_independent=False,
    )
    explicit.assign_cov_matrix(COV)
    explicit.compute_channel_base_mus()

    assert implicit.prob_g_tau_tau_independent is False
    for name in (
        "mu_taus",
        "base_mus",
        "base_mus_syn",
        "base_mus_nonsyn",
    ):
        pd.testing.assert_frame_equal(
            getattr(implicit, name), getattr(explicit, name)
        )
    # what was computed implicitly is said so
    assert "compute_mu_taus()" in caplog.text
    assert "compute_base_mus(prob_g_tau_tau_independent=False)" in (
        caplog.text
    )


def test_an_earlier_choice_is_kept(models, tmp_path):
    """An explicit compute_base_mus(True) before is inherited, not
    replaced by the tau-dependent default."""
    model = Model(_dataset(models, tmp_path / "a"))
    model.assign_cov_matrix(COV)
    model.compute_mu_taus()
    model.compute_base_mus(prob_g_tau_tau_independent=True)
    model.compute_channel_base_mus()

    explicit = Model(_dataset(models, tmp_path / "b"))
    explicit.assign_cov_matrix(COV)
    explicit.compute_mu_taus()
    explicit.compute_base_mus(prob_g_tau_tau_independent=True)
    explicit.compute_channel_base_mus(prob_g_tau_tau_independent=True)

    assert model.prob_g_tau_tau_independent is True
    for name in ("base_mus_syn", "base_mus_nonsyn"):
        pd.testing.assert_frame_equal(
            getattr(model, name), getattr(explicit, name)
        )


def test_site_weights_compute_the_type_rates_first(
    models, tmp_path, monkeypatch
):
    """The probe of the tutorial's step 3: Model(dataset) alone."""
    from sigmutsel import channel_universe

    class _Reached(Exception):
        pass

    def stop(*args, **kwargs):  # the exome's annotation is not here
        raise _Reached

    monkeypatch.setattr(
        channel_universe, "load_or_build_transcript_models", stop
    )
    probe = Model(_dataset(models, tmp_path))
    assert not probe.has_mu_taus()
    with pytest.raises(_Reached):
        probe.estimate_site_weights()
    assert probe.has_mu_taus()


def _full_context_dataset(path):
    """The synthetic three-gene dataset the channel fits are tested on
    (every context present, so the merged baseline is finite), with
    the signature assignments compute_mu_taus reads."""
    from tests.test_channel_cov_effects import (
        _mutation_db,
        _synthetic_opportunities,
    )
    from tests.test_estimate_rg import _synthetic_variant_db

    contexts, syn, nonsyn = _synthetic_opportunities()
    db = _mutation_db()
    db["type"] = canonical_types_order[: len(db)]
    dataset = MutationDataset(location_maf_files=path)
    dataset._mutation_db = db
    dataset._contexts_by_gene = contexts
    dataset._contexts_by_gene_syn = syn
    dataset._contexts_by_gene_nonsyn = nonsyn
    dataset.compute_gene_presence()
    dataset.compute_gene_presence_non_silent()
    dataset.compute_gene_presence_silent()
    dataset.compute_gene_counts_channels()
    dataset._variant_db = _synthetic_variant_db()

    rng = np.random.default_rng(0)
    emissions = rng.random((96, 2))
    emissions /= emissions.sum(axis=0)
    dataset._signature_matrix = pd.DataFrame(
        emissions,
        index=pd.Index(canonical_types_order, name="MutationType"),
        columns=["SBS1", "SBS5"],
    )
    burden = db.groupby("Tumor_Sample_Barcode").size().astype(float)
    dataset._sig_assignments = pd.DataFrame(
        {"SBS1": 0.6 * burden, "SBS5": 0.4 * burden}
    )
    return dataset


@pytest.mark.slow
def test_tutorial_calls_fit_save_and_load(tmp_path, monkeypatch):
    dataset = _full_context_dataset(tmp_path / "data")
    dataset.save_dataset(tmp_path / "dataset")  # as the tutorial does
    model = Model(dataset)
    model.assign_cov_matrix(
        pd.DataFrame(
            {"cov1": [0.5, -0.5, 0.1]},
            index=["ENSG_A", "ENSG_B", "ENSG_C"],
        )
    )
    model.compute_channel_base_mus()
    model.estimate_channel_rg_cov_effects(sample="MAP")

    # one variant rate, from the fitted model
    variant = model.dataset.variant_db.index[0]
    rate = model.mu_ms.loc[variant]
    assert np.isfinite(rate).all() and (rate > 0).all()

    model.save_model(tmp_path / "saved")

    def must_not_run(*args, **kwargs):
        raise AssertionError("loading recomputed a baseline")

    monkeypatch.setattr(Model, "compute_mu_taus", must_not_run)
    monkeypatch.setattr(Model, "compute_base_mus", must_not_run)
    loaded = Model.load_model(tmp_path / "saved")

    assert loaded.prob_g_tau_tau_independent is False
    for name in (
        "mu_taus",
        "base_mus",
        "base_mus_syn",
        "base_mus_nonsyn",
    ):
        pd.testing.assert_frame_equal(
            getattr(loaded, name),
            getattr(model, name),
            check_freq=False,
        )
    pd.testing.assert_series_equal(loaded.mu_ms.loc[variant], rate)
