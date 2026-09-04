"""Tests for gene-level cross-validation: the ``train_genes``/``genes``
restriction parameters on ``Model.estimate_cov_effects``/
``Model.estimate_passenger_genes_r2``/
``Model.estimate_channel_rg_cov_effects``, and the
``gene_cv_passenger_r2``/``channel_gene_cv_passenger_r2`` wrappers
that orchestrate them (see TODO.md's "Cross-validated R² test for PCA
n_components selection", and the sigmutsel branch audit that flagged
this as a currently-broken dependency of
tcga_analysis/code/pca_nc_cv_sweep.py)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sigmutsel.cross_validation import (
    channel_gene_cv_passenger_r2,
    gene_cv_passenger_r2,
)
from sigmutsel.models import Model, MutationDataset


def _minimal_model_for_cov_effects(tmp_path):
    """A bare Model with just enough state to exercise
    estimate_cov_effects' gene-filtering steps, with compute_mu_gs/
    compute_mu_ms/estimate_passenger_genes_r2 stubbed out (Step 5's
    automatic recomputation) since this helper is only used to test
    which genes reach the fit, not the fit's own numerical output."""
    tumors = ["T1", "T2", "T3"]
    genes = ["ENSG_A", "ENSG_B", "ENSG_C"]
    genes_present = pd.DataFrame(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]], index=genes, columns=tumors
    )
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._genes_present = genes_present

    base_mus = pd.DataFrame(
        [[0.1, 0.2, 0.3]] * 3, index=genes, columns=tumors
    )
    cov_matrix = pd.DataFrame({"cov1": [0.5, 0.6, 0.7]}, index=genes)

    model = Model.__new__(Model)
    model.dataset = dataset
    model._base_mus = base_mus
    model.cov_matrix = cov_matrix
    model.cov_effects_kwargs = {}
    model._mu_gs = None
    model._passenger_genes_r2 = None
    model.compute_mu_gs = lambda *a, **k: None
    model.compute_mu_ms = lambda *a, **k: None
    model.estimate_passenger_genes_r2 = lambda *a, **k: None
    return model


def _fake_estimate_covariates_effect(
    mus, presence_matrix, cov_matrix, **kw
):
    """Stand-in for the real PyMC-backed fit: records nothing here
    directly, just returns a MAP-shaped result dict so
    estimate_cov_effects' Step 4/5 bookkeeping doesn't crash. Callers
    inspect the ``cov_matrix`` argument's row count (the number of
    genes that reached the fit) via a wrapping mock/side_effect.

    Patched with autospec=True below: estimate_cov_effects introspects
    the real function's signature via inspect.signature() before
    calling it (to resolve default bounds), so a plain Mock without a
    matching signature breaks that lookup."""
    return {"c": np.zeros(cov_matrix.shape[1] + 1)}


def test_estimate_cov_effects_train_genes_restricts_fit_genes(
    tmp_path,
):
    """train_genes should intersect with passenger_genes_complete
    before the arrays reach estimate_covariates_effect -- this is the
    train-fold half of gene_cv_passenger_r2's mechanism."""
    model = _minimal_model_for_cov_effects(tmp_path)

    with patch(
        "sigmutsel.estimate_covariates_effect.estimate_covariates_effect",
        side_effect=_fake_estimate_covariates_effect,
        autospec=True,
    ) as mocked:
        model.estimate_cov_effects(train_genes=["ENSG_A", "ENSG_B"])

    n_genes_fit = mocked.call_args.kwargs["cov_matrix"].shape[0]
    assert n_genes_fit == 2
    assert model._n_in_cov_effects_estimation == 2


def test_estimate_cov_effects_train_genes_none_uses_all_passenger_genes(
    tmp_path,
):
    """Omitting train_genes must reproduce today's behavior exactly:
    every passenger gene with complete covariate data is used."""
    model = _minimal_model_for_cov_effects(tmp_path)

    with patch(
        "sigmutsel.estimate_covariates_effect.estimate_covariates_effect",
        side_effect=_fake_estimate_covariates_effect,
        autospec=True,
    ) as mocked:
        model.estimate_cov_effects()

    n_genes_fit = mocked.call_args.kwargs["cov_matrix"].shape[0]
    assert n_genes_fit == 3
    assert model._n_in_cov_effects_estimation == 3


def test_estimate_cov_effects_train_genes_outside_universe_is_dropped(
    tmp_path,
):
    """A train_genes ID that isn't a passenger gene with complete
    covariates (typo, driver gene, wrong cohort) must simply be
    excluded via the intersection, not raise or get included."""
    model = _minimal_model_for_cov_effects(tmp_path)

    with patch(
        "sigmutsel.estimate_covariates_effect.estimate_covariates_effect",
        side_effect=_fake_estimate_covariates_effect,
        autospec=True,
    ) as mocked:
        model.estimate_cov_effects(
            train_genes=["ENSG_A", "ENSG_NOT_IN_MODEL"]
        )

    n_genes_fit = mocked.call_args.kwargs["cov_matrix"].shape[0]
    assert n_genes_fit == 1


def _minimal_model_for_r2(tmp_path):
    tumors = ["T1", "T2", "T3"]
    genes_present = pd.DataFrame(
        [[1, 0, 1], [0, 1, 1], [1, 1, 1]],
        index=["ENSG_A", "ENSG_B", "ENSG_C"],
        columns=tumors,
    )
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._genes_present = genes_present

    mu_gs = pd.DataFrame(
        [[0.5, 0.1, 0.5], [0.1, 0.5, 0.5], [0.3, 0.3, 0.3]],
        index=["ENSG_A", "ENSG_B", "ENSG_C"],
        columns=tumors,
    )

    model = Model.__new__(Model)
    model.dataset = dataset
    model._mu_gs = mu_gs
    model._passenger_genes_r2 = None
    return model


def test_passenger_genes_r2_genes_restricts_scored_genes(tmp_path):
    """genes=[...] must score only the intersection of that set with
    the passenger-gene universe -- the held-out-fold half of
    gene_cv_passenger_r2's mechanism. Checked by comparing against a
    model built with the other genes already dropped, the same
    equivalence-to-never-having-it pattern used for excluded_samples
    in test_contexts_by_gene.py."""
    model_restricted = _minimal_model_for_r2(tmp_path)
    r2_restricted = model_restricted.estimate_passenger_genes_r2(
        genes=["ENSG_A", "ENSG_B"]
    )

    model_never_had_c = _minimal_model_for_r2(tmp_path)
    model_never_had_c._mu_gs = model_never_had_c._mu_gs.loc[
        ["ENSG_A", "ENSG_B"]
    ]
    r2_reference = model_never_had_c.estimate_passenger_genes_r2()

    assert np.isclose(r2_restricted, r2_reference)


def test_passenger_genes_r2_genes_none_matches_today(tmp_path):
    """Omitting genes (default None) must reproduce today's
    unrestricted behavior exactly."""
    model_default = _minimal_model_for_r2(tmp_path)
    r2_default = model_default.estimate_passenger_genes_r2()

    model_explicit_all = _minimal_model_for_r2(tmp_path)
    r2_explicit_all = model_explicit_all.estimate_passenger_genes_r2(
        genes=["ENSG_A", "ENSG_B", "ENSG_C"]
    )
    assert r2_default == r2_explicit_all


class _StubModel:
    """Duck-typed stand-in for Model, recording which gene sets
    gene_cv_passenger_r2 sends to each fit/score call without running
    any real (expensive) fitting -- isolates the wrapper's fold-
    orchestration logic from the fit/eval numerics already covered by
    the estimate_cov_effects/estimate_passenger_genes_r2 tests above.
    """

    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
        self.train_gene_calls = []
        self.test_gene_calls = []

    def estimate_cov_effects(
        self, train_genes=None, excluded_samples=None
    ):
        self.train_gene_calls.append(set(train_genes))

    def estimate_passenger_genes_r2(
        self, genes=None, excluded_samples=None
    ):
        genes = set(genes)
        self.test_gene_calls.append(genes)
        return float(len(genes))  # deterministic fake "r2" per fold


def test_gene_cv_passenger_r2_folds_partition_gene_universe():
    genes = [f"ENSG_{i:03d}" for i in range(10)]
    cov_matrix = pd.DataFrame({"cov1": range(10)}, index=genes)
    model = _StubModel(cov_matrix)

    result = gene_cv_passenger_r2(model, n_splits=5, random_state=0)

    assert result["n_genes"] == 10
    assert len(result["fold_r2"]) == 5

    all_test_genes = set()
    for train, test in zip(
        model.train_gene_calls, model.test_gene_calls
    ):
        assert train & test == set()  # disjoint within a fold
        assert train | test == set(
            genes
        )  # together, the full universe
        assert not (all_test_genes & test)  # disjoint across folds
        all_test_genes |= test
    assert all_test_genes == set(
        genes
    )  # every gene scored exactly once


def test_gene_cv_passenger_r2_aggregates_fold_scores():
    genes = [f"ENSG_{i:03d}" for i in range(10)]
    cov_matrix = pd.DataFrame({"cov1": range(10)}, index=genes)
    model = _StubModel(cov_matrix)

    result = gene_cv_passenger_r2(model, n_splits=5, random_state=0)

    assert result["mean"] == pytest.approx(np.mean(result["fold_r2"]))
    assert result["std"] == pytest.approx(np.std(result["fold_r2"]))


def test_gene_cv_passenger_r2_requires_cov_matrix():
    model = _StubModel(cov_matrix=None)
    with pytest.raises(ValueError, match="cov_matrix"):
        gene_cv_passenger_r2(model)


# --- channel-split (consequence-split, shared r_g) model ---


class _StubChannelModel:
    """Duck-typed stand-in for Model's channel-split entry points,
    mirroring _StubModel above but for
    estimate_channel_rg_cov_effects/estimate_passenger_genes_r2's
    "non_silent_counts" target -- isolates
    channel_gene_cv_passenger_r2's fold-orchestration logic from the
    real (expensive) PyMC fit."""

    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
        self.train_gene_calls = []
        self.test_gene_calls = []
        self.target_calls = []

    def estimate_channel_rg_cov_effects(
        self, train_genes=None, excluded_samples=None
    ):
        self.train_gene_calls.append(set(train_genes))

    def compute_r_g_for_evaluation(self):
        return pd.Series(dtype=float)

    def estimate_passenger_genes_r2(
        self,
        excluded_samples=None,
        target="any",
        gene_scaling=None,
        genes=None,
    ):
        genes = set(genes)
        self.target_calls.append(target)
        self.test_gene_calls.append(genes)
        return float(len(genes))  # deterministic fake "r2" per fold


def test_channel_gene_cv_passenger_r2_folds_partition_gene_universe():
    genes = [f"ENSG_{i:03d}" for i in range(10)]
    cov_matrix = pd.DataFrame({"cov1": range(10)}, index=genes)
    model = _StubChannelModel(cov_matrix)

    result = channel_gene_cv_passenger_r2(
        model, n_splits=5, random_state=0
    )

    assert result["n_genes"] == 10
    assert len(result["fold_r2"]) == 5
    assert all(t == "non_silent_counts" for t in model.target_calls)

    all_test_genes = set()
    for train, test in zip(
        model.train_gene_calls, model.test_gene_calls
    ):
        assert train & test == set()
        assert train | test == set(genes)
        assert not (all_test_genes & test)
        all_test_genes |= test
    assert all_test_genes == set(genes)


def test_channel_gene_cv_passenger_r2_requires_cov_matrix():
    model = _StubChannelModel(cov_matrix=None)
    with pytest.raises(ValueError, match="cov_matrix"):
        channel_gene_cv_passenger_r2(model)


def test_channel_gene_statistics_train_genes_restricts_non_silent_only():
    """train_genes on _channel_gene_statistics should zero out the
    non-silent channel for held-out genes while leaving them (and any
    driver genes) present in the silent channel -- see the method's
    train_genes docstring for why dropping them entirely would be
    wrong."""
    genes = ["ENSG_A", "ENSG_B", "ENSG_C"]
    tumors = ["T1", "T2"]
    cov_matrix = pd.DataFrame({"cov1": [0.1, 0.2, 0.3]}, index=genes)

    dataset = MutationDataset.__new__(MutationDataset)
    dataset._genes_counts_silent = pd.DataFrame(
        [[1, 0], [0, 1], [2, 1]], index=genes, columns=tumors
    )
    dataset._genes_counts_non_silent = pd.DataFrame(
        [[3, 1], [1, 2], [0, 1]], index=genes, columns=tumors
    )

    model = Model.__new__(Model)
    model.cov_matrix = cov_matrix
    model.dataset = dataset
    model._base_mus_syn = pd.DataFrame(
        [[0.01, 0.02]] * 3, index=genes, columns=tumors
    )
    model._base_mus_nonsyn = pd.DataFrame(
        [[0.03, 0.04]] * 3, index=genes, columns=tumors
    )

    stats = model._channel_gene_statistics(
        include_drivers=True, train_genes=["ENSG_A"]
    )

    # ENSG_A (trained on) keeps its real non-silent counts/baseline;
    # ENSG_B/ENSG_C (held out) are zeroed on the non-silent channel
    # only, while all three remain in the silent channel's gene index.
    assert set(stats["genes"]) == set(genes)
    assert stats["counts_non_silent"]["ENSG_A"] == 4  # 3 + 1
    assert stats["counts_non_silent"]["ENSG_B"] == 0
    assert stats["counts_non_silent"]["ENSG_C"] == 0
    assert stats["counts_silent"]["ENSG_B"] == 1  # still counted
    assert stats["counts_silent"]["ENSG_C"] == 3
