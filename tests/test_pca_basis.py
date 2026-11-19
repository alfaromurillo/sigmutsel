"""The covariate PCA is exact by default, and a model saves its basis.

A model's components must be reproducible from its inputs and equal
to any larger decomposition's leading ones (a published one, say), so
the default solver is the exact one; and the basis -- loadings,
centres and scales, which genes were complete -- is saved with the
model, since a reader cannot place a gene or a column in the
components without it.
"""

import numpy as np
import pandas as pd

from sigmutsel.channel_universe import (
    sigprofiler_genome_dir as _real_genome_dir,
)
from sigmutsel.models import Model, MutationDataset
from sigmutsel.utils import run_pca_on_covariates

from .test_covariate_fallback import (
    _FALLBACK,
    _pca_fallback_model,
    _pca_imputed_model,
)


def _wide(n=120, p=600, seed=0):
    """Wide enough (p > 500) that scikit-learn's own default would
    pick its randomized solver."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 8))
    x = latent @ rng.normal(size=(8, p)) + 0.3 * rng.normal(
        size=(n, p)
    )
    return pd.DataFrame(
        x,
        index=[f"G{i}" for i in range(n)],
        columns=[f"c{j}" for j in range(p)],
    )


def test_components_do_not_depend_on_how_many_are_asked_for():
    df = _wide()
    small = run_pca_on_covariates(df, n_components=5)
    large = run_pca_on_covariates(df, n_components=60)
    assert small.attrs["svd_solver"] == "full"
    np.testing.assert_allclose(
        small.to_numpy(), large.iloc[:, :5].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        small.attrs["components"],
        large.attrs["components"][:5],
        atol=1e-12,
    )


def test_an_explicit_solver_is_kept():
    pcs = run_pca_on_covariates(
        _wide(), n_components=5, svd_solver="randomized"
    )
    assert pcs.attrs["svd_solver"] == "randomized"
    assert pcs.attrs["random_state"] == 0


def _scores_from_basis(raw, basis):
    table = basis["columns"]
    z = (raw[table.index] - table["center"]) / table["scale"]
    z = z.fillna(0.0) - table["pca_mean"].to_numpy()
    return z.to_numpy() @ basis["loadings"].to_numpy(dtype=float).T


def test_imputed_basis_reproduces_the_scores(tmp_path):
    model, raw = _pca_imputed_model(tmp_path)
    basis = model.pca_basis
    assert basis["loadings"].dtypes.eq(np.float32).all()
    assert list(basis["loadings"].index) == ["PC1"]
    assert list(basis["loadings"].columns) == ["a1", "a2", "b1"]
    genes = basis["genes"]
    assert list(genes.index) == list(model.cov_matrix.index)
    assert not genes.loc[_FALLBACK, "complete"]
    assert genes.drop(index=_FALLBACK)["complete"].all()
    meta = basis["meta"]
    assert meta["incomplete"] == "impute"
    assert meta["svd_solver"] == "full"
    assert meta["dropna"] == "none"
    assert meta["indicator_columns"] == ["miss_b"]
    # Mean-filled before standardizing, so a missing value is 0.
    filled = raw.reindex(model.cov_matrix.index)
    filled = filled.fillna(filled.mean())
    np.testing.assert_allclose(
        basis["columns"]["center"], filled.mean()
    )
    np.testing.assert_allclose(
        basis["columns"]["scale"], filled.std()
    )
    np.testing.assert_allclose(
        _scores_from_basis(
            raw.reindex(model.cov_matrix.index), basis
        ).ravel(),
        model.cov_matrix["PC1"],
        atol=1e-6,
    )


def test_projected_basis_lists_the_complete_genes(tmp_path):
    model = _pca_fallback_model(tmp_path)
    genes = model.pca_basis["genes"]
    assert list(genes.index) == ["ENSG_A", "ENSG_B"]
    assert genes["complete"].all()
    assert model.pca_basis["meta"]["incomplete"] == "project"
    assert model.pca_basis["meta"]["dropna"] == "any"


def test_no_basis_without_a_pca(tmp_path):
    model, raw = _pca_imputed_model(tmp_path)
    model.assign_cov_matrix(raw)
    assert model.pca_basis is None


def test_basis_survives_save_and_load(tmp_path):
    model, _ = _pca_imputed_model(tmp_path / "work")
    model.dataset.save_dataset(tmp_path / "ds")
    model.dataset = MutationDataset.load_dataset(tmp_path / "ds")
    saved = model.pca_basis
    model.save_model(tmp_path / "model")
    assert (
        tmp_path / "model" / "pca_basis" / "loadings.parquet"
    ).exists()

    loaded = Model.load_model(tmp_path / "model").pca_basis
    pd.testing.assert_frame_equal(
        loaded["loadings"], saved["loadings"]
    )
    pd.testing.assert_frame_equal(
        loaded["columns"], saved["columns"], check_freq=False
    )
    pd.testing.assert_frame_equal(loaded["genes"], saved["genes"])
    assert loaded["meta"] == saved["meta"]


def test_a_missing_genome_is_warned_about_once(monkeypatch, tmp_path):
    import types

    import pytest

    from sigmutsel import channel_universe as cu

    fake = types.ModuleType("SigProfilerMatrixGenerator")
    fake.__file__ = str(
        tmp_path / "SigProfilerMatrixGenerator" / "__init__.py"
    )
    monkeypatch.setitem(
        __import__("sys").modules, "SigProfilerMatrixGenerator", fake
    )
    monkeypatch.setattr(cu, "_GENOME_WARNED", False)
    with pytest.warns(
        RuntimeWarning, match="No installed GRCh38 genome"
    ):
        assert _real_genome_dir() is None
    import warnings as w

    with w.catch_warnings():
        w.simplefilter("error")
        assert _real_genome_dir() is None
