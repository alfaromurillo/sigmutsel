"""build_full_dataset keeps a mutation database already built (and
cleaned by the two-pass signature step) unless told to rebuild it."""

import logging

import pandas as pd
import pytest

from sigmutsel.models import MutationDataset


class _Stop(Exception):
    pass


@pytest.fixture
def dataset(monkeypatch):
    """A dataset whose pipeline stops right after the mutation step,
    recording whether generate_mutation_db ran and with what."""
    ds = MutationDataset.__new__(MutationDataset)
    ds._mutation_db = None
    ds.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self._mutation_db = pd.DataFrame({"x": [1]})

    def stop(self):
        raise _Stop

    monkeypatch.setattr(
        MutationDataset, "generate_mutation_db", generate
    )
    monkeypatch.setattr(
        MutationDataset, "compute_gene_presence", stop
    )
    return ds


def _run(ds, **kwargs):
    with pytest.raises(_Stop):
        ds.build_full_dataset(**kwargs)


def test_default_builds_when_nothing_is_loaded(dataset):
    _run(dataset, qc_mode=True)
    assert dataset.calls == [{"qc_mode": True}]


def test_default_keeps_a_loaded_database(dataset, caplog):
    dataset._mutation_db = pd.DataFrame({"cleaned": [1]})
    with caplog.at_level(logging.WARNING, logger="sigmutsel.models"):
        _run(dataset, qc_mode=True)
    assert dataset.calls == []
    assert list(dataset._mutation_db.columns) == ["cleaned"]
    assert "qc_mode" in caplog.text  # its ignored arguments are named


def test_true_rebuilds_even_when_loaded(dataset):
    dataset._mutation_db = pd.DataFrame({"cleaned": [1]})
    _run(dataset, regenerate_mutation_db=True)
    assert dataset.calls == [{}]


def test_false_requires_a_loaded_database(dataset):
    with pytest.raises(ValueError, match="no mutation_db is loaded"):
        dataset.build_full_dataset(regenerate_mutation_db=False)
