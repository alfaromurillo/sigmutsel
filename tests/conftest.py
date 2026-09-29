"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _no_installed_genome(monkeypatch):
    """Tests never read an installed reference genome.

    ``TranscriptModels.neighbours`` defaults to SigProfilerMatrixGenerator's
    GRCh38 when it is installed; the toy annotations here use made-up
    coordinates, so a machine with the genome would otherwise read real
    bases at them. A test that wants a genome passes its own.
    """
    from sigmutsel import channel_universe

    monkeypatch.setattr(
        channel_universe,
        "sigprofiler_genome_dir",
        lambda *a, **k: None,
    )
