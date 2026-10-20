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


@pytest.fixture(autouse=True)
def _sequential_pymc_chains(monkeypatch):
    """Run PyMC's chains in this process rather than forked workers.

    ``pm.sample`` starts one forked worker per chain by default, and
    by then the test process is multi-threaded (JAX, imported by
    other tests), so every fork warns that it may deadlock. With
    ``cores=1`` the chains run one after another from the same
    per-chain seeds, so the draws are identical; the gamma tests take
    some seconds longer in all. A test that passes ``cores`` itself
    keeps its value; the package's own sampling defaults are
    unchanged.
    """
    import pymc

    sample = pymc.sample

    def sample_sequentially(*args, **kwargs):
        kwargs.setdefault("cores", 1)
        return sample(*args, **kwargs)

    monkeypatch.setattr(pymc, "sample", sample_sequentially)
