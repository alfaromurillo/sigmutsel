"""The decomposition cache is valid only for its input and settings.

It used to be existence-only: a rebuilt matrix (other calls, after a
QC change) or a changed setting silently got the old exposures.
"""

import pandas as pd

from sigmutsel import signature_decomposition as sd


def _fake_run(calls):
    def run(samples, output, **kwargs):
        calls.append(kwargs)
        activities = (
            sd.Path(output) / "Assignment_Solution" / "Activities"
        )
        activities.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"Samples": ["t1"], "SBS1": [len(calls)]}
        ).to_csv(
            activities / "Assignment_Solution_Activities.txt",
            sep="\t",
            index=False,
        )

    return run


def test_cache_follows_the_input_and_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        sd, "run_signature_decomposition", _fake_run(calls)
    )
    matrix = tmp_path / "matrix.txt"
    matrix.write_text("MutationType\tt1\nA[C>A]A\t3\n")
    out = tmp_path / "results"

    def fit(**kw):
        return sd.signature_decomposition(
            str(out), str(matrix), cosmic_version=3.6, **kw
        )

    assert fit().at["t1", "SBS1"] == 1
    assert fit().at["t1", "SBS1"] == 1  # same input: loaded
    matrix.write_text("MutationType\tt1\nA[C>A]A\t2\n")
    assert fit().at["t1", "SBS1"] == 2  # other calls: refitted
    assert fit(exome=True).at["t1", "SBS1"] == 3  # other setting
    assert len(calls) == 3

    # A cache written before fingerprints existed is refitted once.
    (out / "Assignment_Solution" / "input_fingerprint.txt").unlink()
    assert fit(exome=True).at["t1", "SBS1"] == 4
    assert fit(exome=True).at["t1", "SBS1"] == 4


def test_all_zero_tumor_gets_a_zero_row(tmp_path, monkeypatch):
    # SigProfilerAssignment leaves out a tumor whose matrix column is
    # all zeros (its only SNV outside the channel universe), and
    # signature attribution then raised a KeyError on it.
    calls = []
    monkeypatch.setattr(
        sd, "run_signature_decomposition", _fake_run(calls)
    )
    matrix = tmp_path / "matrix.txt"
    matrix.write_text("MutationType\tt1\tt0\nA[C>A]A\t3\t0\n")

    out = sd.signature_decomposition(
        str(tmp_path / "results"), str(matrix), cosmic_version=3.6
    )

    assert list(out.index) == ["t1", "t0"]
    assert out.at["t0", "SBS1"] == 0
    assert out.at["t1", "SBS1"] == 1
