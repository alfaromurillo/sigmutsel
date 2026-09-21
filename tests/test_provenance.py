"""Tests for provenance stamping and run history.

Covers what a manifest records (the package stamp), what it warns
about on load (a build change), and what ``record_call`` writes into
``_run_history`` -- including the two behaviors the mechanism exists
for: a call that *failed* still leaves a trace, and a fitting loop
cannot grow the history without bound.

The round trip through ``save_dataset``/``load_dataset`` is tested on
a real (empty) ``MutationDataset``, since the point of the feature is
that the history survives the file, not that a dict round-trips.
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from sigmutsel import provenance
from sigmutsel.models import MutationDataset
from sigmutsel.provenance import (
    MAX_RUN_HISTORY,
    check_provenance,
    format_run_history,
    package_provenance,
    record_call,
)


class _Recorded:
    """Minimal object with the two methods the decorator needs."""

    def __init__(self):
        self._run_history = []

    @record_call
    def fit(self, draws, tune=500, frame=None):
        return draws

    @record_call
    def breaks(self, why):
        raise ValueError(why)


def test_stamp_has_version_and_commit_fields():
    stamp = package_provenance()
    assert set(stamp) == {
        "sigmutsel_version",
        "sigmutsel_commit",
        "saved_at",
    }
    assert isinstance(stamp["sigmutsel_version"], str)
    # None off a git checkout; a string on one. Both are valid, so
    # the assertion is on the type, not on this machine's layout.
    assert stamp["sigmutsel_commit"] is None or isinstance(
        stamp["sigmutsel_commit"], str
    )
    assert json.loads(json.dumps(stamp)) == stamp


def test_check_provenance_silent_on_match(caplog):
    with caplog.at_level(logging.WARNING):
        check_provenance(package_provenance(), "model")
    assert caplog.text == ""


def test_check_provenance_silent_on_unstamped_manifest(caplog):
    """A manifest written before stamping existed is not a mismatch."""
    with caplog.at_level(logging.WARNING):
        check_provenance({"version": 1}, "model")
    assert caplog.text == ""


def test_check_provenance_warns_on_different_build(caplog):
    stamp = package_provenance()
    stamp["sigmutsel_commit"] = "deadbeefcafe"
    stamp["sigmutsel_version"] = "0.0.1"
    with caplog.at_level(logging.WARNING):
        check_provenance(stamp, "model", directory="/tmp/some_model")
    assert "deadbeefcafe" in caplog.text
    assert "/tmp/some_model" in caplog.text


def test_check_provenance_falls_back_to_version(caplog, monkeypatch):
    """With no commit resolvable, the version field is the check."""
    monkeypatch.setattr(provenance, "git_commit", lambda: None)
    with caplog.at_level(logging.WARNING):
        check_provenance(
            {"sigmutsel_version": "0.0.1", "sigmutsel_commit": None},
            "dataset",
        )
    assert "0.0.1" in caplog.text


def test_record_call_records_passed_arguments_only():
    obj = _Recorded()
    obj.fit(2000)

    (entry,) = obj._run_history
    assert entry["call"] == "_Recorded.fit"
    # tune was not passed, so it is not recorded -- the entry reads
    # like the call that was written.
    assert entry["args"] == {"draws": 2000}
    assert "failed" not in entry


def test_record_call_summarizes_large_values():
    obj = _Recorded()
    obj.fit(1000, frame=pd.DataFrame(np.zeros((300, 4))))

    args = obj._run_history[0]["args"]
    assert args["frame"] == "<DataFrame (300, 4)>"
    assert json.loads(json.dumps(args)) == args


def test_record_call_marks_a_failed_call():
    obj = _Recorded()
    with pytest.raises(ValueError):
        obj.breaks("diverged")

    (entry,) = obj._run_history
    assert entry["call"] == "_Recorded.breaks"
    assert entry["failed"] == "ValueError"


def test_run_history_is_bounded_and_keeps_both_ends():
    obj = _Recorded()
    for draws in range(MAX_RUN_HISTORY + 50):
        obj.fit(draws)

    history = obj._run_history
    assert len(history) <= MAX_RUN_HISTORY + 1
    assert history[0]["args"] == {"draws": 0}
    assert history[-1]["args"] == {"draws": MAX_RUN_HISTORY + 49}

    (marker,) = [e for e in history if "elided" in e]
    assert marker["elided"] == 50
    assert "50 entries elided" in "\n".join(
        format_run_history(history)
    )


def test_dataset_history_round_trips_through_a_save(tmp_path, caplog):
    maf_dir = tmp_path / "mafs"
    maf_dir.mkdir()
    dataset = MutationDataset(location_maf_files=str(maf_dir))
    provenance.record_event(dataset, "test.event", note="planted")

    saved = tmp_path / "saved_dataset"
    dataset.save_dataset(saved, overwrite=True)

    manifest = json.loads(
        (saved / "dataset_manifest.json").read_text()
    )
    assert manifest["version"] == 4
    assert manifest["sigmutsel_version"]
    assert manifest["run_history"][0]["note"] == "planted"

    with caplog.at_level(logging.WARNING):
        reloaded = MutationDataset.load_dataset(saved)
    assert reloaded.run_history[0]["call"] == "test.event"
    # Same build wrote it, so loading it says nothing about builds.
    assert "was written by sigmutsel" not in caplog.text


def test_dataset_generate_calls_are_recorded(tmp_path):
    """The decorator is actually attached to the real methods."""
    maf_dir = tmp_path / "mafs"
    maf_dir.mkdir()
    dataset = MutationDataset(location_maf_files=str(maf_dir))

    with pytest.raises(ValueError, match="Mutation database"):
        dataset.generate_variant_db(position_tolerance=7)

    (entry,) = dataset.run_history
    assert entry["call"] == "MutationDataset.generate_variant_db"
    assert entry["args"] == {"position_tolerance": 7}
    assert "failed" in entry
