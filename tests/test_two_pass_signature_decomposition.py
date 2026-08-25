"""Tests for MutationDataset.run_two_pass_signature_decomposition.

Mocks out the two actual SigProfilerAssignment fits (pass A via
`run_signature_decomposition`, pass B via the module-level
`signature_decomposition` function) since those need real
SigProfilerAssignment/network-fetched COSMIC signature files -- this
covers the orchestration logic itself: mutation_db gets cleaned before
pass B, pass B's input matrix reflects the cleaned mutation set, and
the dataset's final state (mutation_db / sig_assignments /
signature_matrix) is pass B's, not pass A's.
"""

import pandas as pd
import pytest

from sigmutsel.models import MutationDataset

# SIG_A produces only A[C>A]A; SIG_ARTIFACT produces only A[C>A]C.
# Two real canonical SBS96 types so build_sbs96_matrix_from_mutation_db's
# reindex doesn't silently zero everything out.
_SIG_MATRIX = pd.DataFrame(
    {
        "MutationType": ["A[C>A]A", "A[C>A]C"],
        "SIG_A": [1.0, 0.0],
        "SIG_ARTIFACT": [0.0, 1.0],
    }
).set_index("MutationType")


def _make_dataset(tmp_path):
    dataset = MutationDataset(
        location_maf_files=tmp_path, signature_class="SBS"
    )
    dataset._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S1"],
            "type": ["A[C>A]A", "A[C>A]A", "A[C>A]C"],
            "Variant_Classification": ["Missense_Mutation"] * 3,
        }
    )
    return dataset


def test_two_pass_cleans_mutation_db_and_uses_pass_b_results(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(tmp_path)
    # run_two_pass_signature_decomposition looks up the real
    # ARTIFACT_SIGNATURES (SBS27, SBS43, ...) -- point it at this
    # test's fake artifact signature name instead.
    monkeypatch.setattr(
        "sigmutsel.constants.ARTIFACT_SIGNATURES", ["SIG_ARTIFACT"]
    )

    # S1: 8 mutations from SIG_A, 2 from SIG_ARTIFACT (raw counts).
    pass_a_assignments = pd.DataFrame(
        {"SIG_A": [8], "SIG_ARTIFACT": [2]}, index=["S1"]
    )

    def fake_pass_a(self, *args, **kwargs):
        assert kwargs["exclude_artifacts"] is False
        self._sig_assignments = pass_a_assignments
        self._signature_matrix = _SIG_MATRIX
        return pass_a_assignments

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )

    pass_b_assignments = pd.DataFrame({"SIG_A": [8]}, index=["S1"])
    pass_b_calls = []

    def fake_pass_b(**kwargs):
        pass_b_calls.append(kwargs)
        assert kwargs["exclude_artifacts"] is True
        assert kwargs["input_type"] == "matrix"
        # The matrix file passed to pass B must reflect the cleaned
        # mutation set: A[C>A]A count 2 (both surviving rows), no
        # A[C>A]C mutations left (the artifact-attributed one was
        # dropped).
        matrix = pd.read_csv(
            kwargs["input_data"], sep="\t", index_col=0
        )
        assert matrix.loc["A[C>A]A", "S1"] == 2
        assert matrix.loc["A[C>A]C", "S1"] == 0
        return pass_b_assignments

    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        fake_pass_b,
    )

    result = dataset.run_two_pass_signature_decomposition(
        artifact_threshold=0.5
    )

    assert len(pass_b_calls) == 1
    # A[C>A]C's mutation had artifact-probability mass 1.0 (only
    # SIG_ARTIFACT could produce it) -- dropped from mutation_db.
    assert len(dataset.mutation_db) == 2
    assert (dataset.mutation_db["type"] == "A[C>A]C").sum() == 0
    assert result.equals(pass_b_assignments)
    assert dataset.sig_assignments.equals(pass_b_assignments)
    # Pass B's signature matrix file doesn't exist in this test (no
    # real fit ran) -- signature_matrix should be None, not stale
    # pass-A data.
    assert dataset._signature_matrix is None


def test_two_pass_requires_mutation_db_first(tmp_path):
    dataset = MutationDataset(
        location_maf_files=tmp_path, signature_class="SBS"
    )
    with pytest.raises(ValueError, match="mutation_db"):
        dataset.run_two_pass_signature_decomposition()


def test_two_pass_rejects_exclude_artifacts_kwarg(tmp_path):
    dataset = _make_dataset(tmp_path)
    with pytest.raises(TypeError, match="exclude_artifacts"):
        dataset.run_two_pass_signature_decomposition(
            exclude_artifacts=True
        )


def test_two_pass_only_supports_sbs(tmp_path):
    dataset = MutationDataset(
        location_maf_files=tmp_path, signature_class="DBS"
    )
    dataset._mutation_db = pd.DataFrame(
        {"Tumor_Sample_Barcode": ["S1"], "type": ["AA>CC"]}
    )
    with pytest.raises(NotImplementedError, match="SBS"):
        dataset.run_two_pass_signature_decomposition()
