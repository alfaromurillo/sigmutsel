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
    # test's fake artifact signature name instead. No treatment
    # signatures in play for this test, and treatment_naive defaults
    # to True, so pin TREATMENT_ASSOCIATED_SIGNATURES to empty too --
    # otherwise pass B's exclusion list would pick up the real
    # constant list and the assertion below would need to know it.
    monkeypatch.setattr(
        "sigmutsel.constants.ARTIFACT_SIGNATURES", ["SIG_ARTIFACT"]
    )
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES", []
    )

    # S1: 8 mutations from SIG_A, 2 from SIG_ARTIFACT (raw counts).
    pass_a_assignments = pd.DataFrame(
        {"SIG_A": [8], "SIG_ARTIFACT": [2]}, index=["S1"]
    )

    def fake_pass_a(self, *args, **kwargs):
        assert kwargs["exclude_artifacts"] is False
        # Pass A must be fully unrestricted: neither of these two
        # keys should ever reach it, even though this test's call
        # below doesn't pass them either -- locks in the contract.
        assert "exclude_signature_subgroups" not in kwargs
        assert "treatment_naive" not in kwargs
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
        # exclude_artifacts is never passed to pass B any more -- the
        # final exclusion list is built explicitly and passed as
        # exclude_signature_subgroups instead, so exclude_artifacts
        # (a no-op on an explicit list) would be misleading.
        assert "exclude_artifacts" not in kwargs
        assert kwargs["exclude_signature_subgroups"] == [
            "SIG_ARTIFACT"
        ]
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


def test_two_pass_pass_a_ignores_caller_supplied_table_and_treatment(
    tmp_path, monkeypatch
):
    """Even when the caller passes exclude_signature_subgroups/
    treatment_naive (the normal main.py-style call), pass A must never
    see them -- it has to stay fully unrestricted to serve as steps 2
    and 3's diagnostic fit. Pass B, however, must still end up with
    the correctly resolved final exclusion list."""
    dataset = _make_dataset(tmp_path)
    monkeypatch.setattr(
        "sigmutsel.constants.ARTIFACT_SIGNATURES", ["SIG_ARTIFACT"]
    )
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES",
        ["SIG_TREAT"],
    )
    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.resolve_exclusion_list",
        lambda cancer_type, location=None, available_sigs=None, treatment_naive=True, exclude_artifacts=False: (
            ["SIG_TABLE"] if cancer_type == "FAKE_TYPE" else []
        ),
    )

    pass_a_assignments = pd.DataFrame(
        {"SIG_A": [8], "SIG_ARTIFACT": [2]}, index=["S1"]
    )

    def fake_pass_a(self, *args, **kwargs):
        assert "exclude_signature_subgroups" not in kwargs
        assert "treatment_naive" not in kwargs
        self._sig_assignments = pass_a_assignments
        self._signature_matrix = _SIG_MATRIX
        return pass_a_assignments

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )

    pass_b_calls = []

    def fake_pass_b(**kwargs):
        pass_b_calls.append(kwargs)
        return pd.DataFrame({"SIG_A": [8]}, index=["S1"])

    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        fake_pass_b,
    )

    dataset.run_two_pass_signature_decomposition(
        exclude_signature_subgroups="FAKE_TYPE",
        treatment_naive=True,
    )

    assert len(pass_b_calls) == 1
    assert sorted(pass_b_calls[0]["exclude_signature_subgroups"]) == [
        "SIG_ARTIFACT",
        "SIG_TABLE",
        "SIG_TREAT",
    ]


def test_two_pass_treatment_load_qc_drops_flagged_sample(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(tmp_path)
    dataset._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S2", "S2"],
            "type": ["A[C>A]A"] * 4,
            "Variant_Classification": ["Missense_Mutation"] * 4,
        }
    )
    monkeypatch.setattr("sigmutsel.constants.ARTIFACT_SIGNATURES", [])
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES",
        ["SIG_TREAT"],
    )

    # S1: 90% treatment load (flagged at threshold=0.2); S2: 10%
    # (not flagged). Both well above min_burden_for_diagnostics=5.
    pass_a_assignments = pd.DataFrame(
        {"SIG_A": [1, 9], "SIG_TREAT": [9, 1]}, index=["S1", "S2"]
    )
    sig_matrix = pd.DataFrame(
        {
            "MutationType": ["A[C>A]A"],
            "SIG_A": [1.0],
            "SIG_TREAT": [1.0],
        }
    ).set_index("MutationType")

    def fake_pass_a(self, *args, **kwargs):
        self._sig_assignments = pass_a_assignments
        self._signature_matrix = sig_matrix
        return pass_a_assignments

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )

    pass_b_calls = []

    def fake_pass_b(**kwargs):
        pass_b_calls.append(kwargs)
        matrix = pd.read_csv(
            kwargs["input_data"], sep="\t", index_col=0
        )
        # S1 was dropped entirely; only S2 survives into pass B.
        assert list(matrix.columns) == ["S2"]
        return pd.DataFrame({"SIG_A": [9]}, index=["S2"])

    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        fake_pass_b,
    )

    dataset.run_two_pass_signature_decomposition(
        treatment_load_threshold=0.2,
        min_burden_for_diagnostics=5,
    )

    assert len(pass_b_calls) == 1
    assert set(dataset.mutation_db["Tumor_Sample_Barcode"]) == {"S2"}


def test_two_pass_treatment_load_qc_ignores_low_burden_samples(
    tmp_path, monkeypatch
):
    """A sample below min_burden_for_diagnostics must never be
    flagged, however high its treatment-load fraction looks -- that
    fraction is NNLS noise at low mutation counts, not a trustworthy
    signal (see mutation_rates TODO.md's 2026-08-25 groundwork)."""
    dataset = _make_dataset(tmp_path)
    dataset._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1"],
            "type": ["A[C>A]A"] * 2,
            "Variant_Classification": ["Missense_Mutation"] * 2,
        }
    )
    monkeypatch.setattr("sigmutsel.constants.ARTIFACT_SIGNATURES", [])
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES",
        ["SIG_TREAT"],
    )

    # S1: 100% treatment load, but only 2 mutations total -- below
    # min_burden_for_diagnostics=5.
    pass_a_assignments = pd.DataFrame(
        {"SIG_TREAT": [2]}, index=["S1"]
    )
    sig_matrix = pd.DataFrame(
        {"MutationType": ["A[C>A]A"], "SIG_TREAT": [1.0]}
    ).set_index("MutationType")

    def fake_pass_a(self, *args, **kwargs):
        self._sig_assignments = pass_a_assignments
        self._signature_matrix = sig_matrix
        return pass_a_assignments

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )

    def fake_pass_b(**kwargs):
        matrix = pd.read_csv(
            kwargs["input_data"], sep="\t", index_col=0
        )
        assert list(matrix.columns) == ["S1"]
        return pd.DataFrame({"SIG_TREAT": [2]}, index=["S1"])

    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        fake_pass_b,
    )

    dataset.run_two_pass_signature_decomposition(
        treatment_load_threshold=0.2,
        min_burden_for_diagnostics=5,
    )

    assert set(dataset.mutation_db["Tumor_Sample_Barcode"]) == {"S1"}


def test_two_pass_prevalence_override_keeps_recurring_signature(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(tmp_path)
    dataset._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S2", "S3"],
            "type": ["A[C>A]A"] * 3,
            "Variant_Classification": ["Missense_Mutation"] * 3,
        }
    )
    monkeypatch.setattr("sigmutsel.constants.ARTIFACT_SIGNATURES", [])
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES", []
    )
    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.resolve_exclusion_list",
        lambda cancer_type, location=None, available_sigs=None, treatment_naive=True, exclude_artifacts=False: [
            "SIG_RECURRING"
        ],
    )

    # SIG_RECURRING appears (at low magnitude) in 2/3 adequately-
    # powered samples -- above a 50% prevalence bar, so it should be
    # kept in pass B's basis despite the table excluding it.
    pass_a_assignments = pd.DataFrame(
        {
            "SIG_A": [8, 9, 10],
            "SIG_RECURRING": [2, 1, 0],
        },
        index=["S1", "S2", "S3"],
    )
    sig_matrix = pd.DataFrame(
        {
            "MutationType": ["A[C>A]A"],
            "SIG_A": [1.0],
            "SIG_RECURRING": [1.0],
        }
    ).set_index("MutationType")

    def fake_pass_a(self, *args, **kwargs):
        self._sig_assignments = pass_a_assignments
        self._signature_matrix = sig_matrix
        return pass_a_assignments

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )

    pass_b_calls = []

    def fake_pass_b(**kwargs):
        pass_b_calls.append(kwargs)
        return pd.DataFrame(
            {"SIG_A": [8, 9, 10]}, index=["S1", "S2", "S3"]
        )

    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        fake_pass_b,
    )

    dataset.run_two_pass_signature_decomposition(
        exclude_signature_subgroups="FAKE_TYPE",
        prevalence_override_min_fraction=0.5,
        min_burden_for_diagnostics=1,
    )

    assert (
        "SIG_RECURRING"
        not in pass_b_calls[0]["exclude_signature_subgroups"]
    )


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
