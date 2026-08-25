"""Tests for signature_decomposition.py's cancer-type-table resolution.

Covers the shared row-matching/combination helpers (including the
ESCA-style ambiguous-TCGA-code case) and resolve_exclusion_list's
treatment_naive / exclude_artifacts combinator, using small synthetic
tables rather than the real (large, network-dependent) reference
files.
"""

import pandas as pd
import pytest

from sigmutsel.constants import canonical_types_order
from sigmutsel.signature_decomposition import (
    _expand_subvariants,
    _match_cancer_type_rows,
    _signatures_from_rows,
    _write_filtered_signature_database,
    build_sbs96_matrix_from_mutation_db,
    resolve_exclusion_list,
)


def _exclusion_table(rows):
    """Build a small exclusion-semantics table (value=1 means exclude).

    rows : dict of {pcawg_label: (applicable_tcga, {sig: 0/1, ...})}
    """
    sig_names = sorted(
        {sig for _, sigs in rows.values() for sig in sigs}
    )
    records = []
    for pcawg, (tcga, sigs) in rows.items():
        record = {
            "PCAWG": pcawg,
            "Applicable_TCGA": tcga,
            "Number_of_tumors": 10,
            "Description": pcawg,
        }
        record.update({sig: sigs.get(sig, 0) for sig in sig_names})
        records.append(record)
    return pd.DataFrame(records)


# --- _match_cancer_type_rows / _signatures_from_rows ----------------------


def test_match_cancer_type_rows_single_match():
    df = _exclusion_table(
        {
            "ColoRect-AdenoCA": (
                "COAD, READ",
                {"SBS1": 0, "SBS2": 1},
            ),
        }
    )
    rows = _match_cancer_type_rows(df, "COAD")
    assert len(rows) == 1
    assert rows.iloc[0]["PCAWG"] == "ColoRect-AdenoCA"


def test_match_cancer_type_rows_no_match_raises():
    df = _exclusion_table(
        {"ColoRect-AdenoCA": ("COAD, READ", {"SBS1": 0})}
    )
    with pytest.raises(ValueError):
        _match_cancer_type_rows(df, "TGCT")


def test_match_cancer_type_rows_exact_pcawg_label_unambiguous():
    # Regression test for the ESCA bug: an exact PCAWG-label match
    # must resolve to exactly that row even when the ambiguous TCGA
    # code would match multiple rows.
    df = _exclusion_table(
        {
            "Eso-AdenoCA": ("ESCA", {"SBS29": 1}),
            "Eso-SCC": ("ESCA", {"SBS29": 0}),
        }
    )
    rows = _match_cancer_type_rows(df, "Eso-SCC")
    assert len(rows) == 1
    assert rows.iloc[0]["PCAWG"] == "Eso-SCC"


def test_match_cancer_type_rows_ambiguous_tcga_code_returns_all():
    df = _exclusion_table(
        {
            "Eso-AdenoCA": ("ESCA", {"SBS29": 1}),
            "Eso-SCC": ("ESCA", {"SBS29": 0}),
        }
    )
    rows = _match_cancer_type_rows(df, "ESCA")
    assert set(rows["PCAWG"]) == {"Eso-AdenoCA", "Eso-SCC"}


def test_signatures_from_rows_exclusion_is_intersection():
    # SBS29 marked excluded (1) only in Eso-AdenoCA, not Eso-SCC --
    # under exclusion semantics, the ambiguous "ESCA" combination
    # should NOT exclude it (intersection), since it's plausible for
    # at least one of the two histologies actually present.
    df = _exclusion_table(
        {
            "Eso-AdenoCA": ("ESCA", {"SBS29": 1, "SBS4": 1}),
            "Eso-SCC": ("ESCA", {"SBS29": 0, "SBS4": 1}),
        }
    )
    rows = _match_cancer_type_rows(df, "ESCA")
    sig_cols = [c for c in df.columns if c.startswith("SBS")]
    excluded = _signatures_from_rows(rows, sig_cols, "exclusion")
    assert "SBS29" not in excluded  # not excluded in both rows
    assert "SBS4" in excluded  # excluded in both rows


def test_signatures_from_rows_inclusion_is_union():
    df = _exclusion_table(
        {
            "Eso-AdenoCA": ("ESCA", {"SBS29": 1, "SBS4": 0}),
            "Eso-SCC": ("ESCA", {"SBS29": 0, "SBS4": 0}),
        }
    )
    rows = _match_cancer_type_rows(df, "ESCA")
    sig_cols = [c for c in df.columns if c.startswith("SBS")]
    included = _signatures_from_rows(rows, sig_cols, "inclusion")
    assert "SBS29" in included  # marked 1 in at least one row
    assert "SBS4" not in included  # marked 0 in every row


def test_signatures_from_rows_bad_semantics_raises():
    df = _exclusion_table({"X": ("Y", {"SBS1": 1})})
    rows = _match_cancer_type_rows(df, "Y")
    with pytest.raises(ValueError):
        _signatures_from_rows(rows, ["SBS1"], "bogus")


# --- _expand_subvariants ---------------------------------------------------


def test_expand_subvariants():
    available = ["SBS10a", "SBS10b", "SBS10c", "SBS10d", "SBS11"]
    expanded = _expand_subvariants(["SBS10"], available)
    assert set(expanded) == {
        "SBS10",
        "SBS10a",
        "SBS10b",
        "SBS10c",
        "SBS10d",
    }


def test_expand_subvariants_no_subvariants_present():
    expanded = _expand_subvariants(["SBS11"], ["SBS11", "SBS12"])
    assert expanded == ["SBS11"]


# --- resolve_exclusion_list -------------------------------------------------


@pytest.fixture
def exclusion_table_path(tmp_path, monkeypatch):
    df = _exclusion_table(
        {
            "ColoRect-AdenoCA": (
                "COAD, READ",
                {"SBS4": 0, "SBS11": 1, "SBS45": 1},
            ),
        }
    )
    path = tmp_path / "exclusion.txt"
    df.to_csv(path, sep="\t", index=False)

    # Patch the artifact/treatment constants so this test is
    # self-contained and doesn't depend on the real (network-scraped)
    # cosmic_signature_etiology.tsv contents.
    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.ARTIFACT_SIGNATURES",
        ["SBS45"],
    )
    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.TREATMENT_ASSOCIATED_SIGNATURES",
        ["SBS31"],
    )
    return path


def test_resolve_exclusion_list_table_only(exclusion_table_path):
    excluded = resolve_exclusion_list(
        "COAD",
        location=exclusion_table_path,
        treatment_naive=False,
        exclude_artifacts=False,
    )
    assert "SBS11" in excluded  # table says exclude
    assert "SBS4" not in excluded  # table says keep
    assert (
        "SBS45" not in excluded
    )  # artifact carve-out: never via table
    assert "SBS31" not in excluded  # treatment_naive is False


def test_resolve_exclusion_list_treatment_naive_adds_treatment_sigs(
    exclusion_table_path,
):
    excluded = resolve_exclusion_list(
        "COAD",
        location=exclusion_table_path,
        treatment_naive=True,
        exclude_artifacts=False,
    )
    assert "SBS31" in excluded
    assert (
        "SBS45" not in excluded
    )  # still not excluded: exclude_artifacts=False


def test_resolve_exclusion_list_exclude_artifacts_adds_artifact_sigs(
    exclusion_table_path,
):
    excluded = resolve_exclusion_list(
        "COAD",
        location=exclusion_table_path,
        treatment_naive=False,
        exclude_artifacts=True,
    )
    assert "SBS45" in excluded


def test_resolve_exclusion_list_artifact_carve_out_survives_table(
    exclusion_table_path,
):
    """The table itself marks SBS45 (an artifact) excluded=1 for
    COAD -- this must never leak through unless exclude_artifacts is
    explicitly True, regardless of what the table says."""
    excluded = resolve_exclusion_list(
        "COAD",
        location=exclusion_table_path,
        treatment_naive=True,
        exclude_artifacts=False,
    )
    assert "SBS45" not in excluded


# --- build_sbs96_matrix_from_mutation_db ------------------------------


def test_build_sbs96_matrix_has_all_96_types_and_correct_counts(
    tmp_path,
):
    mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S1", "S2"],
            "type": [
                "A[C>A]A",
                "A[C>A]A",
                "A[C>A]C",
                "A[C>A]A",
            ],
        }
    )
    out = tmp_path / "matrix.txt"
    build_sbs96_matrix_from_mutation_db(mutation_db, out)

    matrix = pd.read_csv(out, sep="\t", index_col=0)
    assert matrix.shape == (96, 2)
    assert set(matrix.index) == set(canonical_types_order)
    assert matrix.loc["A[C>A]A", "S1"] == 2
    assert matrix.loc["A[C>A]C", "S1"] == 1
    assert matrix.loc["A[C>A]A", "S2"] == 1
    # types with zero mutations in either sample are present as 0,
    # not silently dropped from the matrix
    assert matrix.loc["A[C>A]C", "S2"] == 0
    assert matrix.loc["T[T>G]T", "S1"] == 0


def test_build_sbs96_matrix_creates_parent_dirs(tmp_path):
    mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1"],
            "type": ["A[C>A]A"],
        }
    )
    out = tmp_path / "nested" / "dir" / "matrix.txt"
    result_path = build_sbs96_matrix_from_mutation_db(
        mutation_db, out
    )
    assert result_path == out
    assert out.exists()


# --- _write_filtered_signature_database -------------------------------


def _fake_reference_db():
    return pd.DataFrame(
        {
            "SBS1": [0.1, 0.2],
            "SBS4": [0.3, 0.1],
            "SBS25": [0.5, 0.6],
            "SBS45": [0.9, 0.05],
        },
        index=["A[C>A]A", "A[C>A]C"],
    )


def test_write_filtered_signature_database_exclude_mode(tmp_path):
    base = _fake_reference_db()
    out = tmp_path / "filtered.txt"
    keep = [c for c in base.columns if c not in {"SBS25", "SBS45"}]
    result_path = _write_filtered_signature_database(base, keep, out)

    written = pd.read_csv(result_path, sep="\t", index_col=0)
    assert list(written.columns) == ["SBS1", "SBS4"]
    assert "SBS25" not in written.columns
    assert "SBS45" not in written.columns
    # values preserved exactly, not renormalized
    assert written.loc["A[C>A]A", "SBS1"] == 0.1


def test_write_filtered_signature_database_keep_columns_not_in_base_ignored(
    tmp_path,
):
    base = _fake_reference_db()
    out = tmp_path / "filtered.txt"
    # "SBS999" doesn't exist in base -- should be silently ignored,
    # same convention as SigProfilerAssignment's own
    # processAvg.drop(..., errors="ignore").
    _write_filtered_signature_database(base, ["SBS1", "SBS999"], out)
    written = pd.read_csv(out, sep="\t", index_col=0)
    assert list(written.columns) == ["SBS1"]


def test_write_filtered_signature_database_empty_result_raises(
    tmp_path,
):
    base = _fake_reference_db()
    out = tmp_path / "filtered.txt"
    with pytest.raises(ValueError, match="No signatures remain"):
        _write_filtered_signature_database(base, ["SBS999"], out)


def test_write_filtered_signature_database_creates_parent_dirs(
    tmp_path,
):
    base = _fake_reference_db()
    out = tmp_path / "nested" / "dir" / "filtered.txt"
    _write_filtered_signature_database(base, ["SBS1"], out)
    assert out.exists()
