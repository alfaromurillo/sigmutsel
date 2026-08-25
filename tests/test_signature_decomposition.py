"""Tests for signature_decomposition.py's cancer-type-table resolution.

Covers the shared row-matching/combination helpers (including the
ESCA-style ambiguous-TCGA-code case) and resolve_exclusion_list's
treatment_naive / exclude_artifacts combinator, using small synthetic
tables rather than the real (large, network-dependent) reference
files.
"""

import pandas as pd
import pytest

from sigmutsel.signature_decomposition import (
    _expand_subvariants,
    _match_cancer_type_rows,
    _signatures_from_rows,
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
