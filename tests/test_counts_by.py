"""Tests for MutationDataset.counts_by.

A cross-tab is easy to get subtly wrong in three ways, so each gets
a test: counting mutation rows rather than tumors, dividing
prevalence by the tumors that happen to appear in the mutation
database rather than by the group's real size, and silently
dropping tumors the grouping does not name.
"""

import pandas as pd
import pytest

from sigmutsel.models import MutationDataset

_ROWS = [
    # tumor, gene id, variant, classification
    ("T1", "ENSG_A", "A p.V1M", "Missense_Mutation"),
    ("T1", "ENSG_A", "A p.V1M", "Missense_Mutation"),
    ("T1", "ENSG_A", "A p.S2L", "Silent"),
    ("T2", "ENSG_A", "A p.V1M", "Missense_Mutation"),
    ("T3", "ENSG_B", "B p.R9H", "Missense_Mutation"),
    ("T4", "ENSG_B", "B p.R9H", "Missense_Mutation"),
]

# T5 is in the cohort and has no mutation at all; T6 has one but no
# group label.
_GROUPS = pd.Series(
    {
        "T1": "MSI",
        "T2": "MSI",
        "T3": "MSS",
        "T4": "MSS",
        "T5": "MSS",
    }
)


def _dataset(tmp_path, rows=_ROWS):
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = pd.DataFrame(
        rows,
        columns=[
            "Tumor_Sample_Barcode",
            "ensembl_gene_id",
            "variant",
            "Variant_Classification",
        ],
    )
    return dataset


def test_counts_tumors_not_mutation_rows(tmp_path):
    """T1 carries A p.V1M twice; that is one tumor."""
    counts = _dataset(tmp_path).counts_by(_GROUPS)

    assert counts.loc["A p.V1M", "MSI"] == 2
    assert counts.loc["B p.R9H", "MSS"] == 2
    assert counts.loc["A p.V1M", "MSS"] == 0


def test_gene_level_and_scope(tmp_path):
    dataset = _dataset(tmp_path)

    any_scope = dataset.counts_by(_GROUPS, level="gene")
    silent = dataset.counts_by(_GROUPS, level="gene", scope="silent")
    non_silent = dataset.counts_by(
        _GROUPS, level="gene", scope="non-silent"
    )

    assert any_scope.loc["ENSG_A", "MSI"] == 2
    assert silent.loc["ENSG_A", "MSI"] == 1
    assert "ENSG_B" not in silent.index
    assert non_silent.loc["ENSG_A", "MSI"] == 2


def test_prevalence_divides_by_the_whole_group(tmp_path):
    """T5 has no mutations but is still an MSS tumor."""
    counts = _dataset(tmp_path).counts_by(_GROUPS, prevalence=True)

    assert counts.attrs["tumors_per_group"] == {"MSI": 2, "MSS": 3}
    # 2 of 3 MSS tumors, not 2 of the 2 that appear in the database.
    assert counts.loc["B p.R9H", "MSS"] == pytest.approx(2 / 3)
    assert counts.loc["A p.V1M", "MSI"] == pytest.approx(1.0)


def test_unlabeled_tumors_are_reported_not_hidden(tmp_path):
    rows = _ROWS + [("T6", "ENSG_B", "B p.R9H", "Missense_Mutation")]
    counts = _dataset(tmp_path, rows).counts_by(_GROUPS)

    assert counts.attrs["unlabeled_tumors"] == 1
    # T6 is in neither column, which is why the count is reported.
    assert counts.loc["B p.R9H"].sum() == 2


def test_units_restriction_and_column_grouping(tmp_path):
    dataset = _dataset(tmp_path)

    restricted = dataset.counts_by(_GROUPS, units=["A p.V1M"])
    assert list(restricted.index) == ["A p.V1M"]

    by_column = dataset.counts_by(
        "Variant_Classification", level="gene"
    )
    assert by_column.loc["ENSG_A", "Silent"] == 1
    assert by_column.loc["ENSG_A", "Missense_Mutation"] == 2


def test_bad_arguments_raise(tmp_path):
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="level must be"):
        dataset.counts_by(_GROUPS, level="compound")
    with pytest.raises(ValueError, match="scope must be"):
        dataset.counts_by(_GROUPS, scope="coding")
    with pytest.raises(ValueError, match="not a column"):
        dataset.counts_by("msi_status")
