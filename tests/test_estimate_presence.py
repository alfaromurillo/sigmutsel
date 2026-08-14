"""Tests for estimate_presence.compute_genes_present."""

import pandas as pd

from sigmutsel.estimate_presence import compute_genes_present

# SAMPLE_C has one mutation, but it's Silent -- a non-silent scope
# filter drops that row entirely, and crosstab drops columns for
# tumors absent from the filtered input. Without reindexing to the
# full tumor list, SAMPLE_C would vanish from the non-silent matrix
# instead of appearing as an all-zero column, silently shrinking the
# sample universe relative to mu_gs/genes_present (scope=None).
_DB = pd.DataFrame(
    {
        "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG1"],
        "Tumor_Sample_Barcode": ["SAMPLE_A", "SAMPLE_B", "SAMPLE_C"],
        "Variant_Classification": [
            "Missense_Mutation",
            "Missense_Mutation",
            "Silent",
        ],
    }
)


def test_non_silent_scope_keeps_full_tumor_universe():
    present = compute_genes_present(_DB, scope="non-silent")

    assert list(present.columns) == [
        "SAMPLE_A",
        "SAMPLE_B",
        "SAMPLE_C",
    ]
    assert present.loc["ENSG1", "SAMPLE_C"] == 0


def test_all_scope_matches_full_tumor_universe():
    present = compute_genes_present(_DB)

    assert set(present.columns) == {
        "SAMPLE_A",
        "SAMPLE_B",
        "SAMPLE_C",
    }
    assert present.loc["ENSG1", "SAMPLE_C"] == 1
