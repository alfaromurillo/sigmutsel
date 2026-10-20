"""Tests for estimate_presence.compute_genes_present and
compute_variants_present."""

import pandas as pd

from sigmutsel.estimate_presence import (
    compute_genes_present,
    compute_variants_present,
)

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


# --- compute_variants_present -----------------------------------------

_VARIANT_DB = pd.DataFrame(
    {
        "Tumor_Sample_Barcode": ["T2", "T2", "T1", "T3", "T1"],
        # T2 carries KRAS G12D twice (two calls, one presence); T3's
        # only call has no protein-level label.
        "variant": [
            "KRAS p.G12D",
            "KRAS p.G12D",
            "TP53 p.R175H",
            None,
            "KRAS p.G12D",
        ],
    }
)


def test_variants_present_is_a_0_1_matrix_over_every_tumor():
    wanted = pd.DataFrame(
        index=["TP53 p.R175H", "BRAF p.V600E", "KRAS p.G12D"]
    )
    present = compute_variants_present(_VARIANT_DB, wanted)

    assert present.index.tolist() == wanted.index.tolist()
    # Every tumor is a column, sorted, even T3 with no labelled call.
    assert present.columns.tolist() == ["T1", "T2", "T3"]
    assert (present.dtypes == "uint8").all()
    assert present.to_numpy().tolist() == [
        [1, 0, 0],
        [0, 0, 0],  # not in the db
        [1, 1, 0],  # T2's two calls count once
    ]


def test_variants_present_reads_only_in_universe_unmasked_calls():
    db = _VARIANT_DB.assign(
        in_universe=[True, True, False, True, True],
        germline_masked=[False, False, False, False, True],
    )
    wanted = pd.DataFrame(index=["KRAS p.G12D", "TP53 p.R175H"])
    present = compute_variants_present(db, wanted)

    # T1's TP53 call is out of the universe and its KRAS call is on a
    # germline-masked allele; T1 stays a column all the same.
    assert present.columns.tolist() == ["T1", "T2", "T3"]
    assert present.to_numpy().tolist() == [[0, 1, 0], [0, 0, 0]]
