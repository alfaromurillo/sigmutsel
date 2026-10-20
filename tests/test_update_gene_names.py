"""Tests for update_gene_names.update_genes_with_gene_set."""

import pandas as pd

from sigmutsel.update_gene_names import update_genes_with_gene_set

# A three-gene stand-in for HGNC's complete set: only the columns the
# function reads, plus one it ignores.
_HGNC = (
    "hgnc_id\tsymbol\talias_symbol\tprev_symbol\tensembl_gene_id\n"
    "HGNC:1\tKMT2D\tALR|AAD10\tMLL2|MLL4\tENSG00000167548\n"
    "HGNC:2\tTP53\tLFS1\t\tENSG00000141510\n"
    "HGNC:3\tCCN2\t\tCTGF\tENSG00000118523\n"
)


def _hgnc(tmp_path):
    path = tmp_path / "hgnc_complete_set.txt"
    path.write_text(_HGNC)
    return path


def test_aliases_and_previous_symbols_become_approved(tmp_path):
    df = pd.DataFrame(
        {
            "gene": ["MLL2", "TP53", "CTGF", "NOT_A_GENE", "AAD10"],
            "variant": [
                "MLL2 p.R1A",
                "TP53 p.R175H",
                "CTGF p.Q2*",
                "NOT_A_GENE p.A3V",
                "AAD10 p.S4L",
            ],
        }
    )
    out = update_genes_with_gene_set(_hgnc(tmp_path), df)

    assert out["gene"].tolist() == [
        "KMT2D",
        "TP53",
        "CCN2",
        "NOT_A_GENE",  # unknown symbols are kept as they are
        "KMT2D",
    ]
    assert out["variant"].tolist() == [
        "KMT2D p.R1A",
        "TP53 p.R175H",
        "CCN2 p.Q2*",
        "NOT_A_GENE p.A3V",
        "KMT2D p.S4L",
    ]
    assert out["ensembl_gene_id"].tolist()[:3] == [
        "ENSG00000167548",
        "ENSG00000141510",
        "ENSG00000118523",
    ]
    assert pd.isna(out["ensembl_gene_id"].iloc[3])
    assert "ensembl_gene_id" not in df.columns  # works on a copy


def test_existing_ensembl_ids_are_corrected_not_dropped(tmp_path):
    df = pd.DataFrame(
        {
            "gene": ["TP53", "UNKNOWN"],
            "ensembl_gene_id": ["ENSG_STALE", "ENSG_KEPT"],
        }
    )
    out = update_genes_with_gene_set(_hgnc(tmp_path), df)

    assert out["ensembl_gene_id"].tolist() == [
        "ENSG00000141510",
        "ENSG_KEPT",  # no HGNC entry to correct it from
    ]
    assert "variant" not in out.columns
