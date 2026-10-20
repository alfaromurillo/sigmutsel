"""Tests for variant_annotation's per-variant tables."""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.variant_annotation import (
    annotate_variants_with_types,
    extract_variants_from_db,
)


def _db(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "variant",
            "gene",
            "ensembl_gene_id",
            "Chromosome",
            "Start_Position",
            "type",
        ],
    )


_DB = _db(
    [
        ("KRAS p.G12D", "KRAS", "ENSG_K", "chr12", 100, "C[C>T]A"),
        ("KRAS p.G12D", "KRAS", "ENSG_K", "chr12", 100, "C[C>T]A"),
        # Two start positions 2 bp apart: within tolerance, averaged.
        (
            "TP53 p.X1_splice",
            "TP53",
            "ENSG_T",
            "chr17",
            500,
            "A[C>A]A",
        ),
        (
            "TP53 p.X1_splice",
            "TP53",
            "ENSG_T",
            "chr17",
            502,
            "A[C>G]A",
        ),
        # 50 bp apart: beyond tolerance, no single position.
        (
            "TP53 p.X2_splice",
            "TP53",
            "ENSG_T",
            "chr17",
            600,
            "T[T>C]T",
        ),
        (
            "TP53 p.X2_splice",
            "TP53",
            "ENSG_T",
            "chr17",
            650,
            "T[T>C]T",
        ),
        # Ensembl ids disagree: the most frequent one wins.
        ("EGFR p.L858R", "EGFR", "ENSG_E", "chr7", 900, "G[T>G]G"),
        ("EGFR p.L858R", "EGFR", "ENSG_E", "chr7", 900, "G[T>G]G"),
        ("EGFR p.L858R", "EGFR", "ENSG_OLD", "chr7", 900, "G[T>G]G"),
    ]
)


def test_extract_variants_one_row_per_variant():
    out = extract_variants_from_db(_DB)

    assert out.index.name == "variant"
    assert sorted(out.index) == sorted(_DB["variant"].unique())
    assert out.columns.tolist() == [
        "gene",
        "ensembl_gene_id",
        "Chromosome",
        "Start_Position",
    ]
    assert out.loc["KRAS p.G12D"].tolist() == [
        "KRAS",
        "ENSG_K",
        "chr12",
        100,
    ]
    assert out.loc["EGFR p.L858R", "ensembl_gene_id"] == "ENSG_E"


def test_extract_variants_start_position_tolerance():
    out = extract_variants_from_db(_DB)
    assert out.loc["TP53 p.X1_splice", "Start_Position"] == 501
    assert np.isnan(out.loc["TP53 p.X2_splice", "Start_Position"])

    # A wider tolerance averages the second one too.
    wide = extract_variants_from_db(_DB, position_tolerance=50)
    assert wide.loc["TP53 p.X2_splice", "Start_Position"] == 625


def test_extract_variants_rejects_a_variant_on_two_genes():
    db = _db(
        [
            ("X p.A1B", "GENE1", "ENSG1", "chr1", 1, "A[C>T]A"),
            ("X p.A1B", "GENE2", "ENSG2", "chr1", 1, "A[C>T]A"),
        ]
    )
    with pytest.raises(ValueError, match="multiple genes"):
        extract_variants_from_db(db)


def test_annotate_variants_with_observed_types():
    variants = pd.DataFrame(
        index=pd.Index(
            ["KRAS p.G12D", "TP53 p.X1_splice", "BRAF p.V600E"],
            name="variant",
        )
    )
    out = annotate_variants_with_types(variants, _DB)

    # One type: a plain string; several: a list; unseen: NaN.
    assert out.loc["KRAS p.G12D", "mut_types"] == "C[C>T]A"
    assert out.loc["TP53 p.X1_splice", "mut_types"] == [
        "A[C>A]A",
        "A[C>G]A",
    ]
    assert pd.isna(out.loc["BRAF p.V600E", "mut_types"])
    assert "channel" not in out.columns
    assert "mut_types" not in variants.columns  # works on a copy


def test_annotate_variants_prefers_routes_and_adds_channel():
    db = _DB.assign(
        routes=lambda d: np.where(
            d["variant"] == "KRAS p.G12D",
            "C[C>T]A;C[C>A]A;C[C>T]A",
            "",
        ),
        channel="mis",
    )
    # A variant column instead of the index is accepted as the key.
    variants = pd.DataFrame(
        {"variant": ["KRAS p.G12D", "EGFR p.L858R"]}
    )
    out = annotate_variants_with_types(variants, db)

    # Every route, repeats kept; an empty route set falls back to the
    # observed types.
    assert out["mut_types"].tolist() == [
        ["C[C>T]A", "C[C>A]A", "C[C>T]A"],
        "G[T>G]G",
    ]
    assert out["channel"].tolist() == ["mis", "mis"]
