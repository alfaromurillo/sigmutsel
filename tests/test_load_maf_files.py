"""Tests for load_maf_files.py's compact_data().

Focused on the columns compact_data keeps -- in particular that
t_depth/t_ref_count/t_alt_count survive compaction (needed downstream
for per-sample VAF-based QC, see sample_qc.py), not a full test of
mutation-type derivation (covered indirectly elsewhere).
"""

import pandas as pd

from sigmutsel.load_maf_files import compact_data


def _valid_snv_row(**overrides):
    row = {
        "Tumor_Sample_Barcode": "TCGA-AA-0001-01A",
        "Chromosome": "chr1",
        "Variant_Classification": "Missense_Mutation",
        "Start_Position": 1000,
        "Hugo_Symbol": "TP53",
        "Gene": "ENSG00000141510",
        "HGVSp_Short": "p.R175H",
        "CONTEXT": "AAAAACAAAAA",
        "Reference_Allele": "C",
        "Tumor_Seq_Allele2": "T",
        "t_depth": 80,
        "t_ref_count": 60,
        "t_alt_count": 20,
    }
    row.update(overrides)
    return row


def test_compact_data_keeps_vaf_columns():
    df = pd.DataFrame([_valid_snv_row()])

    out = compact_data(df)

    for col in ("t_depth", "t_ref_count", "t_alt_count"):
        assert col in out.columns
    assert out["t_depth"].iloc[0] == 80
    assert out["t_ref_count"].iloc[0] == 60
    assert out["t_alt_count"].iloc[0] == 20


def test_compact_data_still_builds_expected_columns():
    df = pd.DataFrame([_valid_snv_row()])

    out = compact_data(df)

    for col in ("gene", "ensembl_gene_id", "variant", "type"):
        assert col in out.columns
    assert out["gene"].iloc[0] == "TP53"
    assert out["variant"].iloc[0] == "TP53 p.R175H"
