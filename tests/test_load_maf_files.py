"""Tests for load_maf_files.py's compact_data().

Focused on the columns compact_data keeps -- in particular that
t_depth/t_ref_count/t_alt_count survive compaction (needed downstream
for per-sample VAF-based QC, see sample_qc.py), not a full test of
mutation-type derivation (covered indirectly elsewhere).
"""

import pandas as pd
import pytest

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


# --- process_single_maf: multi-base rows kept aside ------------------------


def _write_maf(path, rows):
    df = pd.DataFrame(rows)
    df["Variant_Type"] = "SNP"
    df.to_csv(path, sep="\t", index=False)
    return path


def test_multi_base_rows_are_dropped_by_default(tmp_path):
    from sigmutsel.load_maf_files import process_single_maf

    maf = _write_maf(
        tmp_path / "a.maf",
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1001),
            _valid_snv_row(Start_Position=5000),
        ],
    )
    out = process_single_maf(maf, qc_mode=True)
    assert list(out["Start_Position"]) == [5000]
    assert "multi_base" not in out.columns


def test_multi_base_rows_are_kept_aside_when_asked(tmp_path):
    from sigmutsel.load_maf_files import process_single_maf

    maf = _write_maf(
        tmp_path / "a.maf",
        [
            _valid_snv_row(Start_Position=1000),
            _valid_snv_row(Start_Position=1001),
            _valid_snv_row(Start_Position=3000),
            _valid_snv_row(Start_Position=3002),
            _valid_snv_row(Start_Position=5000),
        ],
    )
    out = process_single_maf(maf, qc_mode=True, keep_multi_base=True)
    kinds = dict(zip(out["Start_Position"], out["multi_base"]))
    assert kinds[1000] == kinds[1001] == "dbs"
    assert kinds[3000] == kinds[3002] == "mnv"
    assert pd.isna(kinds[5000])


# --- validators and type builders (pure DataFrame -> DataFrame) ----------


def _rows(*overrides):
    return pd.DataFrame([_valid_snv_row(**o) for o in overrides])


def test_validate_chromosome_drops_unknown_names():
    from sigmutsel.load_maf_files import validate_chromosome

    df = _rows({}, {"Chromosome": "1"}, {"Chromosome": "chrUn_x"})
    out = validate_chromosome(df)
    assert out.index.tolist() == [0]


def test_validate_alleles_snv_keeps_single_nucleotides():
    from sigmutsel.load_maf_files import validate_alleles_snv

    df = _rows(
        {},
        {"Reference_Allele": "c", "Tumor_Seq_Allele2": "t"},
        {"Reference_Allele": "CA"},
        {"Tumor_Seq_Allele2": "-"},
        {"Tumor_Seq_Allele2": "N"},
    )
    out = validate_alleles_snv(df)
    # Case-insensitive on the letters, one base on each side.
    assert out.index.tolist() == [0, 1]


def test_validate_context_snv_checks_length_and_letters():
    from sigmutsel.load_maf_files import validate_context_snv

    df = _rows(
        {},
        {"CONTEXT": "aaaaacaaaaa"},
        {"CONTEXT": "AAAAACAAAA"},
        {"CONTEXT": "AAAAACNAAAA"},
    )
    assert validate_context_snv(df).index.tolist() == [0, 1]
    assert validate_context_snv(
        df, context_length=10
    ).index.tolist() == [2]


def test_validate_context_snv_on_an_empty_frame():
    from sigmutsel.load_maf_files import validate_context_snv

    df = _rows({}).iloc[:0]
    out = validate_context_snv(df)
    assert out.empty
    assert out.columns.tolist() == df.columns.tolist()


def test_validate_reference_matches_context_centre():
    from sigmutsel.load_maf_files import (
        validate_reference_matches_context_snv,
    )

    df = _rows(
        {},
        {"Reference_Allele": "G"},
        {"CONTEXT": "AAAACAAAA", "Reference_Allele": "C"},
    )
    assert validate_reference_matches_context_snv(
        df
    ).index.tolist() == [0]
    # A 9-base context has its centre at index 4.
    assert validate_reference_matches_context_snv(
        df, context_length=9
    ).index.tolist() == [2]


def test_validate_full_applies_every_check():
    from sigmutsel.load_maf_files import validate_full

    df = _rows(
        {},
        {"Chromosome": "chrM2"},
        {"Tumor_Seq_Allele2": "TT"},
        {"CONTEXT": "AAAAACAAAAN"},
        {"Reference_Allele": "A", "Tumor_Seq_Allele2": "G"},
    )
    assert validate_full(df).index.tolist() == [0]
    # Indels skip the SNV checks: only the chromosome is checked.
    assert validate_full(df, variant_type="ID").index.tolist() == [
        0,
        2,
        3,
        4,
    ]


def test_create_mutation_type_column_snp_is_pyrimidine_centred():
    from sigmutsel.load_maf_files import (
        create_mutation_type_column,
        create_mutation_type_column_snp,
    )

    df = _rows(
        {"CONTEXT": "AAAAGCTAAAA", "Reference_Allele": "C"},
        # Purine reference: the same event on the other strand.
        {
            "CONTEXT": "TTTTAGCTTTT",
            "Reference_Allele": "G",
            "Tumor_Seq_Allele2": "A",
        },
        {
            "CONTEXT": "CCCCATGCCCC",
            "Reference_Allele": "T",
            "Tumor_Seq_Allele2": "G",
        },
    )
    out = create_mutation_type_column_snp(df)
    assert out["type"].tolist() == ["G[C>T]T", "G[C>T]T", "A[T>G]G"]
    assert "type" not in df.columns  # works on a copy
    for alias in ("SNP", "sbs", "RNA-SBS"):
        assert (
            create_mutation_type_column(df, variant_type=alias)[
                "type"
            ]
            .eq(out["type"])
            .all()
        )


def test_reduce_context_to_trinucleotides():
    from sigmutsel.load_maf_files import (
        reduce_context_to_trinucleotides,
    )

    df = _rows({"CONTEXT": "AAAAGCTAAAA"}, {"CONTEXT": "ACGTACGTACG"})
    out = reduce_context_to_trinucleotides(df)
    assert out["context"].tolist() == ["GCT", "ACG"]
    assert reduce_context_to_trinucleotides(
        df.assign(CONTEXT=["AGCTA", "CGTAC"]), context_length=5
    )["context"].tolist() == ["GCT", "GTA"]
    assert "context" not in df.columns


def test_filter_db_resolves_signature_class_aliases():
    from sigmutsel.load_maf_files import filter_db

    df = pd.DataFrame(
        {"Variant_Type": ["SNP", "DEL", "INS", "DNP", "SNP"]}
    )
    assert filter_db(df).index.tolist() == [0, 4]
    assert filter_db(df, "sbs").index.tolist() == [0, 4]
    assert filter_db(df, "ID").index.tolist() == [1, 2]
    assert filter_db(df, "indel").index.tolist() == [1, 2]
    assert filter_db(df, "DBS").empty  # DBS maps to DBP, not DNP
    with pytest.raises(KeyError, match="Variant_Type"):
        filter_db(df.rename(columns={"Variant_Type": "vt"}))


def _write_seqinfo(path, rows):
    path.write_text(
        "".join("\t".join(map(str, r)) + "\n" for r in rows)
    )


def test_create_mutation_type_column_id_matches_shifted_indels(
    tmp_path,
):
    from sigmutsel.load_maf_files import (
        build_id83_lookup,
        create_mutation_type_column,
    )

    # SigProfilerMatrixGenerator's seqinfo layout: sample, chromosome
    # number, position, strand-prefixed ID83 label, ref, alt, strand.
    _write_seqinfo(
        tmp_path / "1_seqinfo.txt",
        [
            ("S1", 1, 100, "U:1:Del:C:4", "AC", "A", 0),
            ("S1", 1, 100, "U:1:Del:C:4", "AC", "A", 0),  # duplicate
            (
                "S2",
                1,
                500,
                "T:1:Ins:T:0",
                "A",
                "AT",
                1,
            ),  # other sample
        ],
    )
    _write_seqinfo(
        tmp_path / "7_seqinfo.txt",
        # The anchor base before the indel: one left of the MAF.
        [("S1", 7, 299, "T:2:Del:R:0", "ATG", "A", 1)],
    )
    (tmp_path / "notes.txt").write_text("not a seqinfo file\n")

    lookup = build_id83_lookup(tmp_path)
    assert sorted(lookup["Chromosome"]) == ["chr1", "chr1", "chr7"]
    assert len(lookup) == 3  # the duplicate line counts once

    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S1"],
            "Chromosome": ["chr1", "chr7", "chr1"],
            "Start_Position": [100, 300, 900],
            "Hugo_Symbol": ["A", "B", "C"],
        },
        index=[10, 11, 12],
    )
    out = create_mutation_type_column(
        maf, variant_type="indel", seqinfo_dir=tmp_path
    )

    # Exact match, a match one base to the left, and no match.
    assert out["type"].tolist()[:2] == ["1:Del:C:4", "2:Del:R:0"]
    assert pd.isna(out["type"].iloc[2])
    assert out.columns.tolist() == [*maf.columns, "type"]
    assert out["Start_Position"].tolist() == [100, 300, 900]
