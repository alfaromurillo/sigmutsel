"""Tests for split_maf_file.split_maf_file."""

import pytest

from sigmutsel.split_maf_file import split_maf_file

# Minimal MAF content: two comment lines, a header, and three
# mutations across three samples (one each).
_COMMENT_LINES = "# version 2.4\n# source: synthetic\n"
_HEADER = "\t".join(
    [
        "Hugo_Symbol",
        "Tumor_Sample_Barcode",
        "Variant_Type",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]
)
_ROWS = [
    "\t".join(["KRAS", "SAMPLE_A", "SNP", "C", "T"]),
    "\t".join(["TP53", "SAMPLE_B", "SNP", "G", "A"]),
    "\t".join(["EGFR", "SAMPLE_C", "SNP", "A", "G"]),
]
_MAF_CONTENT = (
    _COMMENT_LINES + _HEADER + "\n" + "\n".join(_ROWS) + "\n"
)


@pytest.fixture()
def maf_file(tmp_path):
    p = tmp_path / "multi_sample.maf"
    p.write_text(_MAF_CONTENT)
    return p


def test_creates_one_file_per_sample(maf_file, tmp_path):
    out_dir = tmp_path / "split"
    split_maf_file(maf_file, out_dir)

    produced = sorted(f.name for f in out_dir.glob("*.maf"))
    assert produced == [
        "SAMPLE_A.maf",
        "SAMPLE_B.maf",
        "SAMPLE_C.maf",
    ]


def test_each_file_contains_only_its_sample(maf_file, tmp_path):
    out_dir = tmp_path / "split"
    split_maf_file(maf_file, out_dir)

    for sample in ("SAMPLE_A", "SAMPLE_B", "SAMPLE_C"):
        text = (out_dir / f"{sample}.maf").read_text()
        lines = [
            ln for ln in text.splitlines() if not ln.startswith("#")
        ]
        # header + one data row
        assert len(lines) == 2
        assert all(
            sample in ln or ln.startswith("Hugo_Symbol")
            for ln in lines
        )


def test_comment_lines_preserved_in_every_file(maf_file, tmp_path):
    out_dir = tmp_path / "split"
    split_maf_file(maf_file, out_dir)

    for sample in ("SAMPLE_A", "SAMPLE_B", "SAMPLE_C"):
        text = (out_dir / f"{sample}.maf").read_text()
        assert "# version 2.4" in text
        assert "# source: synthetic" in text


def test_skip_if_files_already_present(maf_file, tmp_path):
    out_dir = tmp_path / "split"
    split_maf_file(maf_file, out_dir)

    mtimes_before = {
        f.name: f.stat().st_mtime for f in out_dir.glob("*.maf")
    }

    split_maf_file(maf_file, out_dir, force_generation=False)

    mtimes_after = {
        f.name: f.stat().st_mtime for f in out_dir.glob("*.maf")
    }
    assert mtimes_before == mtimes_after


def test_force_generation_overwrites(maf_file, tmp_path):
    out_dir = tmp_path / "split"
    split_maf_file(maf_file, out_dir)

    mtimes_before = {
        f.name: f.stat().st_mtime for f in out_dir.glob("*.maf")
    }

    split_maf_file(maf_file, out_dir, force_generation=True)

    mtimes_after = {
        f.name: f.stat().st_mtime for f in out_dir.glob("*.maf")
    }
    assert mtimes_before != mtimes_after


def test_missing_barcode_column_raises(tmp_path):
    bad_maf = tmp_path / "bad.maf"
    bad_maf.write_text("Hugo_Symbol\tVariant_Type\nKRAS\tSNP\n")
    with pytest.raises(KeyError, match="Tumor_Sample_Barcode"):
        split_maf_file(bad_maf, tmp_path / "out")
