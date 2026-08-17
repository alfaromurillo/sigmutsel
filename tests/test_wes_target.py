"""Tests for wes_target's BED x GENCODE interval-sweep and cache."""

import gzip

import numpy as np
import pandas as pd

from sigmutsel import wes_target


def _genes_df(rows):
    return pd.DataFrame(
        rows, columns=["chrom", "start", "end", "gene_id"]
    )


def _bed_df(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def test_merge_intervals_merges_overlapping():
    starts = np.array([10, 15, 30])
    ends = np.array([20, 25, 40])
    merged_starts, merged_ends = wes_target._merge_intervals(
        starts, ends
    )
    assert list(merged_starts) == [10, 30]
    assert list(merged_ends) == [25, 40]


def test_merge_intervals_merges_touching():
    # end of one interval exactly equals start of the next
    starts = np.array([10, 20])
    ends = np.array([20, 30])
    merged_starts, merged_ends = wes_target._merge_intervals(
        starts, ends
    )
    assert list(merged_starts) == [10]
    assert list(merged_ends) == [30]


def test_overlapping_gene_ids_basic_overlap():
    bed = _bed_df([("chr1", 100, 200)])
    genes = _genes_df([("chr1", 150, 160, "ENSG_IN")])
    assert wes_target._overlapping_gene_ids(bed, genes) == {"ENSG_IN"}


def test_overlapping_gene_ids_no_overlap_excluded():
    bed = _bed_df([("chr1", 100, 200)])
    genes = _genes_df([("chr1", 300, 400, "ENSG_OUT")])
    assert wes_target._overlapping_gene_ids(bed, genes) == set()


def test_overlapping_gene_ids_spans_merged_adjacent_intervals():
    # Two touching BED intervals merge into one; a gene spanning the
    # merge point must still be detected.
    bed = _bed_df([("chr1", 100, 150), ("chr1", 150, 200)])
    genes = _genes_df([("chr1", 140, 160, "ENSG_SPAN")])
    assert wes_target._overlapping_gene_ids(bed, genes) == {
        "ENSG_SPAN"
    }


def test_overlapping_gene_ids_exact_boundary_touch():
    # BED interval ends exactly where the gene starts.
    bed = _bed_df([("chr1", 100, 200)])
    genes = _genes_df([("chr1", 200, 300, "ENSG_TOUCH")])
    assert wes_target._overlapping_gene_ids(bed, genes) == {
        "ENSG_TOUCH"
    }


def test_overlapping_gene_ids_isolated_by_chromosome():
    # Numerically overlapping coordinates on a different chromosome
    # must never match.
    bed = _bed_df([("chr1", 100, 200)])
    genes = _genes_df([("chr2", 150, 160, "ENSG_OTHER_CHROM")])
    assert wes_target._overlapping_gene_ids(bed, genes) == set()


def test_parse_bed_chr_prefixing(tmp_path):
    bed_path = tmp_path / "test.bed"
    bed_path.write_text("1\t100\t200\nX\t300\t400\n")
    df = wes_target._parse_bed(bed_path)
    assert list(df["chrom"]) == ["chr1", "chrX"]


def test_parse_gtf_genes_strips_version_and_filters_feature(tmp_path):
    gtf_path = tmp_path / "test.gtf.gz"
    lines = [
        "#comment line\n",
        (
            "chr1\tHAVANA\tgene\t100\t200\t.\t+\t.\t"
            'gene_id "ENSG00000141510.11"; gene_name "TP53";\n'
        ),
        (
            "chr1\tHAVANA\ttranscript\t100\t200\t.\t+\t.\t"
            'gene_id "ENSG00000141510.11";\n'
        ),
    ]
    with gzip.open(gtf_path, "wt") as f:
        f.writelines(lines)
    df = wes_target._parse_gtf_genes(gtf_path)
    # only the "gene" row survives, and the version suffix is gone
    assert len(df) == 1
    assert df.iloc[0]["gene_id"] == "ENSG00000141510"
    assert df.iloc[0]["start"] == 100
    assert df.iloc[0]["end"] == 200


def test_get_wes_target_gene_ids_reads_existing_cache(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "wes_target_gene_ids_gencode19.txt"
    cache_path.write_text("ENSG00000000001\nENSG00000000002")
    monkeypatch.setattr(
        wes_target, "location_wes_target_gene_ids", cache_path
    )

    def _boom(*args, **kwargs):
        raise AssertionError(
            "should not attempt to download/recompute when cache exists"
        )

    monkeypatch.setattr(
        wes_target.setup, "download_wes_target_bed", _boom
    )
    monkeypatch.setattr(
        wes_target.setup, "download_gencode_gtf", _boom
    )
    monkeypatch.setattr(
        wes_target, "compute_wes_target_gene_ids", _boom
    )

    ids = wes_target.get_wes_target_gene_ids()
    assert ids == {"ENSG00000000001", "ENSG00000000002"}


def test_get_wes_target_gene_ids_force_recompute_ignores_cache(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "wes_target_gene_ids_gencode19.txt"
    cache_path.write_text("ENSG_STALE")
    monkeypatch.setattr(
        wes_target, "location_wes_target_gene_ids", cache_path
    )
    monkeypatch.setattr(
        wes_target.setup, "download_wes_target_bed", lambda **kw: None
    )
    monkeypatch.setattr(
        wes_target.setup, "download_gencode_gtf", lambda **kw: None
    )
    monkeypatch.setattr(
        wes_target,
        "compute_wes_target_gene_ids",
        lambda: {"ENSG_FRESH"},
    )

    ids = wes_target.get_wes_target_gene_ids(force_recompute=True)
    assert ids == {"ENSG_FRESH"}
    assert cache_path.read_text().strip() == "ENSG_FRESH"


def test_download_wes_target_bed_local_reuse(tmp_path, monkeypatch):
    from sigmutsel import setup as setup_mod

    monkeypatch.setattr(setup_mod, "DATA_DIR", tmp_path)
    dest = tmp_path / "gaf_20111020Plusbroad_wex_1.1_hg19.bed"
    dest.write_text("existing")

    def _boom(*args, **kwargs):
        raise AssertionError("should not touch the network")

    monkeypatch.setattr(
        setup_mod.urllib.request, "urlretrieve", _boom
    )
    result = setup_mod.download_wes_target_bed()
    assert result == dest


def test_download_wes_target_bed_shared_cache_hit(
    tmp_path, monkeypatch
):
    from sigmutsel import setup as setup_mod

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    filename = "gaf_20111020Plusbroad_wex_1.1_hg19.bed"
    (shared_dir / filename).write_text("shared contents")

    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv("MC3_WES_TARGET_DATA_HOME", str(shared_dir))

    def _boom(*args, **kwargs):
        raise AssertionError("should not touch the network")

    monkeypatch.setattr(
        setup_mod.urllib.request, "urlretrieve", _boom
    )
    result = setup_mod.download_wes_target_bed()
    assert result.read_text() == "shared contents"


def test_download_wes_target_bed_downloads_when_absent(
    tmp_path, monkeypatch
):
    from sigmutsel import setup as setup_mod

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv(
        "MC3_WES_TARGET_DATA_HOME",
        str(tmp_path / "nonexistent_shared"),
    )

    calls = []

    def _fake_download_file(url, dest, decompress=False):
        calls.append((url, dest, decompress))
        dest.write_text("downloaded")
        return dest

    monkeypatch.setattr(
        setup_mod, "download_file", _fake_download_file
    )
    result = setup_mod.download_wes_target_bed()
    assert len(calls) == 1
    assert (
        calls[0][0]
        == setup_mod.DOWNLOAD_URLS[
            "gaf_20111020Plusbroad_wex_1.1_hg19.bed"
        ]
    )
    assert result.read_text() == "downloaded"
