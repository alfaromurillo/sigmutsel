"""The distributed RepeatMasker table and the downloader's resolution.

The distributed table must give exactly the intervals of UCSC's full
rmsk dump, so every repeat flag and territory mask is unchanged; and
the downloader must never reach the network in these tests.
"""

import pandas as pd
import pytest

from sigmutsel import repeatmasker as rm
from sigmutsel import setup as setup_mod
from sigmutsel.qc import load_repeat_intervals


def _rmsk_row(chrom, start, end, n):
    return "\t".join(
        ["585", "463", "13", "6", "17", chrom, str(start), str(end)]
        + ["-1", "+", "A", "LINE", "L1", "1", "2", "0", str(n)]
    )


# Overlapping, adjacent, nested, unsorted rows, and one alt contig.
RMSK_ROWS = [
    _rmsk_row("chr1", 1005, 1020, 1),
    _rmsk_row("chr1", 1000, 1010, 2),
    _rmsk_row(
        "chr1", 1020, 1030, 3
    ),  # touches the merged [1000, 1020)
    _rmsk_row("chr1", 2000, 2100, 4),
    _rmsk_row("chr1", 2010, 2020, 5),  # nested
    _rmsk_row("chr2", 5000, 5010, 6),
    _rmsk_row("chr1_KI270706v1_random", 10, 20, 7),
]


@pytest.fixture
def full_table(tmp_path):
    path = tmp_path / "rmsk.hg38.txt.gz"
    pd.DataFrame([r.split("\t") for r in RMSK_ROWS]).to_csv(
        path, sep="\t", header=False, index=False, compression="gzip"
    )
    return path


def _assert_same_intervals(a, b):
    assert sorted(a) == sorted(b)
    for chrom in a:
        assert a[chrom][0].tolist() == b[chrom][0].tolist()
        assert a[chrom][1].tolist() == b[chrom][1].tolist()


def test_distributed_table_gives_the_full_tables_intervals(
    full_table, tmp_path
):
    out = rm.write_distributed_table(
        full_table, tmp_path / "d.bed.xz"
    )
    full = load_repeat_intervals(full_table)
    _assert_same_intervals(load_repeat_intervals(out), full)
    assert full["chr1"][0].tolist() == [1000, 2000]
    assert full["chr1"][1].tolist() == [1030, 2100]


def test_distributed_table_is_deterministic(full_table, tmp_path):
    a = rm.write_distributed_table(full_table, tmp_path / "a.bed.xz")
    b = rm.write_distributed_table(full_table, tmp_path / "b.bed.xz")
    assert a.read_bytes() == b.read_bytes()


def test_load_repeat_intervals_merges_an_unmerged_bed(tmp_path):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t50\t60\nchr1\t10\t20\nchr1\t15\t30\n")
    out = load_repeat_intervals(bed)
    assert out["chr1"][0].tolist() == [10, 50]
    assert out["chr1"][1].tolist() == [30, 60]


def _fake_urlretrieve(source):
    def fake(url, dest):
        dest.write_bytes(source.read_bytes())

    return fake


def test_download_checks_the_checksum(
    full_table, tmp_path, monkeypatch
):
    source = rm.write_distributed_table(
        full_table, tmp_path / "s.bed.xz"
    )
    monkeypatch.setattr(
        rm.urllib.request, "urlretrieve", _fake_urlretrieve(source)
    )
    monkeypatch.setattr(rm, "DISTRIBUTED_SHA256", rm._sha256(source))
    dest = tmp_path / "cache" / rm.DISTRIBUTED_NAME
    assert rm.download_distributed_table(dest) == dest
    assert dest.read_bytes() == source.read_bytes()


def test_download_rejects_a_wrong_checksum(
    full_table, tmp_path, monkeypatch
):
    source = rm.write_distributed_table(
        full_table, tmp_path / "s.bed.xz"
    )
    monkeypatch.setattr(
        rm.urllib.request, "urlretrieve", _fake_urlretrieve(source)
    )
    monkeypatch.setattr(rm, "DISTRIBUTED_SHA256", "0" * 64)
    dest = tmp_path / rm.DISTRIBUTED_NAME
    with pytest.raises(ValueError, match="Checksum mismatch"):
        rm.download_distributed_table(dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("should not touch the network")

    monkeypatch.setattr(setup_mod.urllib.request, "urlretrieve", boom)
    monkeypatch.setattr(rm.urllib.request, "urlretrieve", boom)
    monkeypatch.setattr(setup_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv(
        "REPEATMASKER_DATA_HOME", str(tmp_path / "shared")
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "shared").mkdir()
    return tmp_path


def test_hg38_uses_the_full_table_when_present(no_network):
    full = no_network / "data" / "rmsk.hg38.txt.gz"
    full.write_text("x")
    assert setup_mod.download_repeatmasker_bed("hg38") == full


def test_hg38_reuses_a_shared_full_table(no_network):
    (no_network / "shared" / "rmsk.hg38.txt.gz").write_text("shared")
    out = setup_mod.download_repeatmasker_bed("hg38")
    assert out == no_network / "data" / "rmsk.hg38.txt.gz"
    assert out.read_text() == "shared"


def test_hg38_defaults_to_the_distributed_table(
    no_network, monkeypatch
):
    calls = []

    def fake(dest, force=False):
        calls.append((dest, force))
        return dest

    monkeypatch.setattr(rm, "download_distributed_table", fake)
    out = setup_mod.download_repeatmasker_bed("hg38")
    assert out == no_network / "data" / rm.DISTRIBUTED_NAME
    assert calls == [(out, False)]


def test_full_true_downloads_the_full_table(no_network, monkeypatch):
    got = []

    def fake_download_file(url, dest, decompress=False):
        got.append((url, dest.name))
        return dest

    monkeypatch.setattr(
        setup_mod, "download_file", fake_download_file
    )
    setup_mod.download_repeatmasker_bed("hg38", full=True)
    setup_mod.download_repeatmasker_bed("hg19")
    assert got == [
        (
            setup_mod.DOWNLOAD_URLS["rmsk.hg38.txt.gz"],
            "rmsk.hg38.txt.gz",
        ),
        (
            setup_mod.DOWNLOAD_URLS["rmsk.hg19.txt.gz"],
            "rmsk.hg19.txt.gz",
        ),
    ]
