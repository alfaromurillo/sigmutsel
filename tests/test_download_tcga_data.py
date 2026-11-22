"""Unpacking gdc-client MAF bundles."""

import gzip

from sigmutsel.download_tcga_data import (
    _extract_maf,
    process_tcga_maf_downloads,
)

_MAF_BYTES = b"#version 2.4\nHugo_Symbol\tChromosome\nTP53\tchr17\n"


def _bundle(root, name, *, annotations=True, logs=True, n_maf=1):
    d = root / name
    d.mkdir()
    for i in range(n_maf):
        with gzip.open(d / f"{name}_{i}.maf.gz", "wb") as fh:
            fh.write(b"#version 2.4\nHugo_Symbol\nTP53\n")
    if annotations:
        (d / "annotations.txt").write_text("id\n")
    if logs:
        (d / "logs").mkdir()


def test_bundle_without_annotations_is_still_extracted(tmp_path):
    tmp, out = tmp_path / "tmp", tmp_path / "out"
    tmp.mkdir()
    _bundle(tmp, "full")
    _bundle(tmp, "no_annotations", annotations=False, logs=False)
    assert process_tcga_maf_downloads(tmp, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "full_0.maf",
        "no_annotations_0.maf",
    ]
    assert (
        (out / "no_annotations_0.maf").read_text().endswith("TP53\n")
    )


def test_bundle_without_a_single_maf_is_skipped(tmp_path):
    tmp, out = tmp_path / "tmp", tmp_path / "out"
    tmp.mkdir()
    _bundle(tmp, "empty", n_maf=0)
    _bundle(tmp, "two", n_maf=2)
    assert not process_tcga_maf_downloads(tmp, out)
    assert not out.exists() or not any(out.iterdir())


def test_extract_maf_writes_uncompressed_bytes(tmp_path):
    archive = tmp_path / "aliquot.wxs.maf.gz"
    with gzip.open(archive, "wb") as fh:
        fh.write(_MAF_BYTES)
    dest = tmp_path / "nested" / "out"  # created on demand

    _extract_maf(archive, dest)

    assert [p.name for p in dest.iterdir()] == ["aliquot.wxs.maf"]
    assert (dest / "aliquot.wxs.maf").read_bytes() == _MAF_BYTES
    assert archive.exists()  # the archive is left in place


def test_extract_maf_overwrites_existing_output(tmp_path):
    archive = tmp_path / "a.maf.gz"
    with gzip.open(archive, "wb") as fh:
        fh.write(_MAF_BYTES)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "a.maf").write_bytes(
        b"stale content that is longer\n" * 9
    )

    _extract_maf(archive, dest)

    assert (dest / "a.maf").read_bytes() == _MAF_BYTES


def _fake_gdc(monkeypatch, files):
    """gdcfetch's search and download, offline: ``files`` maps a GDC
    file id to its MAF bytes, laid out as gdcfetch lays them out."""
    import gdcfetch

    hits = [
        {
            "file_id": fid,
            "file_name": f"{fid}.wxs.aliquot_ensemble_masked.maf.gz",
        }
        for fid in files
    ]
    calls = {"search": [], "download": 0}

    def search_files(project_id, **kwargs):
        calls["search"].append((project_id, kwargs))
        return hits

    def download_files(found, dest, workers=6):
        calls["download"] += 1
        for hit in found:
            d = dest / hit["file_id"]
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / hit["file_name"], "wb") as fh:
                fh.write(files[hit["file_id"]])
        return [dest / h["file_id"] / h["file_name"] for h in found]

    monkeypatch.setattr(gdcfetch, "search_files", search_files)
    monkeypatch.setattr(gdcfetch, "download_files", download_files)
    return calls


def test_download_project_maf_files_flattens_and_decompresses(
    tmp_path, monkeypatch
):
    from sigmutsel.download_tcga_data import (
        download_project_maf_files,
    )

    calls = _fake_gdc(
        monkeypatch, {"a1": _MAF_BYTES, "b2": _MAF_BYTES}
    )
    out = tmp_path / "mafs"

    mafs = download_project_maf_files("HNSC", out)

    assert [p.name for p in mafs] == [
        "a1.wxs.aliquot_ensemble_masked.maf",
        "b2.wxs.aliquot_ensemble_masked.maf",
    ]
    assert all(p.parent == out.resolve() for p in mafs)
    assert mafs[0].read_bytes() == _MAF_BYTES
    # the somatic-mutations preset, for the prefixed project id
    project_id, kwargs = calls["search"][0]
    assert project_id == "TCGA-HNSC"
    assert kwargs["data_type"] == "Masked Somatic Mutation"
    assert kwargs["access"] == "open"
    # compressed downloads kept beside the destination, for reruns
    assert (tmp_path / "mafs_gdc" / "a1").is_dir()
    assert not list(out.glob("*.part"))


def test_download_project_maf_files_keeps_finished_mafs(
    tmp_path, monkeypatch
):
    from sigmutsel.download_tcga_data import (
        download_project_maf_files,
    )

    _fake_gdc(monkeypatch, {"a1": _MAF_BYTES})
    out = tmp_path / "mafs"
    (maf,) = download_project_maf_files("TCGA-HNSC", out)
    maf.write_bytes(b"edited")  # a finished file is not rewritten

    (again,) = download_project_maf_files("TCGA-HNSC", out)

    assert again.read_bytes() == b"edited"


def test_download_project_maf_files_refuses_an_empty_listing(
    tmp_path, monkeypatch
):
    import pytest

    from sigmutsel.download_tcga_data import (
        download_project_maf_files,
    )

    _fake_gdc(monkeypatch, {})
    with pytest.raises(ValueError, match="no open MAF"):
        download_project_maf_files("NOPE", tmp_path / "mafs")
