"""Unpacking gdc-client MAF bundles."""

import gzip

from sigmutsel.download_tcga_data import process_tcga_maf_downloads


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
