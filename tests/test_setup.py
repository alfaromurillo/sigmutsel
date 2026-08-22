"""Tests for setup.py's TCGA ABSOLUTE purity/segments downloaders.

Same local-reuse / shared-cache-hit / download-when-absent pattern
already covered for download_wes_target_bed in test_wes_target.py --
mirrored here for download_tcga_absolute_purity/_segments.
"""

from sigmutsel import setup as setup_mod


def _assert_no_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("should not touch the network")

    monkeypatch.setattr(
        setup_mod.urllib.request, "urlretrieve", _boom
    )


# --- download_tcga_absolute_purity ------------------------------------


def test_download_tcga_absolute_purity_local_reuse(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(setup_mod, "DATA_DIR", tmp_path)
    dest = tmp_path / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
    dest.write_text("existing")
    _assert_no_network(monkeypatch)

    result = setup_mod.download_tcga_absolute_purity()
    assert result == dest


def test_download_tcga_absolute_purity_shared_cache_hit(
    tmp_path, monkeypatch
):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    filename = "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
    (shared_dir / filename).write_text("shared contents")

    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv("TCGA_ABSOLUTE_DATA_HOME", str(shared_dir))
    _assert_no_network(monkeypatch)

    result = setup_mod.download_tcga_absolute_purity()
    assert result.read_text() == "shared contents"


def test_download_tcga_absolute_purity_downloads_when_absent(
    tmp_path, monkeypatch
):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv(
        "TCGA_ABSOLUTE_DATA_HOME",
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
    result = setup_mod.download_tcga_absolute_purity()
    assert len(calls) == 1
    assert (
        calls[0][0]
        == setup_mod.DOWNLOAD_URLS[
            "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
        ]
    )
    assert result.read_text() == "downloaded"


# --- download_tcga_absolute_segments -----------------------------------


def test_download_tcga_absolute_segments_local_reuse(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(setup_mod, "DATA_DIR", tmp_path)
    dest = tmp_path / "TCGA_mastercalls.abs_segtabs.fixed.txt"
    dest.write_text("existing")
    _assert_no_network(monkeypatch)

    result = setup_mod.download_tcga_absolute_segments()
    assert result == dest


def test_download_tcga_absolute_segments_shared_cache_hit(
    tmp_path, monkeypatch
):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    filename = "TCGA_mastercalls.abs_segtabs.fixed.txt"
    (shared_dir / filename).write_text("shared contents")

    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv("TCGA_ABSOLUTE_DATA_HOME", str(shared_dir))
    _assert_no_network(monkeypatch)

    result = setup_mod.download_tcga_absolute_segments()
    assert result.read_text() == "shared contents"


def test_download_tcga_absolute_segments_downloads_when_absent(
    tmp_path, monkeypatch
):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    monkeypatch.setattr(setup_mod, "DATA_DIR", local_dir)
    monkeypatch.setenv(
        "TCGA_ABSOLUTE_DATA_HOME",
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
    result = setup_mod.download_tcga_absolute_segments()
    assert len(calls) == 1
    assert (
        calls[0][0]
        == setup_mod.DOWNLOAD_URLS[
            "TCGA_mastercalls.abs_segtabs.fixed.txt"
        ]
    )
    assert result.read_text() == "downloaded"


def test_download_tcga_absolute_purity_and_segments_share_cache_dir():
    # Both functions must resolve to the same shared-cache directory
    # by default (companion files from the same GDC resource) --
    # checked structurally rather than by running both downloaders.
    import inspect

    purity_src = inspect.getsource(
        setup_mod.download_tcga_absolute_purity
    )
    segments_src = inspect.getsource(
        setup_mod.download_tcga_absolute_segments
    )
    assert "tcga_absolute" in purity_src
    assert "tcga_absolute" in segments_src
    assert "TCGA_ABSOLUTE_DATA_HOME" in purity_src
    assert "TCGA_ABSOLUTE_DATA_HOME" in segments_src
