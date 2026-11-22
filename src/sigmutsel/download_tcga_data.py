"""Utilities for downloading and unpacking TCGA mutation MAF files."""

import argparse
import gzip
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def _check_gdc_client() -> str:
    """Return the path to ``gdc-client`` or raise if it is missing."""
    executable = shutil.which("gdc-client")
    if executable is None:
        raise RuntimeError(
            "Requires gdc-client. Download it from: "
            "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool "
            "and ensure it is on your PATH."
        )
    return executable


def _run_gdc_download(manifest: Path, output_dir: Path) -> None:
    """Invoke ``gdc-client download``."""
    cmd = [
        _check_gdc_client(),
        "download",
        "-m",
        str(manifest),
        "-d",
        str(output_dir),
    ]
    logger.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _extract_maf(compressed_maf: Path, destination_dir: Path) -> None:
    """Uncompress a single ``*.maf.gz`` file into ``destination_dir``."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        destination_dir / compressed_maf.with_suffix("").name
    )
    # Written beside the target and renamed, so an interrupted run
    # never leaves a truncated MAF under the final name.
    partial = output_path.with_name(output_path.name + ".part")
    with (
        gzip.open(compressed_maf, "rb") as f_in,
        partial.open("wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    partial.replace(output_path)


def process_tcga_maf_downloads(
    temp_dir: Path, destination_dir: Path
) -> bool:
    """Move MAFs downloaded by gdc-client into ``destination_dir``.

    Parameters
    ----------
    temp_dir : Path
        Directory containing per-sample subdirectories created by
        ``gdc-client download``.
    destination_dir : Path
        Output directory where decompressed ``*.maf`` files should be
        stored.

    Returns
    -------
    bool
        True if every subdirectory contained exactly one ``*.maf.gz``,
        False otherwise (processing continues for all subdirs).
    """
    if not temp_dir.exists():
        raise FileNotFoundError(
            f"Temporary TCGA directory not found at {temp_dir}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)

    subdirs: Iterable[Path] = sorted(
        d for d in temp_dir.iterdir() if d.is_dir()
    )
    if not subdirs:
        logger.warning("No subdirectories found under %s", temp_dir)
        return False

    all_single_maf = True

    for subdir in subdirs:
        maf_files = sorted(subdir.glob("*.maf.gz"))
        annotations_file = subdir / "annotations.txt"
        logs_dir = subdir / "logs"

        maf_condition = len(maf_files) == 1
        annotations_condition = annotations_file.is_file()
        logs_condition = logs_dir.is_dir()

        # Only the MAF itself is data. gdc-client fetches
        # annotations.txt from a separate API call that times out under
        # load, and a bundle without it is still a complete MAF:
        # requiring it silently dropped valid tumors.
        if maf_condition:
            logger.info("Processing %s", subdir.name)
            _extract_maf(maf_files[0], destination_dir)
            if not (annotations_condition and logs_condition):
                logger.info(
                    "%s: extracted without annotations.txt (%s) or "
                    "logs dir (%s)",
                    subdir.name,
                    annotations_condition,
                    logs_condition,
                )
        else:
            logger.warning(
                "Skipping %s (maf.gz: %d)",
                subdir.name,
                len(maf_files),
            )

        if not maf_condition:
            all_single_maf = False

    return all_single_maf


def download_project_maf_files(
    project: str,
    destination: Path | str,
    *,
    download_dir: Path | str | None = None,
    workers: int = 6,
) -> list[Path]:
    """Download a TCGA project's open MAFs from the GDC, no ``gdc-client``.

    Searches the GDC for the project's per-aliquot somatic-mutation
    MAFs (gdcfetch's ``somatic-mutations`` preset: open-access
    "Masked Somatic Mutation" files of the "Aliquot Ensemble Somatic
    Variant Merging and Masking" workflow), downloads them with
    gdcfetch over HTTPS, and decompresses each into ``destination``,
    the flat directory of ``*.maf`` files that
    :class:`~sigmutsel.MutationDataset` reads.

    The compressed downloads are kept in ``download_dir``, so a rerun
    after an interrupted download fetches only what is missing (and
    checks every file's size and MD5). The GDC holds one MAF per
    sequenced aliquot, including recurrent and metastatic samples and
    several aliquots of one patient: select the files with
    :func:`sigmutsel.tcga_sample_selection.select_tcga_maf_files`
    before building a dataset.

    Parameters
    ----------
    project : str
        TCGA study code, with or without the ``TCGA-`` prefix
        (``"HNSC"`` or ``"TCGA-HNSC"``).
    destination : Path or str
        Directory for the decompressed ``*.maf`` files; created if
        missing.
    download_dir : Path or str, optional
        Where the compressed downloads go, laid out as
        ``<download_dir>/<file_id>/<file_name>``. Default: a sibling
        of ``destination`` named ``<destination>_gdc``.
    workers : int, default 6
        Parallel downloads.

    Returns
    -------
    list of Path
        The decompressed MAF files in ``destination``, sorted.

    Raises
    ------
    ValueError
        If the GDC lists no MAF for ``project``.
    OSError
        If some downloads still fail after gdcfetch's retries; rerun
        to fetch only those.

    Examples
    --------
    >>> from sigmutsel.download_tcga_data import (
    ...     download_project_maf_files)
    >>> mafs = download_project_maf_files(  # doctest: +SKIP
    ...     "HNSC", "hnsc/all_maf_files_raw")
    >>> len(mafs)  # doctest: +SKIP
    511
    """
    from gdcfetch import download_files, get_preset, search_files

    project_id = (
        project if project.startswith("TCGA-") else f"TCGA-{project}"
    )
    destination_dir = Path(destination).resolve()
    if download_dir is None:
        download_dir = destination_dir.with_name(
            f"{destination_dir.name}_gdc"
        )
    download_dir = Path(download_dir).resolve()

    hits = search_files(
        project_id, **get_preset("somatic-mutations").search_kwargs()
    )
    if not hits:
        raise ValueError(
            f"The GDC lists no open MAF for {project_id}"
        )
    logger.info(
        "%s: %d MAFs listed by the GDC", project_id, len(hits)
    )
    download_files(hits, download_dir, workers=workers)

    destination_dir.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        compressed = download_dir / hit["file_id"] / hit["file_name"]
        target = destination_dir / compressed.with_suffix("").name
        if not target.exists():
            _extract_maf(compressed, destination_dir)
    return sorted(destination_dir.glob("*.maf"))


def download_tcga_maf_files(
    gdc_manifest: Path | str, destination: Path | str
) -> bool:
    """Download the MAFs of a GDC manifest via ``gdc-client``, unpacked.

    Needs NCI's ``gdc-client`` executable on ``PATH``. To download a
    whole TCGA project without it, use
    :func:`download_project_maf_files`.

    Parameters
    ----------
    gdc_manifest : Path or str
        Path to the manifest file used by ``gdc-client`` (from the GDC
        Data Portal's cart, or ``gdcfetch manifest``).
    destination : Path or str
        Directory where all decompressed ``*.maf`` files will be saved.

    Returns
    -------
    bool
        ``True`` if every downloaded bundle contained exactly one MAF,
        ``False`` otherwise.

    Examples
    --------
    >>> from sigmutsel.download_tcga_data import download_tcga_maf_files
    >>> download_tcga_maf_files(  # doctest: +SKIP
    ...     "gdc_manifest.txt", "maf_files")
    True
    """
    manifest_path = Path(gdc_manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    destination_dir = Path(destination).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="tcga_maf_", dir=destination_dir.parent
        )
    )
    logger.info("Created temporary directory %s", temp_root)

    try:
        _run_gdc_download(manifest_path, temp_root)
        success = process_tcga_maf_downloads(
            temp_root, destination_dir
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        logger.info("Removed temporary directory %s", temp_root)

    return success


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for download helper."""
    parser = argparse.ArgumentParser(
        description="Download TCGA MAF files via gdc-client and unpack them."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the gdc-client manifest file.",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Directory where decompressed MAF files will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for downloading TCGA mutation data."""
    args = parse_args()
    success = download_tcga_maf_files(
        gdc_manifest=args.manifest, destination=args.destination
    )
    if success:
        logger.info(
            "All download directories contained a single MAF."
        )
    else:
        logger.warning(
            "Some download directories lacked a single MAF file."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )
    main()
