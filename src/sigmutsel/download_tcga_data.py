"""Utilities for downloading and unpacking TCGA mutation MAF files."""

import argparse
import gzip
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


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
    with (
        gzip.open(compressed_maf, "rb") as f_in,
        output_path.open("wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)


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

        if maf_condition and annotations_condition and logs_condition:
            logger.info("Processing %s", subdir.name)
            _extract_maf(maf_files[0], destination_dir)
        else:
            logger.warning(
                "Skipping %s (maf.gz: %d, annotations.txt: %s, logs dir: %s)",
                subdir.name,
                len(maf_files),
                annotations_condition,
                logs_condition,
            )

        if not maf_condition:
            all_single_maf = False

    return all_single_maf


def download_tcga_maf_files(
    gdc_manifest: Path | str, destination: Path | str
) -> bool:
    """Download TCGA COAD mutation MAFs via ``gdc-client`` and unpack them.

    Parameters
    ----------
    gdc_manifest : Path or str
        Path to the manifest file used by ``gdc-client``.
    destination : Path or str
        Directory where all decompressed ``*.maf`` files will be saved.

    Returns
    -------
    bool
        ``True`` if every downloaded bundle contained exactly one MAF,
        ``False`` otherwise.
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
