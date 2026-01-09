"""Download reference data files for sigmutsel.

This module handles downloading large reference files that are not
included in the package distribution. Users can run this as:

    python -m sigmutsel setup

Or import and call programmatically:

    from sigmutsel.setup import download_all
    download_all()
"""

import urllib.request
import gzip
import shutil
import logging
from pathlib import Path

from sigmutsel.locations import DATA_DIR


logger = logging.getLogger(__name__)

# Download URLs for reference files
DOWNLOAD_URLS = {
    "hgnc_complete_set.txt": (
        "https://storage.googleapis.com/public-download-files/"
        "hgnc/tsv/tsv/hgnc_complete_set.txt"
    ),
    "Homo_sapiens.GRCh38.cds.all.fa.gz": (
        "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/"
        "cds/Homo_sapiens.GRCh38.cds.all.fa.gz"
    ),
    "gencode.v38.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/"
        "Gencode_human/release_38/gencode.v38.annotation.gtf.gz"
    ),
    "gencode.v19.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/"
        "Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
    ),
}


def download_file(url: str, dest: Path, decompress: bool = False):
    """Download a file from URL to destination.

    Parameters
    ----------
    url : str
        URL to download from
    dest : Path
        Destination file path
    decompress : bool, default False
        If True and file ends with .gz, decompress after downloading

    Returns
    -------
    Path
        Path to downloaded (and possibly decompressed) file
    """
    logger.info(f"Downloading {dest.name}...")
    logger.info(f"  from {url}")

    # Download to temporary file first
    temp_file = dest.with_suffix(dest.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, temp_file)

        # Handle gzipped files
        if decompress and str(dest).endswith(".gz"):
            logger.info("  Decompressing...")
            final_dest = dest.with_suffix("")  # Remove .gz
            with gzip.open(temp_file, "rb") as f_in:
                with open(final_dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            temp_file.unlink()  # Remove compressed temp file
            logger.info(f"  Saved to {final_dest}")
            return final_dest
        else:
            # Move temp file to final destination
            temp_file.rename(dest)
            logger.info(f"  Saved to {dest}")
            return dest

    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(
            f"Failed to download {dest.name}: {e}"
        ) from e


def download_hgnc(force: bool = False) -> Path:
    """Download HGNC complete set.

    Parameters
    ----------
    force : bool, default False
        If True, download even if file exists

    Returns
    -------
    Path
        Path to downloaded file
    """
    dest = DATA_DIR / "hgnc_complete_set.txt"
    if dest.exists() and not force:
        logger.info(f"HGNC file already exists at {dest}")
        return dest

    url = DOWNLOAD_URLS["hgnc_complete_set.txt"]
    return download_file(url, dest)


def download_cds_fasta(
    force: bool = False, decompress: bool = True
) -> Path:
    """Download Ensembl CDS FASTA file.

    Parameters
    ----------
    force : bool, default False
        If True, download even if file exists
    decompress : bool, default True
        If True, decompress the .gz file after downloading

    Returns
    -------
    Path
        Path to downloaded file
    """
    if decompress:
        dest = DATA_DIR / "Homo_sapiens.GRCh38.cds.all.fa"
        if dest.exists() and not force:
            logger.info(f"CDS FASTA already exists at {dest}")
            return dest
    else:
        dest = DATA_DIR / "Homo_sapiens.GRCh38.cds.all.fa.gz"
        if dest.exists() and not force:
            logger.info(
                f"CDS FASTA (compressed) already exists at {dest}"
            )
            return dest

    url = DOWNLOAD_URLS["Homo_sapiens.GRCh38.cds.all.fa.gz"]
    gz_dest = DATA_DIR / "Homo_sapiens.GRCh38.cds.all.fa.gz"
    return download_file(url, gz_dest, decompress=decompress)


def download_gencode_gtf(
    version: str = "38",
    force: bool = False,
    keep_compressed: bool = True,
) -> Path:
    """Download GENCODE GTF annotation file.

    Parameters
    ----------
    version : str, default "38"
        GENCODE version ("38" or "19")
    force : bool, default False
        If True, download even if file exists
    keep_compressed : bool, default True
        If True, keep file compressed (.gtf.gz)
        If False, decompress to .gtf

    Returns
    -------
    Path
        Path to downloaded file
    """
    filename = f"gencode.v{version}.annotation.gtf.gz"
    dest = DATA_DIR / filename

    if keep_compressed:
        final_dest = dest
    else:
        final_dest = dest.with_suffix("")  # Remove .gz

    if final_dest.exists() and not force:
        logger.info(
            f"GENCODE v{version} already exists at {final_dest}"
        )
        return final_dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=not keep_compressed)


def download_all(
    force: bool = False,
    decompress_fasta: bool = True,
    keep_gtf_compressed: bool = True,
):
    """Download all required reference files.

    Parameters
    ----------
    force : bool, default False
        If True, re-download even if files exist
    decompress_fasta : bool, default True
        If True, decompress FASTA file
    keep_gtf_compressed : bool, default True
        If True, keep GTF files compressed

    Returns
    -------
    dict[str, Path]
        Mapping of file type to downloaded path
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading sigmutsel reference data files")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("=" * 60)

    downloaded = {}

    try:
        # Download HGNC
        logger.info("\n1. HGNC Complete Set")
        downloaded["hgnc"] = download_hgnc(force=force)

        # Download CDS FASTA
        logger.info("\n2. Ensembl CDS FASTA (GRCh38)")
        downloaded["cds_fasta"] = download_cds_fasta(
            force=force, decompress=decompress_fasta
        )

        # Download GENCODE GTF v38
        logger.info("\n3. GENCODE v38 Annotation")
        downloaded["gencode38"] = download_gencode_gtf(
            version="38",
            force=force,
            keep_compressed=keep_gtf_compressed,
        )

        # Download GENCODE GTF v19
        logger.info("\n4. GENCODE v19 Annotation")
        downloaded["gencode19"] = download_gencode_gtf(
            version="19",
            force=force,
            keep_compressed=keep_gtf_compressed,
        )

        logger.info("\n" + "=" * 60)
        logger.info("All reference files downloaded successfully!")
        logger.info("=" * 60)

        return downloaded

    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        logger.error(
            "\nYou can manually download files and place them in:"
        )
        logger.error(f"  {DATA_DIR}")
        raise


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download reference data files for sigmutsel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files with defaults
  python -m sigmutsel setup

  # Re-download even if files exist
  python -m sigmutsel setup --force

  # Keep FASTA compressed
  python -m sigmutsel setup --keep-fasta-compressed

  # Decompress GTF files
  python -m sigmutsel setup --decompress-gtf
        """,
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if files exist",
    )
    parser.add_argument(
        "--keep-fasta-compressed",
        action="store_true",
        help="Keep FASTA file compressed (.fa.gz)",
    )
    parser.add_argument(
        "--decompress-gtf",
        action="store_true",
        help="Decompress GTF files (default: keep compressed)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Custom data directory (default: package data dir)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set custom data directory if specified
    if args.data_dir:
        global DATA_DIR
        DATA_DIR = args.data_dir
        from sigmutsel import locations

        locations.DATA_DIR = args.data_dir

    # Download files
    try:
        download_all(
            force=args.force,
            decompress_fasta=not args.keep_fasta_compressed,
            keep_gtf_compressed=not args.decompress_gtf,
        )
    except Exception as e:
        logger.error(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
