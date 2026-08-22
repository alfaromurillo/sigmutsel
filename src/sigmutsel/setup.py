"""Download reference data files for sigmutsel.

This module handles downloading large reference files that are not
included in the package distribution. Users can run this as:

    python -m sigmutsel setup

Or import and call programmatically:

    from sigmutsel.setup import download_all
    download_all()
"""

import gzip
import logging
import os
import shutil
import urllib.request
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
    "gaf_20111020Plusbroad_wex_1.1_hg19.bed": (
        "https://api.gdc.cancer.gov/data/"
        "b1e303a5-a542-4389-8ddb-1d151218be75"
    ),
    "rmsk.hg38.txt.gz": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/"
        "rmsk.txt.gz"
    ),
    "rmsk.hg19.txt.gz": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/"
        "rmsk.txt.gz"
    ),
    "TCGA_mastercalls.abs_tables_JSedit.fixed.txt": (
        "https://api.gdc.cancer.gov/data/"
        "4f277128-f793-4354-a13d-30cc7fe9f6b5"
    ),
    "TCGA_mastercalls.abs_segtabs.fixed.txt": (
        "https://api.gdc.cancer.gov/data/"
        "0f4f5701-7b61-41ae-bda9-2805d1ca9781"
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
            with (
                gzip.open(temp_file, "rb") as f_in,
                open(final_dest, "wb") as f_out,
            ):
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

    # GENCODE GTFs are a universal reference, not specific to this
    # package -- sigmutselcovs and any other consumer that follows
    # the same convention share this location (GENCODE_DATA_HOME, or
    # $XDG_DATA_HOME/gencode; data, not cache, since these files are
    # worth keeping, not disposable). No import here -- sigmutselcovs
    # is a separate, optional package -- just a filesystem
    # convention both sides know about, so a user with both
    # installed never downloads the same file twice.
    xdg_data_home = Path(
        os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )
    shared = (
        Path(
            os.environ.get(
                "GENCODE_DATA_HOME", str(xdg_data_home / "gencode")
            )
        )
        / filename
    )
    if shared.exists():
        logger.info(f"Reusing GENCODE v{version} already at {shared}")
        if keep_compressed:
            shutil.copyfile(shared, dest)
            return dest
        with (
            gzip.open(shared, "rb") as f_in,
            open(final_dest, "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        return final_dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=not keep_compressed)


def download_wes_target_bed(force: bool = False) -> Path:
    """Download MC3's TCGA WES capture-kit target BED.

    Parameters
    ----------
    force : bool, default False
        If True, download even if file exists

    Returns
    -------
    Path
        Path to downloaded file

    Notes
    -----
    This is `gaf_20111020Plusbroad_wex_1.1_hg19.bed` from MC3
    (Ellrott et al. 2018, Cell Systems) -- the intersection of TCGA's
    WES capture kits across sequencing centers, applied uniformly
    across all 33 TCGA cohorts. hg19, not gzipped, ~5MB.
    """
    filename = "gaf_20111020Plusbroad_wex_1.1_hg19.bed"
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        logger.info(f"WES target BED already exists at {dest}")
        return dest

    # Same shared-cache convention as download_gencode_gtf: a
    # filesystem convention, not an import, so any consumer that
    # follows the same env-var + filename pattern shares the
    # download without duplicating it.
    xdg_data_home = Path(
        os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )
    shared = (
        Path(
            os.environ.get(
                "MC3_WES_TARGET_DATA_HOME",
                str(xdg_data_home / "mc3_wes_target"),
            )
        )
        / filename
    )
    if shared.exists():
        logger.info(f"Reusing WES target BED already at {shared}")
        shutil.copyfile(shared, dest)
        return dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=False)


def download_repeatmasker_bed(
    genome_build: str = "hg38", force: bool = False
) -> Path:
    """Download UCSC's RepeatMasker (rmsk) track.

    Parameters
    ----------
    genome_build : {"hg38", "hg19"}, default "hg38"
        hg38 matches this package's usual GRCh38 MAF coordinates
        (see the module docstring of :mod:`wes_target` for why the
        rest of the package is hg38-based); hg19 is provided for
        callers working against hg19 coordinates directly.
    force : bool, default False
        If True, download even if the file exists.

    Returns
    -------
    Path
        Path to the downloaded ``rmsk.<build>.txt.gz`` file. This is
        UCSC's raw tab-separated rmsk table dump (not a BED file) --
        see :func:`qc.load_repeat_intervals` for parsing it into
        per-chromosome intervals.

    Notes
    -----
    Kept gzipped -- hg38's rmsk table is genome-wide (millions of
    rows); pandas reads a ``.gz`` file directly, so there's no need
    to decompress it to disk.
    """
    if genome_build not in ("hg38", "hg19"):
        raise ValueError(
            f"genome_build must be 'hg38' or 'hg19', got {genome_build!r}."
        )
    filename = f"rmsk.{genome_build}.txt.gz"
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        logger.info(
            f"RepeatMasker ({genome_build}) already exists at {dest}"
        )
        return dest

    # Same shared-cache convention as download_gencode_gtf/
    # download_wes_target_bed: a filesystem convention, not an
    # import, so any consumer following the same env-var + filename
    # pattern shares the download without duplicating a genome-wide
    # table.
    xdg_data_home = Path(
        os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )
    shared = (
        Path(
            os.environ.get(
                "REPEATMASKER_DATA_HOME",
                str(xdg_data_home / "repeatmasker"),
            )
        )
        / filename
    )
    if shared.exists():
        logger.info(
            f"Reusing RepeatMasker ({genome_build}) already at {shared}"
        )
        shutil.copyfile(shared, dest)
        return dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=False)


def download_tcga_absolute_purity(force: bool = False) -> Path:
    """Download TCGA's Pan-Cancer Atlas ABSOLUTE purity/ploidy calls.

    A single pan-cancer file covering every TCGA cohort and sample --
    not cohort-specific, so this only needs downloading once, ever,
    not once per cohort or per rerun.

    Parameters
    ----------
    force : bool, default False
        If True, download even if the file exists.

    Returns
    -------
    Path
        Path to ``TCGA_mastercalls.abs_tables_JSedit.fixed.txt``. One
        row per sample (15-character barcode, column ``array``), with
        columns including ``purity``, ``ploidy``, and
        ``Subclonal genome fraction``.

    Notes
    -----
    From the GDC PanCanAtlas publication resources (Liu et al. 2018,
    Hoadley et al. 2018) -- see
    https://gdc.cancer.gov/about-data/publications/PanCan-CellOfOrigin.
    Companion file: :func:`download_tcga_absolute_segments`.
    """
    filename = "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        logger.info(
            f"TCGA ABSOLUTE purity/ploidy already exists at {dest}"
        )
        return dest

    # Same shared-cache convention as download_gencode_gtf/
    # download_repeatmasker_bed -- a pan-cancer reference table, not
    # specific to any cohort or this package, so any consumer
    # following the same env-var + filename pattern shares the
    # download.
    xdg_data_home = Path(
        os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )
    shared = (
        Path(
            os.environ.get(
                "TCGA_ABSOLUTE_DATA_HOME",
                str(xdg_data_home / "tcga_absolute"),
            )
        )
        / filename
    )
    if shared.exists():
        logger.info(
            f"Reusing TCGA ABSOLUTE purity/ploidy already at {shared}"
        )
        shutil.copyfile(shared, dest)
        return dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=False)


def download_tcga_absolute_segments(force: bool = False) -> Path:
    """Download TCGA's Pan-Cancer Atlas ABSOLUTE copy-number segments.

    A single pan-cancer file (~250MB, ~1.9M rows) covering every TCGA
    cohort and sample -- not cohort-specific, so this only needs
    downloading once, ever, not once per cohort or per rerun.

    Parameters
    ----------
    force : bool, default False
        If True, download even if the file exists.

    Returns
    -------
    Path
        Path to ``TCGA_mastercalls.abs_segtabs.fixed.txt``. One row
        per (sample, segment), with ``Sample``, ``Chromosome``,
        ``Start``, ``End``, ``Modal_Total_CN``, and
        ``Cancer_cell_frac_a1``/``_a2`` columns (the latter describe
        the copy-number event's own clonality, not an individual
        point mutation's -- see the purity/CN-corrected VAF QC
        presentation, ``mutation_rates/presentations/
        2026_08_22_vaf_purity_qc.tex``, for why that distinction
        matters).

    Notes
    -----
    Companion file: :func:`download_tcga_absolute_purity`. Same
    source as that function's docstring.
    """
    filename = "TCGA_mastercalls.abs_segtabs.fixed.txt"
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        logger.info(f"TCGA ABSOLUTE segments already exist at {dest}")
        return dest

    xdg_data_home = Path(
        os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )
    shared = (
        Path(
            os.environ.get(
                "TCGA_ABSOLUTE_DATA_HOME",
                str(xdg_data_home / "tcga_absolute"),
            )
        )
        / filename
    )
    if shared.exists():
        logger.info(
            f"Reusing TCGA ABSOLUTE segments already at {shared}"
        )
        shutil.copyfile(shared, dest)
        return dest

    url = DOWNLOAD_URLS[filename]
    return download_file(url, dest, decompress=False)


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

        # Download MC3 WES target BED
        logger.info("\n5. MC3 WES Target BED")
        downloaded["wes_target_bed"] = download_wes_target_bed(
            force=force
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
    # Top-level CLI handler: print a clean error and a non-zero exit
    # code instead of a raw traceback, regardless of failure cause.
    except Exception as e:  # noqa: BLE001
        logger.error(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
