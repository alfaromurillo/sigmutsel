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
    "hg38ToHg19.over.chain.gz": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/"
        "hg38ToHg19.over.chain.gz"
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


def download_liftover_chain(force: bool = False) -> Path:
    """Download UCSC's hg38-to-hg19 liftOver chain (~1.2 MB).

    Used by :mod:`sigmutsel.channel_universe` to test GRCh38
    positions against the hg19 MC3 capture BED one position at a
    time (see :mod:`sigmutsel.liftover` for why the direction is
    hg38 to hg19 and not the other way round).

    Parameters
    ----------
    force : bool, default False
        If True, download even if the file exists.

    Returns
    -------
    Path
        Path to the gzipped chain file.
    """
    filename = "hg38ToHg19.over.chain.gz"
    dest = DATA_DIR / filename
    if dest.exists() and not force:
        logger.info(f"liftOver chain already exists at {dest}")
        return dest
    return download_file(
        DOWNLOAD_URLS[filename], dest, decompress=False
    )


def download_repeatmasker_bed(
    genome_build: str = "hg38",
    force: bool = False,
    full: bool = False,
) -> Path:
    """The RepeatMasker regions the call QC and the territory use.

    Parameters
    ----------
    genome_build : {"hg38", "hg19"}, default "hg38"
        hg38 matches this package's usual GRCh38 MAF coordinates
        (see the module docstring of :mod:`wes_target` for why the
        rest of the package is hg38-based); hg19 is provided for
        callers working against hg19 coordinates directly.
    force : bool, default False
        If True, download even if the file exists.
    full : bool, default False
        hg38 only. If True, UCSC's full rmsk table (149 MB, every
        column); otherwise the full table if it is already present,
        else the distributed table of merged intervals (17 MB; see
        :mod:`repeatmasker`), which gives the same intervals. hg19
        always takes the full table.

    Returns
    -------
    Path
        Either UCSC's raw tab-separated ``rmsk.<build>.txt.gz`` dump
        or the distributed BED (``repeatmasker.DISTRIBUTED_NAME``).
        :func:`qc.load_repeat_intervals` reads both.

    Notes
    -----
    Kept compressed: pandas reads ``.gz`` and ``.xz`` files directly,
    so there's no need to decompress them to disk.

    Examples
    --------
    >>> from sigmutsel.qc import load_repeat_intervals
    >>> from sigmutsel.setup import download_repeatmasker_bed
    >>> intervals = load_repeat_intervals(download_repeatmasker_bed("hg38"))
    >>> dataset.generate_mutation_db(
    ...     qc_mode=True, qc_kwargs={"repeat_intervals": intervals})
    """
    if genome_build not in ("hg38", "hg19"):
        raise ValueError(
            f"genome_build must be 'hg38' or 'hg19', got {genome_build!r}."
        )
    filename = f"rmsk.{genome_build}.txt.gz"
    dest = DATA_DIR / filename

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

    if genome_build == "hg38" and not full:
        if not force and dest.exists():
            logger.info(
                f"Using the full RepeatMasker table at {dest}"
            )
            return dest
        if not force and shared.exists():
            logger.info(
                f"Reusing the full RepeatMasker table at {shared}"
            )
            shutil.copyfile(shared, dest)
            return dest
        from sigmutsel.repeatmasker import (
            DISTRIBUTED_NAME,
            download_distributed_table,
        )

        return download_distributed_table(
            DATA_DIR / DISTRIBUTED_NAME, force=force
        )

    if dest.exists() and not force:
        logger.info(
            f"RepeatMasker ({genome_build}) already exists at {dest}"
        )
        return dest

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sigmutsel.setup import download_tcga_absolute_purity
    >>> purity = pd.read_csv(download_tcga_absolute_purity(), sep="\t")
    >>> purity[["array", "purity", "ploidy"]].head(2)
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
        point mutation's, which the purity/CN-corrected VAF QC of
        :mod:`sigmutsel.sample_qc` relies on).

    Notes
    -----
    Companion file: :func:`download_tcga_absolute_purity`. Same
    source as that function's docstring.

    Examples
    --------
    >>> from sigmutsel.sample_qc import load_copy_number_segments_from_file
    >>> from sigmutsel.setup import download_tcga_absolute_segments
    >>> segments = load_copy_number_segments_from_file(
    ...     download_tcga_absolute_segments())
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
    genome: bool = True,
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
    genome : bool, default True
        Also install SigProfilerMatrixGenerator's GRCh38 genome
        (:func:`install_reference_genome`, about 3 GB), which it keeps
        inside its own package directory, not in the data directory.

    Returns
    -------
    dict[str, Path]
        Mapping of file type to downloaded path

    Examples
    --------
    From the command line, ``sigmutsel-setup`` does the same (and
    ``sigmutsel-setup --data-dir DIR`` into another directory):

    >>> from sigmutsel.setup import download_all
    >>> paths = download_all()  # about 300 MB plus the 3 GB genome
    >>> sorted(paths)[:3]
    ['cds_fasta', 'gencode19', 'gencode38']
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

        # hg38 -> hg19 chain for the capture territory
        logger.info("\n6. UCSC hg38ToHg19 liftOver chain")
        downloaded["liftover_chain"] = download_liftover_chain(
            force=force
        )

        # Germline-mask table (gnomAD v2.1.1 exomes, AF > 5e-5)
        logger.info("\n7. Germline-mask table (gnomAD v2.1.1, 15 MB)")
        downloaded["germline_mask"] = download_germline_mask(
            force=force
        )

        # RepeatMasker intervals (hg38, merged, for the call QC and
        # the territory)
        logger.info("\n8. RepeatMasker intervals (hg38, 17 MB)")
        downloaded["repeatmasker"] = download_repeatmasker_bed(
            "hg38", force=force
        )

        # The WES-target gene universe, derived from items 4 and 5
        # (MutationDataset's default gene universe; seconds to build)
        logger.info("\n9. MC3 WES-target gene list (derived)")
        from sigmutsel import wes_target

        wes_target.get_wes_target_gene_ids(force_recompute=force)
        downloaded["wes_target_gene_ids"] = (
            wes_target.location_wes_target_gene_ids
        )

        if genome:
            logger.info(
                "\n10. GRCh38 reference genome (SigProfilerMatrixGenerator)"
            )
            # Not reinstalled by force=True: 3 GB, and an installed
            # genome does not go stale (install_reference_genome(
            # force=True) reinstalls it).
            downloaded["genome"] = install_reference_genome("GRCh38")

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


def install_reference_genome(
    build: str = "GRCh38", force: bool = False
) -> Path:
    """Install SigProfilerMatrixGenerator's reference genome (~3 GB).

    The signature decomposition and every step that reads the genome
    (pentanucleotide contexts, splice-flank compositions, the
    opportunity across exon junctions) use SigProfilerMatrixGenerator's
    installed reference. Without it those steps do not fail: they
    warn once and fall back, and the rates differ from an environment
    that has it. Install it before building a dataset.

    Parameters
    ----------
    build : str, default "GRCh38"
        Genome build, as SigProfilerMatrixGenerator names it.
    force : bool, default False
        Reinstall even if the genome is already installed.

    Returns
    -------
    Path
        The installed genome's directory
        (``references/chromosomes/tsb/<build>`` inside the
        SigProfilerMatrixGenerator package).

    Examples
    --------
    >>> from sigmutsel.setup import install_reference_genome
    >>> install_reference_genome("GRCh38")  # doctest: +SKIP
    """
    from ._global_state import keep_global_state

    with keep_global_state():
        import SigProfilerMatrixGenerator
        from SigProfilerMatrixGenerator import install as genInstall

    # The directory sigprofiler_genome_dir reads, checked without its
    # missing-genome warning (its absence is what this installs).
    path = (
        Path(SigProfilerMatrixGenerator.__file__).parent
        / "references"
        / "chromosomes"
        / "tsb"
        / build
    )
    if path.is_dir() and any(path.glob("*.txt")) and not force:
        logger.info(f"{build} reference genome already at {path}")
        return path
    logger.info(
        f"Installing the {build} reference genome with "
        "SigProfilerMatrixGenerator (about 3 GB; several minutes)..."
    )
    with keep_global_state():
        genInstall.install(build, rsync=False, bash=True)
    if not (path.is_dir() and any(path.glob("*.txt"))):
        raise RuntimeError(
            f"SigProfilerMatrixGenerator did not install {build} at "
            f"{path}"
        )
    return path


def download_germline_mask(force: bool = False) -> Path:
    """Download the germline-mask table (15 MB, gnomAD AF > 5e-5).

    See :mod:`sigmutsel.germline_mask`; enough for any mask threshold
    at or above 5e-5.
    """
    from sigmutsel.germline_mask import download_germline_mask_table

    return download_germline_mask_table(force=force)


def build_full_germline_table(force: bool = False) -> Path:
    """Build the full gnomAD SNV table (every AF > 0).

    Not part of :func:`download_all`: it streams about 92 GB (nothing
    but a 14.6-million-row table is kept). Only needed for a mask
    below AF 5e-5, or to measure call survival by frequency. See
    :mod:`sigmutsel.germline_mask`.
    """
    from sigmutsel.germline_mask import (
        build_gnomad_snv_table,
        default_table_path,
    )

    path = default_table_path()
    if path.exists() and not force:
        logger.info(f"gnomAD SNV table already exists at {path}")
        return path
    return build_gnomad_snv_table(path)


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

  # Download into another directory (then export
  # SIGMUTSEL_DATA_DIR=/path/to/data in every session that uses it)
  python -m sigmutsel setup --data-dir /path/to/data

  # Without the 3 GB GRCh38 genome (a dataset build needs it)
  python -m sigmutsel setup --skip-genome

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
        help=(
            "Keep FASTA file compressed (.fa.gz); building a dataset "
            "reads the uncompressed file, so only for a download to "
            "decompress later"
        ),
    )
    parser.add_argument(
        "--decompress-gtf",
        action="store_true",
        help="Decompress GTF files (default: keep compressed)",
    )
    parser.add_argument(
        "--germline-mask-full",
        action="store_true",
        help=(
            "Also build the full gnomAD SNV table (every AF > 0), "
            "for masks below AF 5e-5; WARNING: streams about 92 GB"
        ),
    )
    parser.add_argument(
        "--repeatmasker-full",
        action="store_true",
        help=(
            "Also download UCSC's full hg38 rmsk table (149 MB, every "
            "column); the default merged intervals give the same QC"
        ),
    )
    parser.add_argument(
        "--skip-genome",
        action="store_true",
        help=(
            "Do not install SigProfilerMatrixGenerator's GRCh38 genome "
            "(about 3 GB); every dataset build needs it"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Custom data directory (default: package data dir)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # A custom data directory must be set before sigmutsel is
    # imported: every module resolves its file paths at import time,
    # so patching DATA_DIR here would send some files to the custom
    # directory and others to the package's. Run the same command
    # again in a fresh process with SIGMUTSEL_DATA_DIR set instead.
    if args.data_dir:
        import os
        import subprocess
        import sys

        from sigmutsel import locations

        data_dir = args.data_dir.expanduser().resolve()
        if data_dir != Path(locations.DATA_DIR).resolve():
            argv, skip = [], False
            for arg in sys.argv[1:]:
                if skip:
                    skip = False
                elif arg == "--data-dir":
                    skip = True
                elif not arg.startswith("--data-dir="):
                    argv.append(arg)
            env = {**os.environ, "SIGMUTSEL_DATA_DIR": str(data_dir)}
            data_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.call(
                [sys.executable, "-m", "sigmutsel.setup", *argv],
                env=env,
            )

    if args.keep_fasta_compressed:
        logger.warning(
            "--keep-fasta-compressed: building a dataset reads the "
            "uncompressed CDS FASTA; decompress it (gunzip) before "
            "building one."
        )

    # Download files
    try:
        download_all(
            force=args.force,
            decompress_fasta=not args.keep_fasta_compressed,
            keep_gtf_compressed=not args.decompress_gtf,
            genome=not args.skip_genome,
        )
        if args.repeatmasker_full:
            download_repeatmasker_bed(
                "hg38", force=args.force, full=True
            )
        if args.germline_mask_full:
            logger.warning(
                "Building the full gnomAD SNV table streams about 92 GB."
            )
            build_full_germline_table(force=args.force)
    # Top-level CLI handler: print a clean error and a non-zero exit
    # code instead of a raw traceback, regardless of failure cause.
    except Exception as e:  # noqa: BLE001
        logger.error(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
