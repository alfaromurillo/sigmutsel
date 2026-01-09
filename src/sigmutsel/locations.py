"""Location of reference data files for sigmutsel.

This module provides paths to reference data files bundled with the
package or downloaded during setup. Small reference files are
included in the package, while large files (FASTA, GTF, HGNC) are
downloaded on first use or via `sigmutsel-setup`.
"""

import os
from pathlib import Path


# Package data directory
_PKG_DATA_DIR = Path(__file__).parent / "data"

# Allow override via environment variable
DATA_DIR = Path(os.environ.get("SIGMUTSEL_DATA_DIR", _PKG_DATA_DIR))


# ============================================================
# Small files (included in package)
# ============================================================

location_exclusion_signatures_matrix = (
    DATA_DIR / "exclusion_signatures_matrix_by_cancer_type.txt"
)

location_inclusion_signatures_matrix = (
    DATA_DIR / "inclusion_signatures_matrix_by_cancer_type.txt"
)


# ============================================================
# Large files (downloaded on setup or first use)
# ============================================================

location_hgnc_complete_set = DATA_DIR / "hgnc_complete_set.txt"

location_cancer_gene_census = (
    DATA_DIR / "Census_allFri May 30 16_47_00 2025.tsv"
)

location_cosmic_cancer_gene_census = (
    DATA_DIR / "Cosmic_CancerGeneCensus_v101_GRCh38.tsv"
)

location_cds_fasta = DATA_DIR / "Homo_sapiens.GRCh38.cds.all.fa"

location_gencode38_annotation = (
    DATA_DIR / "gencode.v38.annotation.gtf.gz"
)

location_gencode19_annotation = (
    DATA_DIR / "gencode.v19.annotation.gtf.gz"
)


# ============================================================
# Helper functions
# ============================================================


def get_data_dir() -> Path:
    """Get the package data directory.

    Returns
    -------
    Path
        Path to data directory (either package default or user-configured)

    Notes
    -----
    Set SIGMUTSEL_DATA_DIR environment variable to use a custom
    location:

        export SIGMUTSEL_DATA_DIR=/path/to/your/data
    """
    return DATA_DIR


def check_data_file(file_path: Path, name: str = None) -> Path:
    """Check if a data file exists, provide helpful error if not.

    Parameters
    ----------
    file_path : Path
        Path to check
    name : str, optional
        Human-readable name of the file for error messages

    Returns
    -------
    Path
        The file path if it exists

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist, with instructions to download
    """
    if not file_path.exists():
        name = name or file_path.name
        raise FileNotFoundError(
            f"{name} not found at {file_path}.\n"
            f"Download it using:\n"
            f"    python -m sigmutsel setup\n"
            f"Or manually download and place in: {DATA_DIR}"
        )
    return file_path


def list_data_files() -> dict[str, bool]:
    """List all reference data files and their availability.

    Returns
    -------
    dict[str, bool]
        Mapping of file name to whether it exists
    """
    files = {
        "exclusion_signatures_matrix": location_exclusion_signatures_matrix,
        "inclusion_signatures_matrix": location_inclusion_signatures_matrix,
        "hgnc_complete_set": location_hgnc_complete_set,
        "cancer_gene_census": location_cancer_gene_census,
        "cosmic_cancer_gene_census": location_cosmic_cancer_gene_census,
        "cds_fasta": location_cds_fasta,
        "gencode38_annotation": location_gencode38_annotation,
        "gencode19_annotation": location_gencode19_annotation,
    }
    return {name: path.exists() for name, path in files.items()}


def print_data_status():
    """Print status of all reference data files."""
    print(f"Data directory: {DATA_DIR}\n")
    print("File status:")
    print("-" * 60)

    status = list_data_files()
    for name, exists in status.items():
        symbol = "✓" if exists else "✗"
        status_str = "Found" if exists else "Missing"
        print(f"{symbol} {name:<30} {status_str}")

    missing = [name for name, exists in status.items() if not exists]
    if missing:
        print("\nTo download missing files, run:")
        print("    python -m sigmutsel setup")
