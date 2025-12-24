"""Configuration and data location management for sigmutsel.

DEPRECATED: Use sigmutsel.locations instead.

This module is kept for backward compatibility but new code should
use sigmutsel.locations which provides the same functionality with
better integration.
"""

import warnings

warnings.warn(
    "sigmutsel.config is deprecated, use sigmutsel.locations instead",
    DeprecationWarning,
    stacklevel=2
)

import os
from pathlib import Path


# Default data directory (can be overridden by environment variable)
DEFAULT_DATA_DIR = Path(
    os.environ.get('SIGMUTSEL_DATA_DIR', Path.home() / '.sigmutsel'))


def get_data_path(filename: str | None = None) -> Path:
    """Get path to data directory or specific data file.

    Parameters
    ----------
    filename : str or None
        If provided, returns path to specific file in data directory.
        If None, returns the data directory itself.

    Returns
    -------
    Path
        Path to data directory or file.

    Examples
    --------
    >>> # Get data directory
    >>> data_dir = get_data_path()

    >>> # Get specific file
    >>> hgnc_file = get_data_path('hgnc_complete_set.txt')

    Notes
    -----
    Set the SIGMUTSEL_DATA_DIR environment variable to change the
    default data location:

        export SIGMUTSEL_DATA_DIR=/path/to/your/data
    """
    data_dir = DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        return data_dir
    return data_dir / filename


# Common reference file locations (optional defaults)
def get_hgnc_file() -> Path | None:
    """Get HGNC complete set file if available.

    Returns None if not found, requiring user to provide path.
    """
    path = get_data_path('hgnc_complete_set.txt')
    return path if path.exists() else None


def get_cancer_gene_census() -> Path | None:
    """Get Cancer Gene Census file if available."""
    path = get_data_path('Cosmic_CancerGeneCensus_v101_GRCh38.tsv')
    return path if path.exists() else None


def get_cds_fasta() -> Path | None:
    """Get CDS FASTA file if available."""
    path = get_data_path('Homo_sapiens.GRCh38.cds.all.fa')
    return path if path.exists() else None
