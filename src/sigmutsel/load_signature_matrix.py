"""Load and normalize mutational signature matrices.

This module provides functions to load mutational signature matrices
(e.g., COSMIC v3.x) and optionally normalize them by trinucleotide
context opportunity. It supports both genome-wide and exome-based
formats, and ensures consistent formatting of index and column labels.

- Functions
  ----------
  - load_signature_matrix :
      Load a signature matrix with mutation types as index and signatures
      as columns. Supports COSMIC-formatted or SigProfiler-derived files.

Usage
-----
>>> from signature_matrix_loader import load_signature_matrix
>>> sig_matrix = load_signature_matrix("/path/to/matrix.txt")

Notes
-----
- Signature matrices must contain either a 'MutationType' or 'Type'
  column for indexing.
- The returned DataFrames will always have index name 'type'.
"""

import os
import logging
import pandas as pd


logger = logging.getLogger(__name__)


def load_signature_matrix(location: str) -> pd.DataFrame:
    """Load a mutational signature matrix from file.

    Parameters
    ----------
    location : str
        Path to the signature matrix file. Must be a tab-delimited
        file with either a 'MutationType' or 'Type' column as index.

    Returns
    -------
    pd.DataFrame
        Signature matrix with mutation types as index (named 'type' to
        avoid confusion) and signature names as columns.

    Raises
    ------
    ValueError
        If the file does not exist or lacks the required columns.

    """
    if not os.path.exists(location):
        msg = f"Signature matrix file does not exist: {location}"
        logger.error(msg)
        raise ValueError(msg)

    df = pd.read_csv(location, sep="\t")

    if "MutationType" in df.columns:
        df = df.set_index("MutationType")
    elif "Type" in df.columns:
        df = df.set_index("Type")
    else:
        msg = ("Signature matrix must contain either a 'MutationType' "
               "or 'Type' column.")
        logger.error(msg)
        raise ValueError(msg)

    df.index.name = "type"
    return df
