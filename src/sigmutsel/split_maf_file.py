"""Split a multi-sample MAF file into per-sample MAF files.

Reads a single MAF file containing multiple samples and writes one
individual MAF file per sample, suitable as input for
SigProfilerMatrixGenerator.  Comment lines (lines starting with '#')
are preserved verbatim in every output file.
"""

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def split_maf_file(
        maf_file: str | Path,
        output_dir: str | Path,
        *,
        force_generation: bool = False) -> Path:
    """Split a multi-sample MAF file into per-sample MAF files.

    Reads a tab-delimited MAF file that may contain an arbitrary
    number of leading comment lines (starting with ``#``), followed
    by a header row and data rows keyed on
    ``Tumor_Sample_Barcode``.  One ``.maf`` file per sample is
    written to *output_dir*; comment lines from the source file are
    prepended to every output file.

    If *output_dir* already contains ``.maf`` files and
    *force_generation* is ``False``, the split is skipped entirely.

    Parameters
    ----------
    maf_file : str or pathlib.Path
        Path to the multi-sample MAF file.
    output_dir : str or pathlib.Path
        Directory where per-sample ``.maf`` files are written.
        Created (including parents) if absent.
    force_generation : bool, default False
        If ``True``, delete *output_dir* and redo the split from
        scratch.  If ``False``, skip when ``.maf`` files are already
        present.

    Returns
    -------
    pathlib.Path
        Resolved path to *output_dir*.

    Raises
    ------
    KeyError
        If ``Tumor_Sample_Barcode`` is not a column in *maf_file*.
    """
    maf_file = Path(maf_file)
    output_dir = Path(output_dir)

    if force_generation and output_dir.exists():
        logger.info(f"Deleting previous split files from {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not force_generation and any(output_dir.glob("*.maf")):
        logger.info(
            f"Per-sample MAF files already present in "
            f"{output_dir}, skipping split.")
        return output_dir

    # Capture leading comment lines before pandas discards them.
    comment_lines = []
    with maf_file.open("r") as fh:
        for line in fh:
            if line.startswith("#"):
                comment_lines.append(line)
            else:
                break
    comment_block = "".join(comment_lines)

    logger.info(f"Reading MAF file: {maf_file}")
    df = pd.read_csv(
        maf_file, sep="\t", comment="#", low_memory=False)

    if "Tumor_Sample_Barcode" not in df.columns:
        raise KeyError(
            "Column 'Tumor_Sample_Barcode' not found in "
            f"{maf_file.name}. Cannot split.")

    n_samples = df["Tumor_Sample_Barcode"].nunique()
    logger.info(
        f"Loaded {len(df):,} mutations across "
        f"{n_samples} samples.")

    n_written = 0
    for sample, sample_df in df.groupby("Tumor_Sample_Barcode"):
        out_path = output_dir / f"{sample}.maf"

        with out_path.open("w") as fh:
            fh.write(comment_block)
            sample_df.to_csv(fh, sep="\t", index=False)

        logger.debug(
            f"Wrote {len(sample_df):,} mutations for "
            f"{sample} -> {out_path.name}")
        n_written += 1

    logger.info(
        f"Done: {n_written}/{n_samples} sample files written "
        f"to {output_dir}.")
    return output_dir
