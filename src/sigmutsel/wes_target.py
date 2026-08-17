"""MC3 WES-target gene universe.

Computes the set of Ensembl gene IDs (version stripped) overlapping
TCGA's uniform whole-exome capture-kit intersection, as published by
MC3 (Ellrott et al. 2018, *Scalable Open Science Approach for
Mutation Calling of Tumor Exomes Using Multiple Genomic Pipelines*,
Cell Systems). MC3 built this single BED file
(`gaf_20111020Plusbroad_wex_1.1_hg19.bed`) as the intersection of
capture kits used across TCGA sequencing centers, applied uniformly
across all 33 TCGA cohorts specifically because reliable per-cohort
or per-sample capture-kit metadata was never available -- the same
problem this module exists to solve.

The BED is in hg19, so it's intersected against GENCODE v19 gene
coordinates (also hg19) rather than the GRCh38 coordinates used
elsewhere in this package. No liftover is needed: Ensembl gene IDs
are stable across GENCODE releases (version suffix aside), so the
resulting gene-ID set applies directly to GRCh38-based tables like
``contexts_by_gene`` -- this package already joins hg19/hg38 sources
this way.

Typical usage is through :func:`get_wes_target_gene_ids`, which
downloads and caches both source files and the derived gene-ID list.
"""

import gzip
import logging
import re

import numpy as np
import pandas as pd

from . import setup
from .locations import (
    location_gencode19_annotation,
    location_wes_target_bed,
    location_wes_target_gene_ids,
)

logger = logging.getLogger(__name__)

_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')


def _parse_bed(bed_path) -> pd.DataFrame:
    """Read a 3-column BED file, prefixing chromosome names with
    ``chr`` to match GENCODE's convention."""
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end"],
    )
    bed["chrom"] = "chr" + bed["chrom"].astype(str)
    return bed


def _parse_gtf_genes(gtf_path) -> pd.DataFrame:
    """Extract gene-level rows from a (possibly gzipped) GTF file.

    Returns a DataFrame with columns chrom/start/end/gene_id, where
    gene_id has its version suffix stripped.
    """
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    genes = []
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            m = _GENE_ID_RE.search(fields[8])
            gene_id = m.group(1).split(".")[0]
            genes.append(
                (fields[0], int(fields[3]), int(fields[4]), gene_id)
            )
    return pd.DataFrame(
        genes, columns=["chrom", "start", "end", "gene_id"]
    )


def _merge_intervals(starts: np.ndarray, ends: np.ndarray):
    """Merge overlapping/adjacent sorted intervals into a disjoint set.

    ``starts``/``ends`` must already be sorted by start position.
    Returns ``(merged_starts, merged_ends)`` as numpy arrays.
    """
    merged = []
    cur_s, cur_e = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    merged_starts = np.array([m[0] for m in merged])
    merged_ends = np.array([m[1] for m in merged])
    return merged_starts, merged_ends


def _overlapping_gene_ids(
    bed_df: pd.DataFrame, genes_df: pd.DataFrame
) -> set:
    """Return gene IDs whose interval overlaps at least one BED
    interval on the same chromosome.

    Per chromosome: merge the BED intervals into a disjoint sorted
    set, then for each gene binary-search for the nearest merged
    interval and check a small neighborhood for overlap. This is a
    plain pandas/numpy interval sweep (no bedtools/pybedtools
    dependency).
    """
    overlapping_ids = set()
    for chrom, bed_chr in bed_df.groupby("chrom"):
        genes_chr = genes_df[genes_df["chrom"] == chrom]
        if genes_chr.empty:
            continue

        intervals = (
            bed_chr[["start", "end"]].sort_values("start").to_numpy()
        )
        merged_starts, merged_ends = _merge_intervals(
            intervals[:, 0], intervals[:, 1]
        )

        for _, g in genes_chr.iterrows():
            idx = (
                np.searchsorted(merged_starts, g["end"], side="right")
                - 1
            )
            lo, hi = max(0, idx - 1), min(len(merged_starts), idx + 2)
            for i in range(lo, hi):
                if (
                    merged_starts[i] <= g["end"]
                    and merged_ends[i] >= g["start"]
                ):
                    overlapping_ids.add(g["gene_id"])
                    break
    return overlapping_ids


def compute_wes_target_gene_ids(bed_path=None, gtf_path=None) -> set:
    """Compute the WES-target gene-ID set from source files.

    Pure computation, no caching or downloading -- see
    :func:`get_wes_target_gene_ids` for the cached entry point most
    callers want.

    Parameters
    ----------
    bed_path : str, Path, or None
        MC3 WES-target BED file. Defaults to
        ``locations.location_wes_target_bed``.
    gtf_path : str, Path, or None
        GENCODE v19 GTF file (hg19, matching the BED). Defaults to
        ``locations.location_gencode19_annotation``.

    Returns
    -------
    set[str]
        Ensembl gene IDs (version stripped) overlapping the WES
        target.
    """
    bed_path = bed_path or location_wes_target_bed
    gtf_path = gtf_path or location_gencode19_annotation

    bed_df = _parse_bed(bed_path)
    genes_df = _parse_gtf_genes(gtf_path)
    return _overlapping_gene_ids(bed_df, genes_df)


def get_wes_target_gene_ids(force_recompute: bool = False) -> set:
    """Cached WES-target gene-ID set, downloading source files if needed.

    This is the entry point :class:`~sigmutsel.models.MutationDataset`
    uses. On first call (or with ``force_recompute=True``), downloads
    the MC3 BED and GENCODE v19 GTF via :mod:`sigmutsel.setup` if not
    already present, computes the overlap, and caches the resulting
    gene-ID list to :data:`locations.location_wes_target_gene_ids`.
    Subsequent calls just read the cache.

    Parameters
    ----------
    force_recompute : bool, default False
        If True, ignore any cached gene-ID list and recompute from
        the source files (re-downloading them first if missing).

    Returns
    -------
    set[str]
        Ensembl gene IDs (version stripped) overlapping the WES
        target.
    """
    if location_wes_target_gene_ids.exists() and not force_recompute:
        with open(location_wes_target_gene_ids) as f:
            return set(f.read().splitlines())

    setup.download_wes_target_bed()
    setup.download_gencode_gtf(version="19")

    gene_ids = compute_wes_target_gene_ids()

    location_wes_target_gene_ids.parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(location_wes_target_gene_ids, "w") as f:
        f.write("\n".join(sorted(gene_ids)))
    logger.info(
        "Computed WES-target gene universe: %d genes (cached at %s)",
        len(gene_ids),
        location_wes_target_gene_ids,
    )
    return gene_ids
