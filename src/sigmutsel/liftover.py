"""Point liftover from UCSC chain files, without external tools.

Only what :mod:`sigmutsel.channel_universe` needs: map individual
1-based positions from one assembly to another through a UCSC
``.over.chain`` file, vectorised with numpy, and test the mapped
positions against a BED file.

The direction matters. The MC3 capture BED is hg19 and everything
else in this package is GRCh38, so the territory test lifts each
GRCh38 *position* to hg19 and asks whether it lies inside the BED,
rather than lifting the BED's *intervals* to GRCh38. Positions map
one to one where intervals do not, and that avoids both traps an
interval liftover has (cancereffectsizeR's liftOver notes): an
interval that lands on the reverse strand needs its coordinates
flipped, and two source loci can land on one target locus and
create duplicates. A membership test cares about neither --
strand does not change whether a position is inside a target, and
two GRCh38 positions landing on one hg19 position are simply both
inside or both outside.

The hg38ToHg19 chain covers each GRCh38 position at most once
(checked by :func:`read_chain`), so the point map is a function.
"""

import gzip
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_chain(path) -> pd.DataFrame:
    """Parse a (gzipped) UCSC chain file into aligned blocks.

    Returns
    -------
    pandas.DataFrame
        One row per ungapped block, with columns ``tchrom``,
        ``tstart``, ``tend`` (0-based half-open, source assembly),
        ``qchrom``, ``qstart`` (0-based, on the query strand),
        ``qstrand`` and ``qsize``. A position ``t`` in
        ``[tstart, tend)`` maps to ``qstart + t - tstart`` on
        ``qstrand``; on ``-`` that is converted to the forward
        strand as ``qsize - 1 - q``.

    Raises
    ------
    ValueError
        If two blocks cover the same source position, since the
        point map would then not be a function.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("chain"):
                h = line.split()
                tchrom, tpos = h[2], int(h[5])
                qchrom, qsize, qstrand, qpos = (
                    h[7],
                    int(h[8]),
                    h[9],
                    int(h[10]),
                )
                continue
            fields = line.split()
            if not fields:
                continue
            size = int(fields[0])
            rows.append(
                (
                    tchrom,
                    tpos,
                    tpos + size,
                    qchrom,
                    qpos,
                    qstrand,
                    qsize,
                )
            )
            if len(fields) == 3:
                tpos += size + int(fields[1])
                qpos += size + int(fields[2])

    blocks = pd.DataFrame(
        rows,
        columns=[
            "tchrom",
            "tstart",
            "tend",
            "qchrom",
            "qstart",
            "qstrand",
            "qsize",
        ],
    )
    blocks = blocks.sort_values(["tchrom", "tstart"]).reset_index(
        drop=True
    )
    for chrom, group in blocks.groupby("tchrom"):
        ends = np.maximum.accumulate(group["tend"].to_numpy())
        if (group["tstart"].to_numpy()[1:] < ends[:-1]).any():
            raise ValueError(
                f"Chain blocks overlap on {chrom}: a position would "
                "map to more than one target."
            )
    return blocks


def lift_positions(blocks, chroms, positions):
    """Map 1-based positions through chain blocks.

    Parameters
    ----------
    blocks : pandas.DataFrame
        As returned by :func:`read_chain`.
    chroms : array-like of str
        Source chromosome of each position (e.g. ``"chr1"``).
    positions : array-like of int
        1-based source positions.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Target chromosome (object array, ``None`` where unmapped)
        and 1-based target position (``-1`` where unmapped).
    """
    chroms = np.asarray(chroms, dtype=object)
    pos0 = np.asarray(positions, dtype=np.int64) - 1
    out_chrom = np.full(len(pos0), None, dtype=object)
    out_pos = np.full(len(pos0), -1, dtype=np.int64)

    by_chrom = dict(tuple(blocks.groupby("tchrom")))
    for chrom in pd.unique(chroms):
        group = by_chrom.get(chrom)
        if group is None:
            continue
        idx = np.flatnonzero(chroms == chrom)
        starts = group["tstart"].to_numpy()
        ends = group["tend"].to_numpy()
        j = np.searchsorted(starts, pos0[idx], side="right") - 1
        jj = np.clip(j, 0, None)
        hit = (j >= 0) & (pos0[idx] < ends[jj])
        offset = pos0[idx] - starts[jj]
        qstart = group["qstart"].to_numpy()[jj]
        qsize = group["qsize"].to_numpy()[jj]
        minus = group["qstrand"].to_numpy()[jj] == "-"
        q = np.where(
            minus, qsize - 1 - (qstart + offset), qstart + offset
        )
        out_chrom[idx[hit]] = group["qchrom"].to_numpy()[jj][hit]
        out_pos[idx[hit]] = q[hit] + 1
    return out_chrom, out_pos


def merged_bed_intervals(bed: pd.DataFrame, padding: int = 0):
    """Per-chromosome disjoint intervals of a BED, optionally padded.

    Parameters
    ----------
    bed : pandas.DataFrame
        Columns ``chrom``, ``start`` (0-based), ``end`` (exclusive).
    padding : int
        Bases added on both sides of every interval before merging.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Chromosome to sorted, disjoint ``(starts, ends)``.
    """
    merged = {}
    for chrom, group in bed.groupby("chrom"):
        group = group.sort_values("start")
        starts = group["start"].to_numpy() - padding
        ends = group["end"].to_numpy() + padding
        keep_s, keep_e = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= keep_e[-1]:
                keep_e[-1] = max(keep_e[-1], e)
            else:
                keep_s.append(s)
                keep_e.append(e)
        merged[chrom] = (np.array(keep_s), np.array(keep_e))
    return merged


def in_intervals(merged, chroms, positions):
    """Whether each 1-based position lies in the merged intervals.

    Unmapped positions (chromosome ``None`` or position ``-1``) are
    outside.
    """
    chroms = np.asarray(chroms, dtype=object)
    pos0 = np.asarray(positions, dtype=np.int64) - 1
    inside = np.zeros(len(pos0), dtype=bool)
    for chrom, (starts, ends) in merged.items():
        idx = np.flatnonzero(chroms == chrom)
        if not len(idx):
            continue
        j = np.searchsorted(starts, pos0[idx], side="right") - 1
        inside[idx] = (j >= 0) & (
            pos0[idx] < ends[np.clip(j, 0, None)]
        )
    return inside
