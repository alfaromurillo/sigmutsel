"""RepeatMasker regions, as the call QC and the territory use them.

The call QC (:func:`qc.flag_repetitive_regions`) and the territory
(:func:`channel_universe.repeat_masks`) need only *where* the repeats
are: UCSC's rmsk rows merged into disjoint intervals per sequence
(:func:`qc.load_repeat_intervals`). Two files give those intervals,
for GRCh38 (hg38):

- **The distributed table** (default): the merged intervals of every
  sequence in UCSC's hg38 rmsk table, as a three-column BED (0-based,
  half-open), xz-compressed: 4.25 million intervals on 711 sequences,
  17 MB. Downloaded from a sigmutsel GitHub release and checked
  against its SHA-256. It is genome-wide, not cut to a capture
  territory, because calls in exome MAFs fall up to megabases from
  any target; so it gives exactly the intervals of the full table
  everywhere, and every repeat flag and mask is unchanged.
- **The full table**: UCSC's raw hg38 rmsk dump, 149 MB, with every
  column (repeat name, class, family). Needed only for those columns.
  Used whenever it is already present
  (:func:`setup.download_repeatmasker_bed`).

UCSC's database files are freely usable for any purpose; cite the
UCSC Genome Browser (Kent et al. 2002, *Genome Research* 12:996) and
RepeatMasker (Smit, Hubley and Green, RepeatMasker Open-4.0).
"""

import hashlib
import logging
import lzma
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DISTRIBUTED_NAME = "rmsk_hg38_merged.bed.xz"
DISTRIBUTED_URL = (
    "https://github.com/alfaromurillo/sigmutsel/releases/download/"
    "data-repeatmasker/" + DISTRIBUTED_NAME
)
DISTRIBUTED_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_distributed_table(
    dest, force=False, url=DISTRIBUTED_URL
):
    """Download the distributed table to ``dest`` and check its SHA-256.

    Returns
    -------
    pathlib.Path
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    logger.info("Downloading the RepeatMasker intervals from %s", url)
    urllib.request.urlretrieve(url, tmp)
    digest = _sha256(tmp)
    if digest != DISTRIBUTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {url}: got {digest}, expected "
            f"{DISTRIBUTED_SHA256}."
        )
    tmp.replace(dest)
    return dest


def write_distributed_table(full_path, out_path):
    """Cut the distributed table out of the full rmsk dump (for maintainers).

    Writes the merged intervals of :func:`qc.load_repeat_intervals`,
    sequences in sorted order, xz-compressed. Deterministic: the same
    full table always gives the same file and checksum.

    Returns
    -------
    pathlib.Path
    """
    from .qc import load_repeat_intervals

    intervals = load_repeat_intervals(full_path)
    out_path = Path(out_path)
    n = 0
    with lzma.open(out_path, "wb", preset=9) as fout:
        for chrom in sorted(intervals):
            starts, ends = intervals[chrom]
            fout.write(
                "".join(
                    f"{chrom}\t{s}\t{e}\n"
                    for s, e in zip(starts, ends)
                ).encode()
            )
            n += len(starts)
    logger.info("Distributed table: %d intervals -> %s", n, out_path)
    return out_path
