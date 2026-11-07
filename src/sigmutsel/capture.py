"""Capture territory: how observable each site is under a sequencing assay.

The territory of :mod:`channel_universe` says which sites count at all.
Within it, a hybrid-capture exome does not see every site equally: the
sites outside the capture design's targets are seen only through
off-target reads (or not at all, when the caller drops off-target
calls), and capture efficiency falls at extreme GC. This module gives
every coding site and essential splice site a relative observability

    T_o = g(GC_e(o)) * [1(o in targets) + (1 - 1(o in targets)) s_out]

where ``GC_e(o)`` is the GC fraction of the coding exon holding the
site (a splice site takes its anchor exon's), ``g`` a capture-GC curve
given as a table of ``log g`` over GC, and ``s_out`` the relative
sensitivity outside the targets. ``T`` multiplies the site weights
(:mod:`site_weights`) wherever they apply -- the opportunity tables,
each variant's route weights, site rates -- so the rate of a site is
``mu_tau upsilon_o T_o / D_tau(upsilon T)``. Its overall scale cancels
in ``D_tau``; only relative values matter.

The targets are a BED on hg19 (as most capture designs of the TCGA era
are published); positions are lifted GRCh38 -> hg19 one at a time, as
for the territory itself (:func:`channel_universe.territory_masks`), so
no BED liftover is needed. Splice sites are intronic and are tested
against the BED padded by ``splice_padding`` bases.

How ``g`` and ``s_out`` are estimated is up to the caller: ``g`` is
best fitted within genes (each gene's calls over its exons), so that
gene-level biology correlated with GC stays in the covariate model.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CaptureTerritory:
    """One assay's capture territory (see the module docstring).

    Parameters
    ----------
    bed_path : str
        Target BED (see ``build``), at least three columns (chrom, 0-based start,
        end); chromosome names with or without ``chr``.
    outside : float
        Relative sensitivity of sites outside the targets, in [0, 1].
    gc : sequence of float, optional
        GC grid of the capture-GC curve (increasing).
    log_g : sequence of float, optional
        ``log g`` at each grid point; interpolated linearly, constant
        beyond the ends. Without a curve, ``g = 1``.
    splice_padding : int
        Bases of target padding for splice sites.
    build : {"hg19", "hg38"}
        The BED's genome build: ``"hg19"`` lifts the (GRCh38) sites
        into it, ``"hg38"`` reads them as they are.
    """

    bed_path: str
    outside: float = 0.0
    gc: tuple | None = None
    log_g: tuple | None = None
    splice_padding: int = 10
    build: str = "hg19"

    def __post_init__(self):
        if self.build not in ("hg19", "hg38"):
            raise ValueError(
                f"build must be 'hg19' or 'hg38', got {self.build!r}"
            )
        if not 0.0 <= float(self.outside) <= 1.0:
            raise ValueError(
                f"outside must be in [0, 1], got {self.outside!r}"
            )
        if (self.gc is None) != (self.log_g is None):
            raise ValueError("gc and log_g go together.")
        if self.gc is not None:
            self.gc = tuple(float(x) for x in self.gc)
            self.log_g = tuple(float(x) for x in self.log_g)
            if len(self.gc) != len(self.log_g) or len(self.gc) < 2:
                raise ValueError(
                    "gc and log_g need the same length >= 2."
                )
            if np.any(np.diff(self.gc) <= 0):
                raise ValueError("gc must be strictly increasing.")

    def to_dict(self):
        return {
            "bed_path": str(self.bed_path),
            "bed_sha1": _file_sha1(self.bed_path),
            "outside": float(self.outside),
            "gc": None if self.gc is None else list(self.gc),
            "log_g": None if self.log_g is None else list(self.log_g),
            "splice_padding": int(self.splice_padding),
            "build": self.build,
        }

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        return cls(
            bed_path=d["bed_path"],
            outside=d.get("outside", 0.0),
            gc=d.get("gc"),
            log_g=d.get("log_g"),
            splice_padding=d.get("splice_padding", 10),
            build=d.get("build", "hg19"),
        )

    def g_at(self, gc):
        """The capture-GC factor at GC fractions ``gc``."""
        gc = np.asarray(gc, dtype=float)
        if self.gc is None:
            return np.ones_like(gc)
        return np.exp(np.interp(gc, self.gc, self.log_g))

    def label(self):
        """Short content hash (BED content included), for cache paths."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


def _file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bed(path):
    import pandas as pd

    bed = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        comment="#",
        dtype={"chrom": str},
    )
    bed["chrom"] = "chr" + bed["chrom"].str.removeprefix("chr")
    return bed


def exon_index(models):
    """Exon number of every coding flat position (-1 for padding).

    An exon is a maximal run of a gene's coding positions at
    consecutive genomic coordinates.
    """
    coding = np.flatnonzero(models.gpos >= 0)
    g = models.gene_of[coding]
    gp = models.gpos[coding]
    new = np.ones(len(coding), dtype=bool)
    new[1:] = (g[1:] != g[:-1]) | (np.abs(gp[1:] - gp[:-1]) != 1)
    out = np.full(len(models.codes), -1, dtype=np.int64)
    out[coding] = np.cumsum(new) - 1
    return out


def exon_gc(models, exon=None):
    """GC fraction of each exon (over its A/C/G/T coding bases)."""
    if exon is None:
        exon = exon_index(models)
    coding = exon >= 0
    n = int(exon.max()) + 1
    base = models.codes[coding]
    gc = np.bincount(
        exon[coding], ((base == 1) | (base == 2)).astype(float), n
    )
    acgt = np.bincount(exon[coding], (base < 4).astype(float), n)
    return gc / np.maximum(acgt, 1)


def splice_flat_index(models):
    """Flat index of the coding base each splice site hangs off."""
    sp = models.splice
    g = sp["gene"].to_numpy()
    pad = np.bincount(
        models.gene_of[models.gpos < 0], minlength=len(models.offsets)
    )
    return models.offsets[g] + pad[g] + sp["anchor"].to_numpy() - 1


def capture_weights(models, capture, chain_path=None):
    """Per-site observability ``T`` under ``capture``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Float weights over ``models.codes`` (padding 0) and over
        ``models.splice`` rows. Cached next to the transcript models,
        keyed on :meth:`CaptureTerritory.label`.
    """
    from .liftover import (
        in_intervals,
        lift_positions,
        merged_bed_intervals,
        read_chain,
    )
    from .locations import location_liftover_chain_hg38_to_hg19

    cache = None
    if getattr(models, "directory", None) is not None:
        cache = (
            Path(models.directory) / f"capture_{capture.label()}.npz"
        )
        if cache.exists():
            arrays = np.load(cache)
            return arrays["coding_w"], arrays["splice_w"]

    if capture.build == "hg19":
        chain_path = Path(
            chain_path or location_liftover_chain_hg38_to_hg19
        )
        if not chain_path.exists():
            from . import setup

            setup.download_liftover_chain()
        blocks = read_chain(chain_path)

        def lift(chroms, positions):
            return lift_positions(blocks, chroms, positions)

    else:

        def lift(chroms, positions):
            chroms = np.array(
                ["chr" + str(c).removeprefix("chr") for c in chroms],
                dtype=object,
            )
            return chroms, np.asarray(positions, dtype=np.int64)

    bed = _read_bed(capture.bed_path)

    exon = exon_index(models)
    gc = exon_gc(models, exon)
    coding = np.flatnonzero(exon >= 0)
    chrom19, pos19 = lift(
        models.chrom_of[coding], models.gpos[coding]
    )
    inside = np.zeros(len(coding), dtype=bool)
    ok = pos19 > 0
    inside[ok] = in_intervals(
        merged_bed_intervals(bed, 0), chrom19[ok], pos19[ok]
    )
    coding_w = np.zeros(len(models.codes))
    coding_w[coding] = capture.g_at(gc[exon[coding]]) * np.where(
        inside, 1.0, float(capture.outside)
    )

    sp = models.splice
    splice_chrom = models.selection["chrom"].to_numpy()[
        sp["gene"].to_numpy()
    ]
    chrom19, pos19 = lift(splice_chrom, sp["gpos"].to_numpy())
    s_inside = np.zeros(len(sp), dtype=bool)
    ok = pos19 > 0
    s_inside[ok] = in_intervals(
        merged_bed_intervals(bed, capture.splice_padding),
        chrom19[ok],
        pos19[ok],
    )
    anchor_exon = exon[splice_flat_index(models)]
    splice_w = capture.g_at(gc[anchor_exon]) * np.where(
        s_inside, 1.0, float(capture.outside)
    )
    if cache is not None:
        tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")
        np.savez_compressed(tmp, coding_w=coding_w, splice_w=splice_w)
        os.replace(tmp, cache)
    return coding_w, splice_w
