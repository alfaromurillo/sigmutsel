"""Site weights: an opportunity that varies along a gene.

The opportunity counts, for gene ``g``, channel ``h`` and type ``tau``,
the (site, alternate base) pairs where a type-``tau`` substitution can
happen: ``n^h_{g tau} = sum_{(s, a)} w_{s,a}``. So far every weight was
0 or 1 -- the capture territory and the germline mask decide which
pairs are observable. This module lets a weight be any positive
number, a product of a few per-cohort factors of the site,

    w_{s,a} = u[distance bin of s] * v[substitution class of (s, a),
                                        strand of s],

fitted once per cohort on passenger calls. ``u`` captures detection
along the exon (calls near exon junctions, and in long exon cores in
some cohorts, sit on less read depth and are called less often);
``v`` captures transcription-strand asymmetry (the 96 types merge a
pyrimidine on the coding strand with one on the template, which
transcription-coupled repair treats differently). ``v`` is normalised
to mean 1 within each substitution class, so it carries asymmetry
only; ``u`` to an expectation-weighted mean of 1.

A site's rate is then ``mu_tau w_{s,a} / D_tau(w)`` with ``D_tau(w)`` the
weighted total, so a variant's rate is multiplied by its routes'
weights, and the per-type denominators renormalise. Nothing else in
the model changes: the weights enter through the opportunity tables
(:func:`channel_universe.compute_channel_opportunity`) and through
per-route weights of each variant (:func:`variant_route_weights`).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import canonical_types_order

DIST_EDGES = (3, 10, 20, 40, 70, 120)
DIST_LABELS = (
    "1-3",
    "4-10",
    "11-20",
    "21-40",
    "41-70",
    "71-120",
    ">120",
    "single exon",
)
CLASSES = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
CLASS_OF = np.array(
    [CLASSES.index(t[2:5]) for t in canonical_types_order],
    dtype=np.int64,
)
STRAND_LABELS = tuple(
    f"{c} {s}" for c in CLASSES for s in ("coding", "template")
)


@dataclass
class SiteWeights:
    """Per-cohort factors of the opportunity (see the module docstring).

    ``distance``: one factor per :data:`DIST_LABELS` bin. ``strand``:
    one per class x strand (:data:`STRAND_LABELS`; "coding" = the
    pyrimidine is on the coding strand).
    """

    distance: np.ndarray = field(default_factory=lambda: np.ones(8))
    strand: np.ndarray = field(default_factory=lambda: np.ones(12))
    pseudo_count: float = 50.0
    n_calls: int = 0

    def to_dict(self):
        return {
            "distance": dict(
                zip(DIST_LABELS, map(float, self.distance))
            ),
            "strand": dict(
                zip(STRAND_LABELS, map(float, self.strand))
            ),
            "pseudo_count": float(self.pseudo_count),
            "n_calls": int(self.n_calls),
        }

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        return cls(
            distance=np.array(
                [d["distance"][k] for k in DIST_LABELS]
            ),
            strand=np.array([d["strand"][k] for k in STRAND_LABELS]),
            pseudo_count=d.get("pseudo_count", 50.0),
            n_calls=d.get("n_calls", 0),
        )

    def label(self):
        """Short content hash, for cache paths."""
        import hashlib

        blob = np.concatenate([self.distance, self.strand]).round(6)
        return hashlib.sha1(blob.tobytes()).hexdigest()[:10]


def junction_distance(models):
    """Per flat position: distance along the CDS to the nearest junction.

    1 is the base next to an exon-exon junction. Returns ``(dist,
    has_junction)``; positions of genes with one coding exon get a very
    large distance and ``has_junction[gene]`` False.
    """
    gp, g_of = models.gpos, models.gene_of
    n = len(gp)
    pos = np.arange(n)
    junc = (
        (g_of[1:] == g_of[:-1])
        & (gp[1:] >= 0)
        & (gp[:-1] >= 0)
        & (np.abs(gp[1:] - gp[:-1]) != 1)
    )
    big = np.iinfo(np.int64).max
    last = np.full(n, -1, dtype=np.int64)
    last[1:] = np.where(junc, np.arange(n - 1), -1)
    last = np.maximum.accumulate(last)
    d_left = np.where(last >= models.offsets[g_of], pos - last, big)
    nxt = np.full(n, big, dtype=np.int64)
    nxt[:-1] = np.where(junc, np.arange(n - 1), big)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    ok = (nxt < n) & (g_of[np.minimum(nxt, n - 1)] == g_of)
    d_right = np.where(ok, nxt + 1 - pos, big)
    has_junction = np.zeros(len(models.selection), dtype=bool)
    has_junction[g_of[:-1][junc]] = True
    return np.minimum(d_left, d_right), has_junction


def distance_bin(dist, has_junction_of_position):
    """Bin index into :data:`DIST_LABELS`."""
    b = np.searchsorted(DIST_EDGES, dist, side="left")
    b = np.minimum(b, len(DIST_EDGES))
    return np.where(has_junction_of_position, b, len(DIST_LABELS) - 1)


def position_bins(models):
    """Distance bin and strand orientation of every flat position."""
    dist, has_junction = junction_distance(models)
    dbin = distance_bin(dist, has_junction[models.gene_of]).astype(
        np.int8
    )
    orient = np.isin(models.codes, [0, 2]).astype(
        np.int8
    )  # A/G: template
    return dbin, orient


def weights_at(dbin, orient, types, sw):
    """Weights of substitutions: bins/orientations per site, types per alt."""
    if sw is None:
        return np.ones(np.shape(types))
    cls = CLASS_OF[types]
    return (
        sw.distance[dbin][..., None]
        * sw.strand[cls * 2 + orient[..., None]]
    )


def fit_site_weights(
    dbin,
    orient,
    types,
    expected,
    call_rows,
    call_alts,
    pseudo_count=50.0,
    n_iter=30,
):
    """Fit distance and strand factors by raking two margins.

    Parameters
    ----------
    dbin, orient : (n_sites,) arrays
    types : (n_sites, 3) type indices of the three alternate bases
    expected : (n_sites, 3) expected calls per substitution before the
        weights (rate per type times observability), over the sites the
        fit may use (zero elsewhere, e.g. outside passenger genes)
    call_rows, call_alts : site row and alternate index (0..2) of each
        observed call used for the fit
    pseudo_count : expected calls added to every margin, pulling every
        factor toward 1 where the data are thin

    Returns
    -------
    SiteWeights
    """
    cls_orient = CLASS_OF[types] * 2 + orient[:, None]
    n_sites = len(dbin)
    d_rep = np.repeat(dbin, 3)
    O_d = np.bincount(dbin[call_rows], minlength=8).astype(float)
    O_s = np.bincount(
        cls_orient[call_rows, call_alts], minlength=12
    ).astype(float)
    u, v = np.ones(8), np.ones(12)
    E_s = np.zeros(12)
    for _ in range(n_iter):
        E_d = np.bincount(
            d_rep,
            weights=(expected * v[cls_orient]).ravel(),
            minlength=8,
        )
        u = (O_d + pseudo_count) / (E_d + pseudo_count)
        E_s = np.bincount(
            cls_orient.ravel(),
            weights=(expected * u[dbin][:, None]).ravel(),
            minlength=12,
        )
        v = (O_s + pseudo_count) / (E_s + pseudo_count)
        for k in range(6):
            pair = [2 * k, 2 * k + 1]
            if E_s[pair].sum() > 0:
                v[pair] /= (E_s[pair] * v[pair]).sum() / E_s[
                    pair
                ].sum()
    E_d0 = np.bincount(d_rep, weights=expected.ravel(), minlength=8)
    if E_d0.sum() > 0:
        u /= (E_d0 * u).sum() / E_d0.sum()
    del n_sites
    return SiteWeights(
        distance=u,
        strand=v,
        pseudo_count=pseudo_count,
        n_calls=len(call_rows),
    )


def variant_route_weights(db, models, coding_in, sw):
    """Per variant, the weight of each of its routes, in route order.

    Routes are regenerated with their sites
    (:func:`channel_universe._routes`, ``with_sites=True``) from the
    variant's first in-universe coding call, in the order
    :func:`channel_universe.classify_calls` wrote them, so the list
    lines up with the variant's ``mut_types``. Splice variants (one
    site, outside the coding table) get weight 1.

    Returns
    -------
    pandas.Series
        variant -> list of float
    """
    from .channel_universe import (
        _BASES,
        _CODONS,
        _COMPLEMENT_CODE,
        _routes,
    )

    dbin, orient = position_bins(models)
    calls = db[
        db["in_universe"].fillna(False).astype(bool)
        & db["channel"].isin(["syn", "mis", "non"])
    ].drop_duplicates("variant")
    gene_index = models.gene_index
    coding = np.flatnonzero(models.gpos >= 0)
    keys = (
        models.gene_of[coding].astype(np.int64) * (1 << 32)
        + models.gpos[coding]
    )
    order = np.argsort(keys, kind="stable")
    keys_sorted, coding_sorted = keys[order], coding[order]
    minus_gene = models.selection["strand"].to_numpy() == "-"
    code_of = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = {}
    for row in calls.itertuples(index=False):
        g = gene_index.get(str(row.ensembl_gene_id))
        alt = code_of.get(str(row.Tumor_Seq_Allele2))
        if g is None or alt is None:
            continue
        q = g * (1 << 32) + int(row.Start_Position)
        k = np.searchsorted(keys_sorted, q)
        if k >= len(keys_sorted) or keys_sorted[k] != q:
            continue
        i = coding_sorted[k]
        if minus_gene[g]:
            alt = int(_COMPLEMENT_CODE[alt])
        start = i - (models.local[i] % 3)
        codon = "".join(
            _BASES[c] if c < 4 else "N"
            for c in models.codes[start : start + 3]
        )
        if "N" in codon or len(codon) < 3:
            # classify_calls gives such a call no routes either; the
            # variant keeps weight 1.
            continue
        pos = models.local[i] % 3
        mutated = codon[:pos] + _BASES[alt] + codon[pos + 1 :]
        target = (
            None if row.channel == "syn" else _CODONS.get(mutated)
        )
        types, sites = _routes(
            models,
            start,
            target,
            row.channel,
            coding_in,
            with_sites=True,
        )
        t_idx = np.array(
            [canonical_types_order.index(t) for t in types],
            dtype=np.int64,
        )
        s_idx = np.array([s for s, _ in sites], dtype=np.int64)
        w = weights_at(
            dbin[s_idx], orient[s_idx], t_idx[:, None], sw
        )[:, 0]
        out[row.variant] = [float(x) for x in w]
    return pd.Series(out, dtype=object)
