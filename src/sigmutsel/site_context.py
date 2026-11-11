"""Pentanucleotide context factors, shared across cohorts by process.

The trinucleotide type ``tau`` of a substitution leaves its outer bases
out: ``c = 4 * left2 + right2``, the bases two to the left and right on
the pyrimidine strand (:func:`site_weights.position_context5`; 16 when
unknown). A *context factor* multiplies a site's rate by a function of
``(tau, c)``. It belongs to the mutational process, not the cohort, so
it is built from per-process **flank factors** ``F_k(tau, c)``, one set
per signature group ``k`` (:class:`ContextFactors`, fitted elsewhere on
passenger calls pooled over cohorts), mixed by each signature group's
share of type ``tau``:

    upsilon^ctx(tau, c) = sum_k pi_{tau k} psi_k(tau, c),
    psi_k(tau, c)       = F_k(tau, c) / Fbar_k(tau),

``Fbar_k(tau)`` the mean of ``F_k(tau, .)`` over a reference territory
(so each group's factors average 1 per type there), and ``pi_{tau k}``
the share of type-``tau`` mutations that group ``k``'s signatures emit
(exposure x emission). :func:`context_factor` is that sum; everything
else here calls it. A signature in no group of the factors (and the
contexts with no readable pentamer) takes factor 1.

Two modes of the same factor:

``"opportunity"``
    One table per cohort, the mix ``pi`` pooled over the cohort's
    tumors (:func:`cohort_context_table`), renormalised to a mean of 1
    per type over the territory, put on the opportunity as a factor of
    the site weight (``SiteWeights.context``, applied by
    :func:`site_weights.weights_at`). It changes how a gene's
    opportunity is counted, so gene rates move and a model must be
    refitted.

``"placement"``
    Per tumor, the tumor's own mix, renormalised within each (gene,
    channel, type) group of sites (:func:`placement_multipliers`), so
    a group's total rate in every tumor is unchanged: gene and channel
    rates, ``delta``, ``theta``, ``r_g`` and gene gammas do not move,
    only where in its group a call falls -- the rates of variants.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import canonical_types_order

N_CONTEXTS = 16  # 4 * left2 + right2
NO_CONTEXT = 16  # no readable pentamer: factor 1
MODES = ("placement", "opportunity")


@dataclass
class ContextFactors:
    """Per-process flank factors ``F_k(tau, c)``.

    ``groups``: the K signature-group names. ``log_flank``: ``(K, 96,
    16)``, ``log F_k(tau, c)`` (any per-type scale; it is normalised
    away). ``signature_groups``: signature -> group, for signatures
    whose group is not their own name; a signature named after no
    group, and not mapped, takes factor 1.
    """

    groups: list
    log_flank: np.ndarray
    signature_groups: dict = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        self.groups = [str(g) for g in self.groups]
        self.log_flank = np.asarray(self.log_flank, dtype=float)
        if self.log_flank.shape != (len(self.groups), 96, N_CONTEXTS):
            raise ValueError(
                f"log_flank must be (K, 96, {N_CONTEXTS}) with K = "
                f"{len(self.groups)}, got {self.log_flank.shape}."
            )

    def group_of(self, signature):
        """The group index of a signature, or -1 (factor 1)."""
        g = self.signature_groups.get(signature, signature)
        return self.groups.index(g) if g in self.groups else -1

    def membership(self, signatures):
        """``(S, K + 1)`` 0/1; the last column is the unit group."""
        out = np.zeros((len(signatures), len(self.groups) + 1))
        for i, s in enumerate(signatures):
            out[i, self.group_of(str(s))] = 1.0
        return out

    def to_dict(self):
        return {
            "groups": list(self.groups),
            "log_flank": np.round(self.log_flank, 12).tolist(),
            "signature_groups": dict(self.signature_groups),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            groups=d["groups"],
            log_flank=np.array(d["log_flank"], dtype=float),
            signature_groups=dict(d.get("signature_groups", {})),
            label=d.get("label", ""),
        )


def context_table_at(table, types, c):
    """``table[types, c]``, 1 where ``c`` is :data:`NO_CONTEXT`.

    ``types`` and ``c`` broadcast against each other.
    """
    table = np.asarray(table, dtype=float)
    types = np.asarray(types, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    has = c < N_CONTEXTS
    return np.where(
        has, table[types, np.minimum(c, N_CONTEXTS - 1)], 1.0
    )


def context_opportunity(types, weights, c):
    """Summed weights by type and outer bases, ``(96, 17)``.

    ``types``/``weights`` per (site, alternate base), ``c`` per site
    (or broadcastable to ``types``); column 16 holds the sites with no
    readable pentamer.
    """
    types = np.asarray(types, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if c.ndim == types.ndim - 1:
        c = c[..., None]
    c = np.broadcast_to(c, types.shape)
    w = np.broadcast_to(np.asarray(weights, dtype=float), types.shape)
    flat = types.ravel() * (N_CONTEXTS + 1) + np.minimum(
        c.ravel(), 16
    )
    return np.bincount(
        flat, weights=w.ravel(), minlength=96 * (N_CONTEXTS + 1)
    ).reshape(96, N_CONTEXTS + 1)


def normalised_flank(factors, territory):
    """``psi_k(tau, c)``: each group's factors over their territory mean.

    ``territory`` is ``(96, 16)`` or ``(96, 17)`` summed weights by type
    and outer bases (:func:`context_opportunity`; column 16 ignored).
    Returns ``(K + 1, 96, 16)``, the last group (the unit group) 1.
    """
    W = np.asarray(territory, dtype=float)[:, :N_CONTEXTS]
    F = np.exp(factors.log_flank)
    tot = W.sum(axis=1)
    mean = np.einsum("tc,ktc->kt", W, F) / np.where(tot > 0, tot, 1)
    psi = F / np.where(mean > 0, mean, 1)[..., None]
    return np.concatenate([psi, np.ones((1, 96, N_CONTEXTS))])


def group_mix(factors, exposures, emissions, pooled=False):
    """Each signature group's share of each type, per tumor or pooled.

    ``exposures``: tumors x signatures (mutation counts). ``emissions``:
    the 96 canonical types x signatures. Returns ``(J, 96, K + 1)``
    shares, or with ``pooled`` the ``(96, K + 1)`` shares of the
    cohort's expected calls. A tumor (or type) with nothing to share
    gets the unit group.
    """
    sigs = list(exposures.columns)
    a = exposures.to_numpy(dtype=float)
    e = (
        emissions.reindex(index=canonical_types_order, columns=sigs)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    G = factors.membership(sigs)
    if pooled:
        by = np.einsum("js,ts,sk->tk", a, e, G)
    else:
        by = np.einsum("js,ts,sk->jtk", a, e, G)
    tot = by.sum(axis=-1, keepdims=True)
    unit = np.zeros(by.shape[-1])
    unit[-1] = 1.0
    return np.where(tot > 0, by / np.where(tot > 0, tot, 1), unit)


def context_factor(psi, mix, types, c):
    """The context factor ``sum_k mix_{tau k} psi_k(tau, c)``.

    The one function every mode and every rate path uses.
    ``psi``: ``(K + 1, 96, 16)`` (:func:`normalised_flank`). ``mix``:
    ``(..., 96, K + 1)`` (:func:`group_mix`). ``types`` and ``c``
    broadcast against each other (shape ``P``); the result is
    ``mix.shape[:-2] + P``, 1 where ``c`` is :data:`NO_CONTEXT`.
    """
    types = np.asarray(types, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    types, c = np.broadcast_arrays(types, c)
    has = c < N_CONTEXTS
    cc = np.minimum(c, N_CONTEXTS - 1)
    lead = np.shape(mix)[:-2]
    m = np.asarray(mix, dtype=float)[
        ..., types, :
    ]  # lead + P + (K+1,)
    p = np.moveaxis(psi[:, types, cc], 0, -1)  # P + (K+1,)
    out = np.einsum("...k,...k->...", m, np.broadcast_to(p, m.shape))
    return np.where(np.broadcast_to(has, lead + has.shape), out, 1.0)


def cohort_context_table(factors, exposures, emissions, territory):
    """The opportunity mode: one ``(96, 16)`` table for a cohort.

    The context factor under the cohort's pooled signature mix,
    renormalised so its mean over ``territory`` (summed site weights by
    type and outer bases, :func:`context_opportunity`) is 1 per type:
    type opportunities, and so the type rates' denominators, are kept.
    Put it on ``SiteWeights.context``.
    """
    psi = normalised_flank(factors, territory)
    mix = group_mix(factors, exposures, emissions, pooled=True)
    tt, cc = np.meshgrid(
        np.arange(96), np.arange(N_CONTEXTS), indexing="ij"
    )
    table = context_factor(psi, mix, tt, cc)  # (96, 16)
    W = np.asarray(territory, dtype=float)[:, :N_CONTEXTS]
    tot = W.sum(axis=1)
    mean = (W * table).sum(axis=1) / np.where(tot > 0, tot, 1)
    return table / np.where(mean > 0, mean, 1)[:, None]


def placement_multipliers(psi, mix, types, c, group_opportunity):
    """The placement mode: per route and tumor, ``(R, J)``.

    ``m_{rj} = F_j(tau_r, c_r) A_r / sum_c W_{rc} F_j(tau_r, c)``,
    ``F_j`` the context factor under tumor ``j``'s mix (``mix``, ``(J,
    96, K + 1)``), ``W_r`` the route's group opportunity by outer bases
    (``(R, 17)``: its gene's sites of its channel and type, the
    weights its rate is spread over; column 16 those without a
    pentamer) and ``A_r`` its total. Summed over a group's sites with
    their weights, ``m`` gives the group's total back, in every tumor.
    """
    types = np.asarray(types, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    W = np.asarray(group_opportunity, dtype=float)
    own = context_factor(psi, mix, types, c)  # (J, R)
    grid = np.broadcast_to(
        np.arange(N_CONTEXTS), (len(types), N_CONTEXTS)
    )
    every = context_factor(
        psi, mix, types[:, None], grid
    )  # (J, R, 16)
    norm = (
        np.einsum("jrc,rc->jr", every, W[:, :N_CONTEXTS]) + W[:, 16]
    )
    A = W.sum(axis=1)
    out = own * A[None, :] / np.where(norm > 0, norm, 1.0)
    return np.where(norm > 0, out, 1.0).T


def as_factors(obj):
    """A :class:`ContextFactors` from itself, a dict or a JSON path."""
    if obj is None or isinstance(obj, ContextFactors):
        return obj
    if isinstance(obj, dict):
        return ContextFactors.from_dict(obj)
    import json
    from pathlib import Path

    return ContextFactors.from_dict(json.loads(Path(obj).read_text()))


__all__ = [
    "MODES",
    "NO_CONTEXT",
    "ContextFactors",
    "as_factors",
    "cohort_context_table",
    "context_factor",
    "context_opportunity",
    "context_table_at",
    "group_mix",
    "normalised_flank",
    "placement_multipliers",
]
