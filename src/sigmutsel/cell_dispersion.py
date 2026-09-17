"""Per-cell dispersion of mutation counts across tumors.

A gene's mutations, given how many it has, are allocated across tumors
in proportion to each tumor's rate. A Poisson rate model says that
allocation is Multinomial. Real allocations are often more dispersed:
some tumors carry more of a gene's mutations than their rate predicts.
The natural extension is a per-cell Gamma rate,

    lambda_gj ~ Gamma(shape = phi * p_gj, rate = phi / M_g),

with ``M_g = sum_j mu_gj`` and ``p_gj = mu_gj / M_g``. Conditional on
the gene's total this is a Dirichlet-Multinomial allocation with
concentration ``phi * p_gj``, so ``phi`` can be estimated from the
allocation alone, without refitting the rate model; and as
``phi -> inf`` it reduces to the Multinomial. The same ``phi`` then
enters selection estimation through
:func:`.estimate_gammas.estimate_gamma_from_mus`'s ``cell_dispersion``.

``phi`` is fitted as a trend in the gene's mutation count ``N_g``
rather than as one number. In practice ``phi`` rises with ``N_g`` --
genes with many mutations are relatively less overdispersed per cell
-- so a single pooled value overstates dispersion for exactly the
high-count genes whose selection is usually of interest.
:func:`fit_cell_dispersion_trend` fits ``phi`` within count strata and
then ``log phi = a + b log N_g`` across them; a dataset with little
dispersion gets a large ``phi`` everywhere, which leaves selection
estimates essentially unchanged.

Numerical note: every ``lgamma(a + n) - lgamma(a)`` below is written
as ``sum_{k<n} log(a + k)``. The ``lgamma`` difference cancels
catastrophically once ``a`` is large, which is exactly the
near-Multinomial regime.
"""

import numpy as np
from scipy import optimize
from scipy import special as spc

DEFAULT_STRATA = (
    (2, 3),
    (3, 5),
    (5, 10),
    (10, 20),
    (20, 40),
    (40, np.inf),
)


def _log_rising(a, n):
    """Elementwise ``sum_{k=0}^{n-1} log(a + k)`` for integer ``n``."""
    n = np.asarray(n)
    a = np.broadcast_to(np.asarray(a, dtype=float), np.shape(n))
    out = np.zeros(a.shape, dtype=float)
    kmax = int(n.max()) if n.size else 0
    for k in range(kmax):
        m = n > k
        out[m] += np.log(a[m] + k)
    return out


def dirichlet_multinomial_logpmf(counts, log_weights, phi):
    """Per-gene Dirichlet-Multinomial log-likelihood of an allocation.

    Parameters
    ----------
    counts : ndarray, shape (genes, tumors)
        Observed counts.
    log_weights : ndarray, shape (genes, tumors)
        Log rates up to a per-gene constant (normalised per row).
    phi : float
        Concentration. ``inf`` gives the Multinomial.

    Returns
    -------
    ndarray, shape (genes,)
    """
    counts = np.asarray(counts, dtype=float)
    log_weights = np.broadcast_to(
        np.asarray(log_weights, dtype=float), counts.shape
    )
    totals = counts.sum(axis=1)
    p = np.exp(
        log_weights
        - spc.logsumexp(log_weights, axis=1, keepdims=True)
    )
    head = spc.gammaln(totals + 1.0) - spc.gammaln(counts + 1.0).sum(
        axis=1
    )
    if not np.isfinite(phi):
        hit = counts > 0
        contrib = np.zeros_like(counts)
        contrib[hit] = counts[hit] * np.log(p[hit])
        return head + contrib.sum(axis=1)
    n_int = np.rint(counts).astype(np.int64)
    return (
        head
        - _log_rising(
            np.full(totals.shape, float(phi)),
            np.rint(totals).astype(np.int64),
        )
        + _log_rising(phi * p, n_int).sum(axis=1)
    )


def fit_phi(counts, log_weights, bounds=(1.0, 1e9)):
    """Maximum-likelihood ``phi`` for a set of genes' allocations.

    Returns
    -------
    (phi, at_ceiling) : (float, bool)
        ``at_ceiling`` means the data want no extra dispersion. The MLE
        of a dispersion parameter is a boundary problem: under a true
        Multinomial a large finite estimate is common, so a single
        estimate should not be over-read.
    """

    def neg(log_phi):
        return -dirichlet_multinomial_logpmf(
            counts, log_weights, float(np.exp(log_phi))
        ).sum()

    res = optimize.minimize_scalar(
        neg,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-4},
    )
    phi = float(np.exp(res.x))
    return phi, phi > 0.5 * bounds[1]


def fit_cell_dispersion_trend(
    counts, baseline, strata=DEFAULT_STRATA, min_genes=15
):
    """Fit ``log phi = intercept + slope * log N_g`` across count strata.

    Parameters
    ----------
    counts : ndarray, shape (genes, tumors)
        Observed counts, typically non-silent mutations in passenger
        genes (drivers' allocation reflects selection, not rate).
    baseline : ndarray, shape (genes, tumors)
        The rate model's per-tumor rates for the same cells.
    strata : sequence of (low, high)
        Half-open ranges of ``N_g``; genes with ``N_g < 2`` carry no
        information about dispersion and are never used.
    min_genes : int
        Strata with fewer genes are skipped.

    Returns
    -------
    dict
        ``intercept``, ``slope``, ``pooled_phi``, ``pooled_at_ceiling``,
        ``method`` ("trend" or "pooled"), and ``strata`` -- a list of
        ``{"low", "high", "median_n", "n_genes", "phi", "at_ceiling"}``.
        Strata whose ``phi`` sits at the search ceiling are reported
        but excluded from the trend. With fewer than three usable
        strata the trend falls back to the pooled ``phi`` (slope 0).
    """
    counts = np.asarray(counts, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n_g = counts.sum(axis=1)
    keep = (baseline.sum(axis=1) > 0) & (n_g >= 2)
    counts, baseline, n_g = counts[keep], baseline[keep], n_g[keep]
    with np.errstate(divide="ignore"):
        log_b = np.log(baseline)
    pooled, pooled_ceiling = fit_phi(counts, log_b)

    rows = []
    for low, high in strata:
        sel = (n_g >= low) & (n_g < high)
        if sel.sum() < min_genes:
            continue
        phi, ceiling = fit_phi(counts[sel], log_b[sel])
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "median_n": float(np.median(n_g[sel])),
                "n_genes": int(sel.sum()),
                "phi": phi,
                "at_ceiling": bool(ceiling),
            }
        )
    usable = [r for r in rows if not r["at_ceiling"]]
    if len(usable) >= 3:
        slope, intercept = np.polyfit(
            np.log([r["median_n"] for r in usable]),
            np.log([r["phi"] for r in usable]),
            1,
        )
        method = "trend"
    else:
        slope, intercept, method = (
            0.0,
            float(np.log(pooled)),
            "pooled",
        )
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "pooled_phi": pooled,
        "pooled_at_ceiling": bool(pooled_ceiling),
        "method": method,
        "strata": rows,
    }


def phi_for_count(trend, n):
    """``phi`` for a gene with ``n`` mutations under a fitted trend."""
    return float(
        np.exp(
            trend["intercept"] + trend["slope"] * np.log(max(n, 2.0))
        )
    )
