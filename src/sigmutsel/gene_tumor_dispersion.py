"""Gene-tumor dispersion of mutation counts across tumors.

A gene's mutations, given how many it has, are allocated across tumors
in proportion to each tumor's rate. A Poisson rate model says that
allocation is Multinomial. Real allocations are often more dispersed:
some tumors carry more of a gene's mutations than their rate predicts.
The natural extension is a gene-tumor Gamma rate ``mu_tilde_gj``,
the model's ``mu_gj`` randomized around itself,

    mu_tilde_gj ~ Gamma(shape = phi * p_gj, rate = phi / M_g),

with ``M_g = sum_j mu_gj`` and ``p_gj = mu_gj / M_g``. Conditional on
the gene's total this is a Dirichlet-Multinomial allocation with
concentration ``phi * p_gj``, so ``phi`` can be estimated from the
allocation alone, without refitting the rate model; and as
``phi -> inf`` it reduces to the Multinomial. The same ``phi`` then
enters selection estimation through
:func:`.estimate_gammas.estimate_gamma_from_mus`'s ``gene_tumor_dispersion``.

``phi`` is fitted as a trend in the gene's mutation count ``N_g``
rather than as one number. In practice ``phi`` rises with ``N_g`` --
genes with many mutations are relatively less overdispersed per gene-tumor pair
-- so a single pooled value overstates dispersion for exactly the
high-count genes whose selection is usually of interest.
:func:`fit_gene_tumor_dispersion_trend` fits ``phi`` within count strata and
then ``log phi = a + b log N_g`` across them, with two guards that
matter most for small datasets, where a few genes decide a stratum:

* the slope is floored at zero -- a falling ``phi`` extrapolated to
  the highest-count genes gives them a tiny ``phi`` and an
  unidentifiable selection coefficient;
* the fitted trend must raise the allocation likelihood of held-out
  genes over the Multinomial, or dispersion is switched off
  (``phi = inf``) and selection estimation is exactly the
  dispersion-free model.

So the same procedure applies to every dataset, and the data decide
how much dispersion, if any, enters.

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
    phi : float or ndarray, shape (genes,)
        Concentration, one value or one per gene. ``inf`` gives the
        Multinomial.

    Returns
    -------
    ndarray, shape (genes,)
    """
    counts = np.asarray(counts, dtype=float)
    log_weights = np.broadcast_to(
        np.asarray(log_weights, dtype=float), counts.shape
    )
    phi = np.broadcast_to(
        np.asarray(phi, dtype=float), (counts.shape[0],)
    )
    totals = counts.sum(axis=1)
    p = np.exp(
        log_weights
        - spc.logsumexp(log_weights, axis=1, keepdims=True)
    )
    head = spc.gammaln(totals + 1.0) - spc.gammaln(counts + 1.0).sum(
        axis=1
    )
    out = np.empty(counts.shape[0])
    inf = ~np.isfinite(phi)
    if inf.any():
        hit = counts[inf] > 0
        contrib = np.zeros_like(counts[inf])
        contrib[hit] = counts[inf][hit] * np.log(p[inf][hit])
        out[inf] = head[inf] + contrib.sum(axis=1)
    fin = ~inf
    if fin.any():
        n_int = np.rint(counts[fin]).astype(np.int64)
        out[fin] = (
            head[fin]
            - _log_rising(
                phi[fin], np.rint(totals[fin]).astype(np.int64)
            )
            + _log_rising(phi[fin][:, None] * p[fin], n_int).sum(
                axis=1
            )
        )
    return out


def fit_phi(counts, log_weights, bounds=(1.0, 1e9)):
    """Maximum-likelihood ``phi`` for a set of genes' allocations.

    Returns
    -------
    (phi, at_ceiling) : (float, bool)
        ``at_ceiling`` means the data want no extra dispersion. The MLE
        of a dispersion parameter is a boundary problem: under a true
        Multinomial a large finite estimate is common, so a single
        estimate should not be over-read -- which is why
        :func:`fit_gene_tumor_dispersion_trend` also checks the fit on
        held-out genes.
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


def _fit_trend(counts, log_b, n_g, strata, min_genes):
    """Strata, pooled phi and the (slope-floored) trend on one gene set."""
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
    slope_floored = False
    if len(usable) >= 3:
        x = np.log([r["median_n"] for r in usable])
        y = np.log([r["phi"] for r in usable])
        slope, intercept = np.polyfit(x, y, 1)
        if slope < 0:
            # phi falling with a gene's count is what sparse high-count
            # strata produce by chance; extrapolated to the largest
            # genes it hands them a tiny phi and an unidentifiable
            # gamma. Floor the slope and refit the level.
            slope, intercept, slope_floored = (
                0.0,
                float(np.mean(y)),
                True,
            )
        method = "trend"
    else:
        slope, intercept, method = (
            0.0,
            float(np.log(pooled)),
            "pooled",
        )
        if pooled_ceiling:
            method = "none"
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "slope_floored": slope_floored,
        "pooled_phi": pooled,
        "pooled_at_ceiling": bool(pooled_ceiling),
        "method": method,
        "strata": rows,
    }


def fit_gene_tumor_dispersion_trend(
    counts,
    baseline,
    strata=DEFAULT_STRATA,
    min_genes=15,
    folds=5,
    seed=0,
):
    """Fit ``log phi = intercept + slope * log N_g``, if the data support it.

    ``intercept`` and ``slope`` are written ``a`` and ``b`` in the
    write-up, as ``log phi_g = a + b log N_g``; the names differ, the
    quantities do not.

    Parameters
    ----------
    counts : ndarray, shape (genes, tumors)
        Observed counts, typically non-silent mutations in passenger
        genes (drivers' allocation reflects selection, not rate).
    baseline : ndarray, shape (genes, tumors)
        The rate model's per-tumor rates for the same gene-tumor pairs.
    strata : sequence of (low, high)
        Half-open ranges of ``N_g``; genes with ``N_g < 2`` carry no
        information about dispersion and are never used.
    min_genes : int
        Strata with fewer genes are skipped.
    folds : int
        Genes are split into this many folds; the trend fitted on the
        others is scored on each. If dispersion does not raise the
        held-out allocation likelihood over the Multinomial, the
        result is ``method = "none"`` and :func:`phi_for_count`
        returns ``inf`` -- the dispersion-free model. ``0`` skips the
        check.
    seed : int
        Fold assignment seed.

    Returns
    -------
    dict
        ``intercept``, ``slope``, ``slope_floored``, ``pooled_phi``,
        ``pooled_at_ceiling``, ``method`` ("trend", "pooled" or
        "none"), ``strata`` (per-stratum ``low``, ``high``,
        ``median_n``, ``n_genes``, ``phi``, ``at_ceiling``), and
        ``cv_gain`` / ``cv_gain_per_mutation`` (held-out nats over the
        Multinomial; ``None`` when ``folds == 0``).

        The trend needs three strata whose ``phi`` is not at the search
        ceiling; otherwise the pooled ``phi`` is used. A negative slope
        is floored at zero (``slope_floored``).
    """
    counts = np.asarray(counts, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n_g = counts.sum(axis=1)
    keep = (baseline.sum(axis=1) > 0) & (n_g >= 2)
    counts, baseline, n_g = counts[keep], baseline[keep], n_g[keep]
    with np.errstate(divide="ignore"):
        log_b = np.log(baseline)
    result = _fit_trend(counts, log_b, n_g, strata, min_genes)
    result["cv_gain"] = None
    result["cv_gain_per_mutation"] = None
    if folds and result["method"] != "none" and len(n_g) >= folds:
        order = np.random.default_rng(seed).permutation(len(n_g))
        gain = 0.0
        for f in range(folds):
            test = np.zeros(len(n_g), dtype=bool)
            test[order[f::folds]] = True
            train = _fit_trend(
                counts[~test],
                log_b[~test],
                n_g[~test],
                strata,
                min_genes,
            )
            phis = np.array(
                [phi_for_count(train, n) for n in n_g[test]]
            )
            gain += float(
                (
                    dirichlet_multinomial_logpmf(
                        counts[test], log_b[test], phis
                    )
                    - dirichlet_multinomial_logpmf(
                        counts[test], log_b[test], np.inf
                    )
                ).sum()
            )
        result["cv_gain"] = gain
        result["cv_gain_per_mutation"] = gain / float(n_g.sum())
        if gain <= 0:
            result["method"] = "none"
    return result


def phi_for_count(trend, n):
    """``phi`` for a gene with ``n`` mutations; ``inf`` means none."""
    if trend.get("method") == "none":
        return np.inf
    return float(
        np.exp(
            trend["intercept"] + trend["slope"] * np.log(max(n, 2.0))
        )
    )
