"""Gene-level selection: an exact null test, and an empirical-Bayes gamma.

Two questions, answered by two tools.

**Is the gene under positive selection?** The number of tumors carrying
a non-silent mutation in the gene is tested against its distribution
at ``gamma = 1``: a Poisson-binomial over the tumors, with the gene's
rate correction ``r_g`` integrated over its silent-only posterior by
quadrature (:func:`r_posterior_nodes`), and the upper mid-p
(:func:`upper_mid_p`). Benjamini-Hochberg over the genes tested
(:func:`benjamini_hochberg`) turns the p-values into calls. Quadrature
rather than Monte Carlo draws, because at the p-values that name a
gene the tail is set by the largest ``r_g`` values; the mid-p rather
than a randomised p-value, because a call must not depend on a seed.

**How strong is the selection?** Under a flat prior, a gene's
posterior of gamma is honest about that gene alone, but read across
thousands of genes the flat intervals call many neutral genes "above
one". An empirical-Bayes prior on ``u = log gamma``, fitted on all the
cohort's genes by marginal likelihood (Efron's g-modeling), shrinks
the uninformative genes toward the bulk and leaves the well-measured
ones where their data put them. Its form here is a spike plus a slab:

    g(u) = pi0 N(u; m0, s0^2) + (1 - pi0) exp(Q a) / sum(exp(Q a))

on a grid, the spike a discretised normal, the slab a penalised
B-spline density (``Q`` a cubic B-spline basis of ``df`` columns,
centred and scaled; penalty ``c0 |a|^2``). The slab can be pooled
over several cohorts, each keeping its own spike
(:func:`fit_pooled_spike_slab`).

**The guard.** A spike fitted narrower than the likelihood can
resolve can sit just off ``gamma = 1`` with all its mass on one side;
an uninformative gene's posterior is then the prior, and a weakly
informed one's nearly the spike, so every such gene's interval
excludes one. When the fitted prior's own central interval, or its
spike's, excludes ``gamma = 1``, the spike is refitted with its
centre at one (:func:`guard_triggers`). Where both cover one, the
guard changes nothing.

The likelihood: a gene's rate in tumor ``j`` on stage-1 draw ``d`` is
``exp(sigma_d) b_j`` (a per-gene, per-draw scale times a fixed
per-tumor baseline), so the log-likelihood of ``u`` on every draw is
one function of ``u + sigma_d``, tabulated once per gene
(:class:`FactoredGeneLikelihood`). The posterior is the exact cut: the
average over draws of each draw's normalised conditional posterior
(:func:`cut_posterior`).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy import stats as sps
from scipy.interpolate import BSpline
from scipy.special import expit, logsumexp

# The grid of u = log gamma. The flat prior's support reaches gamma =
# 1e6; the empirical-Bayes prior lives on [e^-4, e^12].
GRID_STEP = 0.02
FLAT_LOG_GAMMA_RANGE = (-8.0, float(np.log(1e6)))
EB_LOG_GAMMA_RANGE = (-4.0, 12.0)

SLAB_DF = 12
SLAB_PENALTY = 0.01
SPIKE_CENTRE_BOUNDS = (-0.5, 0.5)
SPIKE_SD_BOUNDS = (0.02, 0.5)
SPIKE_WEIGHT_LOGIT_BOUNDS = (-6.0, 8.0)
SLAB_COEF_BOUNDS = (-30.0, 30.0)
GUARD_PROB = 0.89

# Quadrature nodes for r_g in the null test.
TEST_NODES = 256

# Tabulation of the summed log P(absent) in v = log(-l).
_TAB_V = np.arange(-14.0, 32.0 + 1e-9, GRID_STEP)


def log_gamma_grid(lo, hi, step=GRID_STEP):
    """Grid of ``u = log gamma`` from ``lo`` to ``hi``, both included."""
    return np.round(np.arange(lo, hi + 1e-9, step), 6)


def flat_grid():
    """The grid the likelihood is computed on (the flat prior's)."""
    return log_gamma_grid(*FLAT_LOG_GAMMA_RANGE)


def eb_grid():
    """The empirical-Bayes prior's grid, a sub-grid of :func:`flat_grid`."""
    return log_gamma_grid(*EB_LOG_GAMMA_RANGE)


def eb_mask(grid=None):
    """Points of ``grid`` (default :func:`flat_grid`) inside the EB range."""
    grid = flat_grid() if grid is None else grid
    lo, hi = EB_LOG_GAMMA_RANGE
    return (grid >= lo - 1e-9) & (grid <= hi + 1e-9)


# ---- the null test ------------------------------------------------------
def r_posterior_nodes(shape, rate, n_nodes=TEST_NODES):
    """Quadrature nodes and weights for a Gamma(shape, rate) ``r_g``.

    An even grid in log r between the 1e-10 and 1 - 1e-10 quantiles,
    weighted by the density times r (the Jacobian). Deterministic, so a
    p-value does not move with a seed.
    """
    lo, hi = sps.gamma.ppf(
        [1e-10, 1 - 1e-10], shape, scale=1.0 / rate
    )
    r = np.exp(np.linspace(np.log(lo), np.log(hi), n_nodes))
    w = sps.gamma.pdf(r, shape, scale=1.0 / rate) * r
    return r, w / w.sum()


def log_absent_null(mu, phi=None, shares=None):
    """``log P(absent)`` per tumor at gamma = 1, as the null test uses it.

    ``mu`` has tumors on its last axis. ``phi`` None or infinite is the
    Poisson link, ``-mu``; otherwise the gene-tumor dispersed link
    ``-phi p_j log(1 + M / phi)``, with ``M`` the gene's total rate and
    ``p_j`` its share in tumor ``j`` (``shares``). No absence floor:
    the test scores the link itself.
    """
    mu = np.asarray(mu, float)
    if phi is None or not np.isfinite(phi):
        return -mu
    m_total = mu.sum(axis=-1, keepdims=True)
    return -phi * np.asarray(shares, float) * np.log1p(m_total / phi)


def poisson_binomial_tail(log_absent, n, weights=None):
    """``P(N < n)`` and ``P(N = n)``, mixed over the rows of ``log_absent``.

    ``log_absent`` is ``(nodes, tumors)``: per node (a value of
    ``r_g``) each tumor's log probability of carrying nothing; ``N`` is
    the number of tumors that carry something. ``weights`` are the
    nodes' weights (equal if None).
    """
    la = np.atleast_2d(np.asarray(log_absent, float))
    w = (
        np.full(la.shape[0], 1.0 / la.shape[0])
        if weights is None
        else np.asarray(weights, float)
    )
    n = int(n)
    if n == 0:
        return 0.0, float(np.exp(la.sum(axis=1)) @ w)
    a, q = np.exp(la), -np.expm1(la)
    f = np.zeros((la.shape[0], n + 1))
    f[:, 0] = 1.0
    for j in range(la.shape[1]):
        f[:, 1:] = (
            f[:, 1:] * a[:, j : j + 1] + f[:, :-1] * q[:, j : j + 1]
        )
        f[:, 0] *= a[:, j]
    return float(f[:, :n].sum(axis=1) @ w), float(f[:, n] @ w)


def upper_mid_p(log_absent, n, weights=None):
    """Upper mid-p of ``n`` carriers: ``P(N > n) + P(N = n) / 2``.

    See :func:`poisson_binomial_tail` for the arguments.
    """
    p_lt, p_eq = poisson_binomial_tail(log_absent, n, weights)
    return max(0.0, 1.0 - p_lt - 0.5 * p_eq)


def benjamini_hochberg(p):
    """Benjamini-Hochberg q-values; NaN p-values stay NaN, untested."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


# ---- the likelihood on the grid ----------------------------------------
def _log1mexp(x):
    """``log(1 - exp(x))`` for ``x < 0``."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            x > -0.693, np.log(-np.expm1(x)), np.log1p(-np.exp(x))
        )


class FactoredGeneLikelihood:
    """One gene's presence likelihood of ``u = log gamma``, on every draw.

    The gene's rate in tumor ``j`` on draw ``d`` is ``exp(sigma_d)
    b_j``, with ``B = sum_j b_j`` and shares ``p_j = b_j / B`` over the
    tumors in the fit. With a scalar gene-tumor dispersion ``phi`` (or
    none), the unfloored log probability that tumor ``j`` carries
    nothing is ``p_j L(t)``, ``t = u + sigma_d``, with

        L(t) = -phi log(1 + e^t B / phi)    (dispersed)
        L(t) = -e^t B                       (Poisson),

    and the absence floor ``f`` makes it ``h(z) = log(f + (1 - f)
    e^z)``. The log-likelihood is ``sum_absent h(p_j L) + sum_present
    log(1 - e^{h(p_j L)})``; the sum over every tumor of ``h(p_j l)``
    is tabulated once in ``log(-l)``, the present tumors added on, so
    every draw reads one table at ``u + sigma_d``.

    Parameters
    ----------
    shares : array (J,)
        ``p_j`` over the tumors in the fit (present and absent).
    total : float
        ``B``.
    present : array of bool (J,)
        Tumors carrying a non-silent mutation in the gene.
    log_scale : array (K,)
        ``sigma_d`` per draw.
    phi : float or None
        Scalar gene-tumor dispersion; None or inf for Poisson.
    absence_floor : float
        The floor ``f`` on P(absent).
    grid : array, optional
        The ``u`` grid (default :func:`flat_grid`); evenly spaced.
    """

    def __init__(
        self,
        shares,
        total,
        present,
        log_scale,
        phi=None,
        absence_floor=1e-3,
        grid=None,
    ):
        self.grid = flat_grid() if grid is None else np.asarray(grid)
        self.step = float(self.grid[1] - self.grid[0])
        p = np.asarray(shares, float)
        present = np.asarray(present, bool)
        self.n_yes = int(present.sum())
        self.total = float(total)
        self.phi = (
            None if phi is None or not np.isfinite(phi) else phi
        )
        self.floor = float(absence_floor)
        self._logf = np.log(self.floor) if self.floor > 0 else -np.inf
        self._log1mf = np.log1p(-self.floor)
        self.log_scale = np.asarray(log_scale, float)
        tab = self._tabulate(p)
        self._table = self._build(tab, p[present])

    def _h(self, z):
        if self.floor <= 0:
            return z
        return np.logaddexp(self._logf, self._log1mf + z)

    def _tabulate(self, p):
        """``sum_j h(p_j l)`` over the tumors, on ``v = log(-l)``."""
        p = p[p > 0]
        ell = -np.exp(_TAB_V)
        if len(p) == 0:
            return np.zeros_like(ell)
        if self.floor <= 0:  # no floor: the sum is linear in l
            return p.sum() * ell
        out = (
            1 - self.floor
        ) * ell  # every p_j |l| below 1e-6: linear
        vlo, vhi = np.log(1e-6 / p.max()), np.log(60.0 / p.min())
        out[_TAB_V > vhi] = (
            len(p) * self._logf
        )  # every tumor at the floor
        idx = np.flatnonzero((_TAB_V >= vlo) & (_TAB_V <= vhi))
        for a in range(0, len(idx), 200):
            sl = idx[a : a + 200]
            out[sl] = self._h(p[None, :] * ell[sl, None]).sum(axis=1)
        return out

    def _L(self, t):
        if self.phi is not None:
            return -self.phi * np.log1p(
                np.exp(t) * self.total / self.phi
            )
        return -np.exp(t) * self.total

    def _build(self, tab, p_present):
        sig = self.log_scale
        nu = len(self.grid)
        t0 = self.grid[0] + sig.min() - self.step
        n_t = (
            int(np.ceil((sig.max() - sig.min()) / self.step)) + nu + 3
        )
        t = t0 + self.step * np.arange(n_t)
        L = self._L(t)
        with np.errstate(divide="ignore"):
            v = np.log(np.maximum(-L, 1e-300))
        g = np.where(
            v < _TAB_V[0],
            (1 - self.floor) * L,
            np.interp(v, _TAB_V, tab),
        )
        if len(p_present):
            z = self._h(p_present[:, None] * L[None, :])
            g = g + (_log1mexp(z) - z).sum(axis=0)
        pos = (self.grid[0] + sig - t0) / self.step
        o = np.floor(pos).astype(int)
        return g, o, pos - o

    def loglik(self, mask=None):
        """``(K, U)`` log-likelihood per draw on the grid (or its ``mask``)."""
        g, o, f = self._table
        cols = np.arange(len(self.grid))
        if mask is not None:
            cols = cols[mask]
        idx = o[:, None] + cols[None, :]
        return g[idx] * (1 - f[:, None]) + g[idx + 1] * f[:, None]

    def marginal(self, mask=None):
        """``log mean_d L_d(u)``: the likelihood with the draws averaged.

        What an empirical-Bayes prior is fitted on: the marginal
        likelihood of the prior multiplies it, not the flat-prior
        posterior (which would weight each draw by its evidence).
        """
        ll = self.loglik(mask)
        return logsumexp(ll, axis=0) - np.log(ll.shape[0])


# ---- posteriors --------------------------------------------------------
def cut_posterior(loglik, log_prior):
    """Posterior mass on the grid: draws' normalised posteriors, averaged.

    ``loglik`` ``(K, U)`` or ``(n, K, U)``; ``log_prior`` ``(U,)``.
    """
    lp = np.asarray(loglik) + np.asarray(log_prior)
    lp = lp - lp.max(axis=-1, keepdims=True)
    w = np.exp(lp)
    w /= w.sum(axis=-1, keepdims=True)
    return w.mean(axis=-2)


def posterior_summary(mass, grid, prob=GUARD_PROB, threshold=2.0):
    """Central interval, median, mean and P(gamma > threshold) of a mass.

    ``mass`` on the ``u`` grid ``grid``. Returns a dict with ``lower``,
    ``median``, ``upper`` (gamma scale), ``p_above`` and ``mean``.
    """
    mass = np.asarray(mass, float)
    tot = mass.sum()
    # each point holds the mass of the cell centred on it
    c = (np.cumsum(mass) - 0.5 * mass) / tot
    tail = (1 - prob) / 2
    lo, med, hi = (
        float(np.exp(np.interp(x, c, grid)))
        for x in (tail, 0.5, 1 - tail)
    )
    return {
        "lower": lo,
        "median": med,
        "upper": hi,
        "p_above": float(mass[grid > np.log(threshold)].sum() / tot),
        "mean": float((mass * np.exp(grid)).sum() / tot),
    }


# ---- the empirical-Bayes prior -----------------------------------------
def slab_basis(grid, df=SLAB_DF):
    """Cubic B-spline basis on ``grid``, columns centred and unit-norm."""
    k = 3
    inner = np.linspace(grid[0], grid[-1], df - k + 1)
    knots = np.r_[[grid[0]] * k, inner, [grid[-1]] * k]
    q = BSpline.design_matrix(grid, knots, k).toarray()
    q -= q.mean(axis=0)
    return q / np.sqrt((q**2).sum(axis=0))


def _log_softmax(e):
    return e - logsumexp(e)


@dataclass
class SpikeSlabPrior:
    """A fitted spike + slab prior on ``u = log gamma``.

    ``log_density`` is the log mass on ``grid`` (it sums to one).
    ``spike_centre`` and ``spike_sd`` are on the log scale.
    ``guard_triggered`` is True when the unguarded fit's own central
    interval excluded gamma = 1 and the spike was refitted at one;
    ``unguarded`` then keeps that first fit's parameters.
    """

    grid: np.ndarray
    log_density: np.ndarray
    spike_weight: float
    spike_centre: float
    spike_sd: float
    slab_coef: np.ndarray
    slab_pooled: bool = False
    guard_triggered: bool = False
    neg_log_lik: float = float("nan")
    n_genes: int = 0
    unguarded: dict = field(default_factory=dict)

    def interval(self, prob=GUARD_PROB):
        """The prior's own central interval of gamma."""
        s = posterior_summary(
            np.exp(self.log_density), self.grid, prob
        )
        return s["lower"], s["upper"]

    def to_dict(self):
        """JSON-ready form (:meth:`from_dict` reads it back)."""
        return {
            "grid_start": float(self.grid[0]),
            "grid_step": float(self.grid[1] - self.grid[0]),
            "grid_size": len(self.grid),
            "log_density": [
                round(float(x), 8) for x in self.log_density
            ],
            "spike_weight": float(self.spike_weight),
            "spike_centre": float(self.spike_centre),
            "spike_sd": float(self.spike_sd),
            "slab_coef": [float(x) for x in self.slab_coef],
            "slab_pooled": bool(self.slab_pooled),
            "guard_triggered": bool(self.guard_triggered),
            "neg_log_lik": float(self.neg_log_lik),
            "n_genes": int(self.n_genes),
            "unguarded": dict(self.unguarded),
        }

    @classmethod
    def from_dict(cls, d):
        grid = np.round(
            d["grid_start"]
            + d["grid_step"] * np.arange(d["grid_size"]),
            6,
        )
        return cls(
            grid=grid,
            log_density=np.asarray(d["log_density"], float),
            spike_weight=d["spike_weight"],
            spike_centre=d["spike_centre"],
            spike_sd=d["spike_sd"],
            slab_coef=np.asarray(d["slab_coef"], float),
            slab_pooled=d.get("slab_pooled", False),
            guard_triggered=d.get("guard_triggered", False),
            neg_log_lik=d.get("neg_log_lik", float("nan")),
            n_genes=d.get("n_genes", 0),
            unguarded=d.get("unguarded", {}),
        )


def _spike_slab_parts(grid, q, a, spike):
    """log g and the pieces its gradient needs."""
    lp0, m0, ls0 = spike
    s0 = np.exp(ls0)
    zz = (grid - m0) / s0
    e = -0.5 * zz**2
    spike_l = _log_softmax(e)
    slab_l = _log_softmax(q @ a)
    pi0 = expit(lp0)
    la = np.log(pi0) + spike_l
    lb = np.log1p(-pi0) + slab_l
    lg = np.logaddexp(la, lb)
    return lg, {
        "r": np.exp(la - lg),
        "slab_p": np.exp(slab_l),
        "spike_p": np.exp(spike_l),
        "z": zz,
        "s0": s0,
        "pi0": pi0,
    }


def spike_slab_log_density(
    grid, slab_coef, spike_weight, centre, sd, df=None
):
    """``log g`` on ``grid`` for given parameters (log scale centre and sd)."""
    q = slab_basis(grid, len(slab_coef) if df is None else df)
    lp0 = np.log(spike_weight) - np.log1p(-spike_weight)
    return _spike_slab_parts(
        grid,
        q,
        np.asarray(slab_coef, float),
        (lp0, centre, np.log(sd)),
    )[0]


def _nll_and_grad(lik, grid, q, a, spike):
    """Negative marginal log-likelihood of one cohort and its gradient.

    ``lik`` is ``exp(loglik - rowmax)``, ``(G, U)``. Returns ``(nll,
    grad_a, grad_spike)``; the slab penalty is added by the caller.
    """
    lg, d = _spike_slab_parts(grid, q, a, spike)
    g = np.exp(lg)
    m = np.maximum(lik @ g, 1e-300)
    nll = -np.log(m).sum()
    dlg = -g * ((1.0 / m) @ lik)  # d nll / d log g_u
    r = d["r"]
    db = dlg * (1 - r)
    grad_a = q.T @ db - db.sum() * (d["slab_p"] @ q)
    da = dlg * r
    z, s0 = d["z"], d["s0"]
    sp = d["spike_p"]
    g_m0 = (da * (z / s0)).sum() - da.sum() * (sp @ (z / s0))
    g_ls0 = (da * z**2).sum() - da.sum() * (sp @ z**2)
    g_lp0 = (dlg * (r - d["pi0"])).sum()
    return nll, grad_a, np.array([g_lp0, g_m0, g_ls0])


def _as_lik(loglik):
    ll = np.asarray(loglik, float)
    ll = np.where(np.isfinite(ll), ll, -1e300)
    return np.exp(ll - ll.max(axis=1, keepdims=True))


def _spike_bounds(centre):
    m0b = (
        (centre, centre)
        if centre is not None
        else SPIKE_CENTRE_BOUNDS
    )
    return [
        SPIKE_WEIGHT_LOGIT_BOUNDS,
        m0b,
        (np.log(SPIKE_SD_BOUNDS[0]), np.log(SPIKE_SD_BOUNDS[1])),
    ]


def _starts(centre):
    m0 = 0.05 if centre is None else centre
    lo = SPIKE_SD_BOUNDS[0]
    return [
        np.array([1.5, m0, np.log(max(s, lo))])
        for s in (0.08, 0.03, 0.15)
    ]


def _fit(liks, grid, df, c0, slab_coef, centres):
    """Joint fit: one slab (free or fixed) and one spike per cohort."""
    q = slab_basis(grid, df)
    names = list(liks)
    fixed_slab = slab_coef is not None
    na = 0 if fixed_slab else df

    def unpack(p):
        a = np.asarray(slab_coef, float) if fixed_slab else p[:na]
        return a, p[na:].reshape(len(names), 3)

    def obj(p):
        a, sps_ = unpack(p)
        val = 0.0 if fixed_slab else c0 * a @ a
        ga = np.zeros(df) if fixed_slab else 2 * c0 * a
        gs = np.zeros_like(sps_)
        for i, c in enumerate(names):
            v, g_a, g_s = _nll_and_grad(liks[c], grid, q, a, sps_[i])
            val += v
            ga = ga + g_a
            gs[i] = g_s
        grad = gs.ravel() if fixed_slab else np.r_[ga, gs.ravel()]
        return val, grad

    bounds = [] if fixed_slab else [SLAB_COEF_BOUNDS] * df
    for c in names:
        bounds.extend(_spike_bounds(centres.get(c)))
    best = None
    for k in range(3):
        s0 = np.concatenate(
            [_starts(centres.get(c))[k] for c in names]
        )
        p0 = s0 if fixed_slab else np.r_[np.zeros(df), s0]
        r = optimize.minimize(
            obj, p0, jac=True, method="L-BFGS-B", bounds=bounds
        )
        if best is None or r.fun < best.fun:
            best = r
    a, sps_ = unpack(best.x)
    out = {}
    for i, c in enumerate(names):
        lg = _spike_slab_parts(grid, q, a, sps_[i])[0]
        nll = _nll_and_grad(liks[c], grid, q, a, sps_[i])[0]
        out[c] = SpikeSlabPrior(
            grid=grid,
            log_density=lg,
            spike_weight=float(expit(sps_[i][0])),
            spike_centre=float(sps_[i][1]),
            spike_sd=float(np.exp(sps_[i][2])),
            slab_coef=np.asarray(a, float).copy(),
            neg_log_lik=float(nll),
            n_genes=int(liks[c].shape[0]),
        )
    return a, out


def prior_excludes_one(log_density, grid, prob=GUARD_PROB):
    """True when the prior's own central ``prob`` interval excludes gamma = 1.

    The guard's trigger: an uninformative gene's posterior is the
    prior, so this is exactly when every such gene's interval
    excludes one.
    """
    g = np.exp(np.asarray(log_density) - np.max(log_density))
    c = np.cumsum(g) / g.sum()
    tail = (1 - prob) / 2
    lo, hi = np.interp(tail, c, grid), np.interp(1 - tail, c, grid)
    return bool(lo > 0 or hi < 0)


def spike_excludes_one(centre, sd, grid, prob=GUARD_PROB):
    """True when the narrow component's own central interval excludes one.

    The normal of centre ``centre`` and sd ``sd`` (log scale),
    discretised on ``grid``. A gene with a little information keeps
    little of the smooth component, whose mass lies where any data
    rule it out, so its posterior is nearly the narrow component: when
    that component's interval excludes gamma = 1, so do the intervals
    of most genes, even if the whole prior's interval, widened by the
    smooth component's tails, still covers one.
    """
    w = np.exp(-0.5 * ((np.asarray(grid) - centre) / sd) ** 2)
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    tail = (1 - prob) / 2
    lo, hi = np.interp(tail, c, grid), np.interp(1 - tail, c, grid)
    return bool(lo > 0 or hi < 0)


def guard_triggers(prior):
    """The guard's trigger: the prior's or its narrow component's interval.

    Either central 89% interval (:func:`prior_excludes_one`,
    :func:`spike_excludes_one`) excluding gamma = 1.
    """
    return prior_excludes_one(
        prior.log_density, prior.grid
    ) or spike_excludes_one(
        prior.spike_centre, prior.spike_sd, prior.grid
    )


def _guarded(prior, refit):
    if not guard_triggers(prior):
        return prior
    new = refit()
    new.guard_triggered = True
    new.unguarded = {
        "spike_weight": prior.spike_weight,
        "spike_centre": prior.spike_centre,
        "spike_sd": prior.spike_sd,
        "neg_log_lik": prior.neg_log_lik,
    }
    return new


def fit_spike_slab(
    loglik,
    grid=None,
    *,
    slab_coef=None,
    df=SLAB_DF,
    c0=SLAB_PENALTY,
    guard=True,
):
    """Fit one cohort's spike + slab prior by maximum marginal likelihood.

    Parameters
    ----------
    loglik : array (G, U)
        Each gene's marginal log-likelihood of ``u`` on ``grid`` (any
        constant per row): :meth:`FactoredGeneLikelihood.marginal`.
    grid : array (U,), optional
        Default :func:`eb_grid`.
    slab_coef : array (df,), optional
        A slab fitted elsewhere (:func:`fit_pooled_spike_slab`): held
        fixed, only the spike is fitted.
    df, c0 : int, float
        Slab basis size and penalty.
    guard : bool, default True
        Refit with the spike's centre at gamma = 1 when the fitted
        prior's own 89% interval excludes one.

    Returns
    -------
    SpikeSlabPrior
    """
    grid = eb_grid() if grid is None else np.asarray(grid)
    lik = {"c": _as_lik(loglik)}
    _, out = _fit(lik, grid, df, c0, slab_coef, {})
    prior = out["c"]
    prior.slab_pooled = slab_coef is not None
    if not guard:
        return prior

    def refit():
        p = _fit(lik, grid, df, c0, slab_coef, {"c": 0.0})[1]["c"]
        p.slab_pooled = slab_coef is not None
        return p

    return _guarded(prior, refit)


def fit_pooled_spike_slab(
    logliks, grid=None, *, df=SLAB_DF, c0=SLAB_PENALTY, guard=True
):
    """One slab for several cohorts, a spike per cohort.

    ``logliks`` maps a cohort to its ``(G, U)`` marginal
    log-likelihoods. The slab's shape is fitted jointly (penalty once);
    each cohort's spike is its own. With ``guard``, a cohort whose
    prior's 89% interval excludes gamma = 1 has its spike refitted at
    one with the pooled slab held fixed.

    Returns
    -------
    (numpy.ndarray, dict)
        The slab coefficients and ``{cohort: SpikeSlabPrior}``.
    """
    grid = eb_grid() if grid is None else np.asarray(grid)
    liks = {c: _as_lik(v) for c, v in logliks.items()}
    a, out = _fit(liks, grid, df, c0, None, {})
    for c, prior in out.items():
        prior.slab_pooled = True
        if guard:

            def refit(c=c):
                p = _fit({c: liks[c]}, grid, df, c0, a, {c: 0.0})[1][
                    c
                ]
                p.slab_pooled = True
                return p

            out[c] = _guarded(prior, refit)
    return a, out


@dataclass
class GeneSelectionLikelihoods:
    """Every gene's likelihood on the grid, as the selection table needs it.

    Built by :meth:`.models.Model.gene_selection_likelihoods`.
    ``likelihoods`` is aligned with ``genes``; ``accounting`` holds the
    per-gene tumor counts and dispersion of the fit.
    """

    genes: object
    likelihoods: list
    accounting: object
    n_draws: int
    seed: int

    def marginal(self):
        """``(G, U)`` marginal log-likelihoods on :func:`eb_grid`."""
        mask = eb_mask()
        return np.stack(
            [lik.marginal(mask) for lik in self.likelihoods]
        )
