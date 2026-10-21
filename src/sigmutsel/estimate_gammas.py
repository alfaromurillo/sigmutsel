"""Selection estimation for somatic variants and genes

This module provides functionality for estimating the selection
intensity parameter γ (gamma) for individual amino acid variants and
for genes. The model assumes that the probability of observing a
variant in a tumor is governed by a Poisson process with rate `γ * μ`,
where `μ` is the estimated mutation rate of the variant in that
sample.

"""

import inspect
import logging

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as tt

from . import constants

logger = logging.getLogger(__name__)

# tt.clip floor/ceiling on P(present) inside estimate_gamma_from_mus.
# Keep in sync with the tt.clip(..., _CLIP_FLOOR, 1 - _CLIP_FLOOR)
# call below: this value determines the numerical ceiling computed
# from it (see _natural_gamma_ceiling).
_CLIP_FLOOR = 1e-12

# The declared robust likelihood (`u` in the article): every tumor
# fails to carry the mutation with probability at least ABSENCE_FLOOR,
# P(absent) = u + (1 - u) exp(-gamma mu), whatever its rate -- a tumor
# that cannot or need not carry it (a subtype off the gene's pathway,
# a hypermutator whose rate counts every site of an oncogene when
# only its hotspots are selected). It replaces what the hard clip
# above did silently: cap an absent tumor's evidence, here at
# -log(u) = 6.9 nats instead of 27.6, with no kink for the sampler.
# 0 restores the clip alone.
ABSENCE_FLOOR = 1e-3

# Floor on the per-tumor sigma of the log-mu cut prior (see
# estimate_gamma_from_mus). Without it, a tumor whose posterior mu
# draws happen to agree to numerical precision would collapse that
# tumor's prior to a point mass -- silently reverting to the pre-cut
# "mu known exactly" behavior for that one tumor, rather than
# reflecting genuinely tight (but nonzero) mu uncertainty.
_MIN_LOG_MU_SIGMA = 1e-3

# How the mu-posterior cut carries the stage-1 rate draws into gamma
# (see estimate_gamma_from_mus): "independent" moment-matches one
# lognormal per tumor, "shared" adds one latent for what every tumor's
# draw shares, "draws" uses the draws themselves (the exact cut).
MU_UNCERTAINTY_MODES = ("independent", "shared", "draws")

# Priors on gamma (see estimate_gamma_from_mus): "uniform" is
# Uniform(0, bound), the prior cancereffectsizeR parity is checked
# with; "jeffreys" is p(gamma) proportional to gamma^(-1/2) on (0,
# bound]; "loguniform" is p(gamma) proportional to 1/gamma on [lower,
# bound], lower = _loguniform_lower(...).
GAMMA_PRIORS = ("uniform", "jeffreys", "loguniform")

# Lowest gamma the "draws" grid reaches below the prior bound, as a
# fraction of it, for the priors that put mass all the way to 0:
# below it a Uniform prior holds a 1e-14 share of its mass and a
# Jeffreys-type one a 1e-7 share, so nothing the grid misses matters.
_GRID_FLOOR = 1e-14


def _loguniform_lower(upper_bound_prior, current_bound):
    """Lower end of the log-uniform prior's support.

    ``1 / upper_bound_prior`` (the support is symmetric in log gamma
    around 1 for the requested bound), kept at least six orders of
    magnitude below the bound actually in use, which the natural
    ceiling can pull far down.
    """
    return min(1.0 / float(upper_bound_prior), current_bound * 1e-6)


def _resolve_mu_uncertainty(mu_uncertainty, shared_mu_uncertainty):
    """The cut's mode, from the new argument or the old boolean."""
    if mu_uncertainty is None:
        return "shared" if shared_mu_uncertainty else "independent"
    if mu_uncertainty not in MU_UNCERTAINTY_MODES:
        raise ValueError(
            f"mu_uncertainty must be one of {MU_UNCERTAINTY_MODES}; "
            f"got {mu_uncertainty!r}."
        )
    if shared_mu_uncertainty and mu_uncertainty != "shared":
        raise ValueError(
            "shared_mu_uncertainty=True is mu_uncertainty='shared'; it "
            f"cannot be combined with mu_uncertainty={mu_uncertainty!r}."
        )
    return mu_uncertainty


def _gamma_prior_rv(gamma_prior, current_bound, upper_bound_prior):
    """The gamma random variable, inside the active model context.

    A Uniform carries the support (so the bound expansion and the
    natural ceiling read it as before) and a Potential turns it into
    the requested density.
    """
    if gamma_prior == "loguniform":
        lower = _loguniform_lower(upper_bound_prior, current_bound)
        gamma = pm.Uniform(
            name="gamma", lower=lower, upper=current_bound
        )
        pm.Potential("gamma_prior", -tt.log(gamma))
        return gamma
    gamma = pm.Uniform(name="gamma", lower=0, upper=current_bound)
    if gamma_prior == "jeffreys":
        pm.Potential("gamma_prior", -0.5 * tt.log(gamma))
    return gamma


def _log_prior_on_log_gamma(u, gamma_prior):
    """log p(log gamma), up to a constant, on the grid ``u``.

    The density of ``u = log gamma`` is p(gamma) gamma: u for the
    Uniform, u / 2 for the Jeffreys-type prior, 0 for the log-uniform.
    """
    if gamma_prior == "uniform":
        return u
    if gamma_prior == "jeffreys":
        return 0.5 * u
    return np.zeros_like(u)


def _compile_draws_loglik(
    n_yes,
    gene_tumor_dispersion,
    shapes_all,
    absence_floor,
    presence_clip,
):
    """log p(y | gamma, mu) for a grid of gamma and a batch of draws.

    Returns ``f(gamma_grid, mu_batch)`` -> ``(batch, grid)``, built
    from the same link functions as the MCMC fit
    (:func:`presence_probability`, :func:`log_absence_probability`),
    so the two paths score one likelihood: the Bernoulli with the clip,
    or the exact log-likelihood with ``presence_clip=False``.
    """
    grid = tt.dvector("gamma_grid")
    batch = tt.dmatrix("mu_batch")
    gamma = grid[None, :, None]
    mu = batch[:, None, :]
    shapes = (
        None
        if shapes_all is None
        else tt.as_tensor_variable(np.asarray(shapes_all, float))[
            None, None, :
        ]
    )
    if presence_clip:
        p = presence_probability(
            gamma,
            mu,
            gene_tumor_dispersion=gene_tumor_dispersion,
            gene_tumor_shape=shapes,
            absence_floor=absence_floor,
        )
        loglik = tt.sum(tt.log(p[:, :, :n_yes]), axis=-1) + tt.sum(
            tt.log1p(-p[:, :, n_yes:]), axis=-1
        )
    else:
        log_absent = log_absence_probability(
            gamma,
            mu,
            gene_tumor_dispersion=gene_tumor_dispersion,
            gene_tumor_shape=shapes,
            absence_floor=absence_floor,
        )
        loglik = tt.sum(
            tt.log1mexp(log_absent[:, :, :n_yes]), axis=-1
        ) + tt.sum(log_absent[:, :, n_yes:], axis=-1)
    return pytensor.function([grid, batch], loglik)


def _draws_loglik(f, grid, mu_draws, max_cells=2e7):
    """``f`` over every draw, in batches of at most ``max_cells``."""
    n_draws, n_tumors = mu_draws.shape
    batch = max(1, int(max_cells // max(1, len(grid) * n_tumors)))
    out = np.empty((n_draws, len(grid)))
    gamma_grid = np.exp(grid)
    for start in range(0, n_draws, batch):
        out[start : start + batch] = f(
            gamma_grid, mu_draws[start : start + batch]
        )
    return out


def _cut_posterior_on_grid(
    f,
    mu_draws,
    lower,
    upper,
    gamma_prior,
    n_grid,
    n_coarse=128,
    coarse_draws=256,
):
    """The exact cut posterior of log gamma, on a grid.

    For each stage-1 draw, the conditional posterior of log gamma
    given that draw's rates, normalised; then their average over the
    draws. A coarse grid over the whole support, on at most
    ``coarse_draws`` of the draws, finds where any of them has mass
    (within 30 nats of its own peak); the fine grid of ``n_grid``
    points covers that region, on every draw.
    """
    coarse = np.linspace(np.log(lower), np.log(upper), n_coarse)
    pick = np.unique(
        np.linspace(0, len(mu_draws) - 1, coarse_draws).astype(int)
    )
    lp = _draws_loglik(f, coarse, mu_draws[pick])
    lp += _log_prior_on_log_gamma(coarse, gamma_prior)[None, :]
    keep = (lp - lp.max(axis=1, keepdims=True) > -30).any(axis=0)
    idx = np.flatnonzero(keep)
    lo = coarse[max(idx.min() - 2, 0)]
    hi = coarse[min(idx.max() + 2, n_coarse - 1)]
    grid = np.linspace(lo, hi, n_grid)
    lp = _draws_loglik(f, grid, mu_draws)
    lp += _log_prior_on_log_gamma(grid, gamma_prior)[None, :]
    w = np.exp(lp - lp.max(axis=1, keepdims=True))
    w /= np.trapezoid(w, grid, axis=1)[:, None]
    return grid, w.mean(axis=0)


def _sample_grid_density(grid, density, n, rng):
    """``n`` draws of log gamma from a density tabulated on ``grid``."""
    cdf = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5 * (density[1:] + density[:-1]) * np.diff(grid)
            ),
        ]
    )
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


def _posterior_from_gamma_draws(values, chains):
    """Gamma draws as an InferenceData/DataTree like pm.sample's."""
    chains = max(1, int(chains))
    per_chain = len(values) // chains
    data = {
        "gamma": values[: chains * per_chain].reshape(
            chains, per_chain
        )
    }
    if "data" in inspect.signature(az.from_dict).parameters:
        return az.from_dict({"posterior": data})
    return az.from_dict(posterior=data)


def _natural_gamma_ceiling(mus_all, clip_floor=_CLIP_FLOOR):
    """Gamma value past which every sample's likelihood term is
    clip-saturated and the (log-)likelihood is flat in gamma.

    ``P(present) = 1 - exp(-gamma * mu)`` is clipped to
    ``[clip_floor, 1 - clip_floor]``. For a given sample, once
    ``gamma * mu > -log(clip_floor)`` (~27.6 at the default 1e-12),
    that sample's exp(-gamma * mu) has already collapsed to
    (numerically) 0 and its likelihood contribution stops changing
    with gamma. Once gamma clears this threshold for *every* sample
    -- governed by the smallest mu, which needs the largest gamma to
    saturate -- the whole likelihood is constant: MCMC can drift
    anywhere from there to the prior's upper bound with clean-looking
    diagnostics (chains agree with each other because there is
    nothing left to disagree about), producing an estimate that is
    numerically clean but scientifically arbitrary.
    """
    return -np.log(clip_floor) / np.min(mus_all)


def _natural_gamma_ceiling_dispersed(
    mus_all, gene_tumor_dispersion, clip_floor=_CLIP_FLOOR
):
    """The natural ceiling under gene-tumor dispersion.

    With ``gene_tumor_dispersion`` = ``phi``, ``P(absent in j) =
    (1 + gamma M / phi) ** (-phi p_j)`` with ``M = sum_j mu_j`` and
    ``p_j = mu_j / M`` (see :func:`estimate_gamma_from_mus`). It
    reaches the clip floor for every sample only once
    ``phi p_min log1p(gamma M / phi) > -log(clip_floor)``, i.e. at a
    much larger gamma than the dispersion-free ceiling -- often one
    that overflows, in which case the likelihood never saturates in
    floating point and the ceiling is infinite.
    """
    mus_all = np.asarray(mus_all, dtype=float)
    m = mus_all.sum()
    p_min = mus_all.min() / m
    with np.errstate(over="ignore", invalid="ignore"):
        ceiling = (gene_tumor_dispersion / m) * np.expm1(
            -np.log(clip_floor) / (gene_tumor_dispersion * p_min)
        )
    return float(ceiling) if np.isfinite(ceiling) else np.inf


def _natural_gamma_ceiling_shaped(
    mus_all, shapes, clip_floor=_CLIP_FLOOR
):
    """The natural ceiling under per-tumor shapes ``k_j``.

    ``P(absent in j) = (1 + gamma mu_j / k_j) ** (-k_j)`` reaches the
    clip floor once ``gamma > (k_j / mu_j) expm1(-log(clip_floor) /
    k_j)``; every tumor must get there, so the largest such value is
    the ceiling (infinite if it overflows).
    """
    mus_all = np.asarray(mus_all, dtype=float)
    shapes = np.asarray(shapes, dtype=float)
    with np.errstate(
        over="ignore", invalid="ignore", divide="ignore"
    ):
        per = (shapes / mus_all) * np.expm1(
            -np.log(clip_floor) / shapes
        )
    per = per[np.isfinite(mus_all) & (mus_all > 0)]
    if per.size == 0 or not np.all(np.isfinite(per)):
        return np.inf
    return float(per.max())


class ConvergenceError(RuntimeError):
    """MCMC sampling completed but convergence diagnostics are bad.

    Distinguishes a fit that ran without crashing but produced an
    untrustworthy posterior (high R-hat, low ESS, divergences) from
    an actual sampling crash (``pm.SamplingError`` etc.). Subclasses
    ``RuntimeError`` so existing giant-non-driver-gene skip logic
    (which catches ``RuntimeError``) still catches this too.
    """


def presence_probability(
    gamma,
    mu,
    gene_tumor_dispersion=None,
    gene_tumor_shape=None,
    absence_floor=0.0,
):
    """P(tumor carries the variant), given gamma and its rate.

    The model's link, in three forms. With no gene-tumor
    dispersion, a tumor's presence is Poisson with rate
    ``gamma * mu``, so ``P = 1 - exp(-gamma mu)``. With dispersion,
    the rate is Gamma-randomized around itself and integrating it
    out gives the negative-binomial form ``P = 1 - (1 + gamma mu /
    k)^(-k)``, written through ``log1p`` for stability: `k` is the
    per-tumor shape ``phi p_gj`` when one is supplied, and
    ``phi mu / sum(mu)`` when a scalar `phi` is.

    Parameters
    ----------
    gamma : tensor or float
        Selection intensity, shared across the tumors.
    mu : tensor or array
        Per-tumor mutation rate, in the fit's tumor order.
    gene_tumor_dispersion : float, optional
        Scalar `phi` for the gene.
    gene_tumor_shape : array, optional
        Per-tumor shape ``k_j``, which takes precedence over a
        scalar `phi`.
    absence_floor : float, default 0.0
        ``u``: the presence probability above is multiplied by
        ``1 - u``, so a tumor is absent with probability at least
        ``u`` however high its rate (:data:`ABSENCE_FLOOR`).

    Returns
    -------
    The per-tumor probability, clipped off 0 and 1 by
    ``_CLIP_FLOOR`` -- the same bound ``_natural_gamma_ceiling``
    computes gamma's identifiable range from, which is why this is
    one constant and not a literal in two places.

    Notes
    -----
    This is the unit ``estimate_gamma_from_mus``'s
    ``presence_model=`` replaces. A replacement is called inside the
    active ``pm.Model()`` context, so it may create random variables
    of its own -- which is the point, for a selection model with
    per-tumor covariates. What it must not change is the
    parameterization the machinery around it depends on: gamma is a
    scalar with a ``Uniform(0, bound)`` prior, and the natural
    ceiling, the bound auto-expansion and the convergence retries
    all read it as one. A model that broke that would keep running
    and quietly stop being retried correctly.
    """
    ceiling = 1 - _CLIP_FLOOR

    log_absent = _log_absent_unfloored(
        gamma, mu, gene_tumor_dispersion, gene_tumor_shape
    )
    present = -tt.expm1(log_absent)
    if absence_floor:
        present = (1 - absence_floor) * present
    return tt.clip(present, _CLIP_FLOOR, ceiling)


def _log_absent_unfloored(
    gamma, mu, gene_tumor_dispersion, gene_tumor_shape
):
    """``log P(absent)`` of the link, before any floor or clip."""
    if gene_tumor_shape is not None:
        return -gene_tumor_shape * tt.log1p(
            gamma * mu / gene_tumor_shape
        )
    if gene_tumor_dispersion is None:
        return -gamma * mu
    m_total = tt.sum(mu, axis=-1, keepdims=True)
    return (
        -gene_tumor_dispersion
        * (mu / m_total)
        * tt.log1p(gamma * m_total / gene_tumor_dispersion)
    )


def log_absence_probability(
    gamma,
    mu,
    gene_tumor_dispersion=None,
    gene_tumor_shape=None,
    absence_floor=0.0,
):
    """``log P(tumor lacks the variant)``, exactly, with no clip.

    The same link as :func:`presence_probability` -- Poisson, or the
    dispersed form, with the floor ``u`` when given: ``P(absent) = u +
    (1 - u) P_0(absent)`` -- on the log scale and without the clip on
    P(present). It is what ``estimate_gamma_from_mus(...,
    presence_clip=False)`` scores: an absent tumor then costs its full
    ``gamma mu`` (with ``u = 0``), as in cancereffectsizeR's
    ``sswm_lik``, however large.
    """
    log_absent = _log_absent_unfloored(
        gamma, mu, gene_tumor_dispersion, gene_tumor_shape
    )
    if absence_floor:
        return tt.logaddexp(
            np.log(absence_floor),
            np.log1p(-absence_floor) + log_absent,
        )
    return tt.as_tensor_variable(log_absent)


def estimate_gamma_from_mus(
    mus_yes,
    mus_no,
    draws=4000,
    upper_bound_prior=1e6,
    burn=1000,
    chains=4,
    save_name=None,
    kwargs=None,
    max_retries=5,
    factor_of_reduction=10,
    auto_expand_bound=True,
    max_bound_expansions=4,
    expand_factor=10.0,
    saturation_ratio=0.2,
    auto_raise_target_accept=True,
    max_convergence_retries=3,
    target_accept_schedule=(0.9, 0.95, 0.99),
    rhat_threshold=1.01,
    ess_threshold=200,
    cap_at_natural_ceiling=True,
    gene_tumor_dispersion=None,
    gene_tumor_shape=None,
    presence_model=None,
    absence_floor=ABSENCE_FLOOR,
    presence_clip=True,
    shared_mu_uncertainty=False,
    mu_uncertainty=None,
    gamma_prior="uniform",
    n_grid=256,
    max_mu_draws=1000,
):
    """Estimate gamma from mu values using a Poisson observation model.

    This function infers the posterior distribution of a selective
    advantage parameter gamma under the model:
        P(variant present) = 1 - exp(-gamma * mu)

    It can be used with multiple variants if it is assumed that they
    have the same selective advantage (same gamma) and that the
    probability that a variant occurs and its selection are not
    impacted by having another one of the variants considered (no
    epistasis between the variants agreggated).

    Parameters
    ----------
    mus_yes : array-like
        Mu values for tumors with the variant(s). Either 1-D
        ``(tumors,)`` point estimates (today's default behavior,
        mu treated as known exactly) or 2-D ``(draws, tumors)``
        posterior draws of mu, one column per tumor. If 2-D,
        ``mus_no`` must be 2-D too, with the same number of draws --
        see the "mu posterior cut" note below.

    mus_no : array-like
        Mu values for tumors without the variant(s). Same shape
        rule as ``mus_yes``.

    gene_tumor_dispersion : float or None, default None
        Optional gene-tumor overdispersion ``phi``. ``None`` (default)
        keeps the model above exactly. Given a value, each tumor's
        rate is ``mu_tilde_j ~ Gamma(shape=phi * p_j, rate=phi / M)``
        with ``M = sum_j mu_j`` and ``p_j = mu_j / M`` -- mean
        ``mu_j``, and, conditional on the total, a
        Dirichlet-Multinomial allocation with concentration
        ``phi * p_j`` -- so that::

            P(present in j) = 1 - (1 + gamma M / phi) ** (-phi p_j)

        which tends to ``1 - exp(-gamma mu_j)`` as ``phi -> inf``.
        Ignoring real dispersion of this kind biases gamma
        **downward**: a mutated tumor is less informative per unit
        of rate when rates vary more between tumors than ``mu``
        says. Both ``mus_yes`` and ``mus_no`` must together cover
        every tumor, since ``M`` sums over all of them. The
        construction also gives the gene's total rate a coefficient
        of variation of ``1 / sqrt(phi)``, which overlaps any
        separate per-gene rate correction already applied to ``mu``.

    gene_tumor_shape : tuple of array-like or None, default None
        The same gene-tumor Gamma dispersion, with each tumor's shape
        ``k_j`` given directly as ``(shapes_yes, shapes_no)`` in the
        order of ``mus_yes``/``mus_no``: ``P(present in j) =
        1 - (1 + gamma mu_j / k_j) ** (-k_j)``. ``gene_tumor_dispersion``
        is the special case ``k_j = phi mu_j / M``. Use this form when
        the dispersion belongs to a larger unit than the item being
        scored -- e.g. a variant inheriting its gene's per-tumor
        multiplier, whose shape is ``phi_gene * p_gene,j`` rather than
        anything computed from the variant's own rates. Mutually
        exclusive with ``gene_tumor_dispersion``.

    presence_model : callable or None, default None
        Replaces :func:`presence_probability`, the link from gamma
        and a rate to P(present). Called as ``presence_model(gamma,
        mu, gene_tumor_dispersion=, gene_tumor_shape=)`` inside the
        active model context, so it may create random variables of
        its own; it must return a per-tumor probability aligned with
        ``mus_yes`` followed by ``mus_no``. See that function's
        notes for what a replacement must leave alone -- everything
        around this call assumes gamma is the bounded scalar it was
        handed.
    absence_floor : float, default ABSENCE_FLOOR
        ``u`` of :func:`presence_probability` (:data:`ABSENCE_FLOOR`);
        not passed to a custom ``presence_model``. Recorded in the
        posterior's attrs.
    presence_clip : bool, default True
        Keep P(present) within ``_CLIP_FLOOR`` of 0 and 1, a numerical
        guard that also caps an absent tumor's cost at -log(1e-12) =
        27.6 nats. False scores the exact log-likelihood instead
        (:func:`log_absence_probability`), so with no dispersion, no
        cut and ``absence_floor=0`` the likelihood is
        cancereffectsizeR's ``sswm_lik`` exactly, and the MAP
        (``draws=1``) its maximum-likelihood estimate. Not available
        with a custom ``presence_model``. Recorded in the posterior's
        attrs.

    shared_mu_uncertainty : bool, default False
        With 2-D draws only. The default cut gives every tumor its own
        ``log_mu`` latent, independent of the others; but a gene's
        draws share their uncertainty across tumors (the covariate
        scale and ``r_g`` multiply every tumor's rate by the same
        factor in a draw), and independent latents average that shared
        part away over the tumors, so gamma's interval keeps almost
        none of it. True splits each draw's ``log mu`` into a shared
        part (the draw's mean over tumors, centred) and per-tumor
        residuals, and fits one shared latent ``log_mu_shared ~
        Normal(0, sd of the shared part)`` added to every tumor's
        mean, plus independent per-tumor latents only where the
        residuals vary (e.g. spectrum draws). For draws that factor as
        a per-draw scale times a fixed per-tumor baseline -- the gene
        draws of :meth:`.models.Model.compute_mu_g_posterior_draws`
        without spectrum draws -- this is the stage-1 posterior
        exactly, as a lognormal. Recorded in the posterior's attrs.
        The same as ``mu_uncertainty="shared"``, kept as an alias.

    mu_uncertainty : {"independent", "shared", "draws"} or None
        How the cut carries the stage-1 draws into gamma; ``None``
        reads ``shared_mu_uncertainty`` (so the default is today's
        "independent"). "independent" and "shared" are the latent
        forms above. "draws" is the exact cut (Plummer 2015):
        ``p(gamma | y) = mean over draws d of p(gamma | y, mu^(d))``,
        each draw's whole rate vector used as it is, correlations
        between tumors included, and the presence data never
        reweighting the draws. gamma is one-dimensional, so this needs
        no MCMC: each draw's conditional posterior of ``log gamma`` is
        evaluated on a grid (the same likelihood as the MCMC fit,
        built from :func:`presence_probability` /
        :func:`log_absence_probability`), normalised, averaged over
        the draws, and ``draws`` values are sampled from the mixture.
        With 1-D mus it is the same grid with mu known exactly. Not
        with a custom ``presence_model`` or ``draws=1``. Recorded in
        the posterior's attrs (``mu_uncertainty``).

    gamma_prior : {"uniform", "jeffreys", "loguniform"}, default "uniform"
        Prior on gamma. "uniform" is ``Uniform(0, bound)``, as before
        (with ``presence_clip=False``, ``absence_floor=0`` and no cut,
        the MAP is cancereffectsizeR's estimate). "jeffreys" is
        ``p(gamma) ~ gamma^(-1/2)`` on ``(0, bound]``, the Jeffreys
        prior of a Poisson rate; "loguniform" is ``p(gamma) ~ 1/gamma``
        on ``[min(1 / upper_bound_prior, bound * 1e-6), bound]``. While
        presences are rare, ``P(present) ~ gamma mu`` and the
        likelihood behaves like ``gamma^n exp(-gamma S)`` (``n``
        carriers), so a prior ``gamma^(a - 1)`` gives a posterior close
        to Gamma(n + a, S): shape ``n + 1`` for the Uniform -- one
        presence more than the data, which calls a gene with few
        expected mutations "selected" too often -- ``n + 1/2`` for the
        Jeffreys-type prior and ``n`` for the log-uniform. Recorded in
        the posterior's attrs.

    n_grid : int, default 256
        Fine-grid points for ``mu_uncertainty="draws"``.
    max_mu_draws : int or None, default 1000
        With ``mu_uncertainty="draws"``, use at most this many of the
        stage-1 draws, evenly spaced (the cost is draws x grid x
        tumors; a production gene has 4,000 draws). Thinning only adds
        Monte Carlo error to the mixture; ``None`` uses every draw.
        The number used is recorded (``mu_draws_used``).

    draws : int, default=10000
        Number of posterior samples to draw. If draws == 1, returns MAP/MLE.

    upper_bound_prior : float, default=1e6
        Initial upper bound for the uniform prior over gamma.

    burn : int, default=1000
        Number of tuning steps for MCMC sampling.

    chains : int, default=8
        Number of MCMC chains to run.

    save_name : str or None, default=None
        If provided, saves the posterior trace to this path.

    kwargs : dict or None, default=None
        Additional arguments to pass to :func:`pymc.sample` or
        :func:`pymc.find_MAP`.

    max_retries : int, default=5
        Maximum number of retries with reduced prior bound, triggered
        when sampling itself fails (numerical errors).

    factor_of_reduction : float, default=10
        Factor by which to reduce the prior upper bound if sampling fails.

    auto_expand_bound : bool, default=True
        A ``Uniform(0, upper_bound_prior)`` prior can silently cap
        gamma: if the data support a value close to or beyond the
        bound, the posterior gets truncated at the ceiling rather
        than converging to its natural scale, while MCMC diagnostics
        (R-hat, ESS) can still look clean, since sampling a uniform
        prior up to a hard edge is not itself a numerical problem.
        If True, after a successful fit, check whether the posterior
        mean exceeds ``saturation_ratio * upper_bound_prior``; if so,
        multiply the bound by ``expand_factor`` and refit, up to
        ``max_bound_expansions`` times, until the estimate settles at
        a value that is a small fraction of its own ceiling.

    max_bound_expansions : int, default=4
        Maximum number of times to multiply the prior bound and
        refit when the posterior looks saturated against it.

    expand_factor : float, default=10.0
        Factor to multiply the prior upper bound by on each
        saturation-triggered expansion.

    saturation_ratio : float, default=0.2
        If the posterior mean (or MAP estimate) exceeds this fraction
        of the current prior upper bound, the fit is considered
        potentially bound-limited and triggers an expansion (while
        expansions remain).

    auto_raise_target_accept : bool, default=True
        A weakly-identified gamma (data barely constrain it, e.g. a
        gene whose mutation count carries little selection signal)
        can produce a sampler that mixes poorly -- divergences, R-hat
        > 1.01, low ESS -- *without* the posterior being anywhere
        near the prior bound, so ``auto_expand_bound`` above does not
        catch or fix it. This is a different failure mode: not a
        truncated prior, but a genuinely hard-to-sample posterior. If
        True, after a successful fit (draws != 1), check R-hat, bulk
        ESS, and divergence count; if any looks bad, resample the
        same bound with a higher NUTS ``target_accept`` (smaller step
        size, the standard first response to divergences), up to
        ``max_convergence_retries`` times.

    max_convergence_retries : int, default=3
        Maximum number of resamples with a higher ``target_accept``
        when convergence diagnostics look bad. Independent of
        ``max_retries`` (hard sampling failures) and
        ``max_bound_expansions`` (bound saturation).

    target_accept_schedule : tuple of float, default=(0.9, 0.95, 0.99)
        ``target_accept`` values to try, in order, on successive
        convergence retries (PyMC's NUTS default is 0.8).

    rhat_threshold : float, default=1.01
        R-hat above this triggers a convergence retry.

    ess_threshold : float, default=200
        Bulk ESS below this triggers a convergence retry.

    cap_at_natural_ceiling : bool, default=True
        ``auto_expand_bound`` was built for a genuinely large true
        gamma getting clipped by too-tight a prior (e.g. KRAS
        p.G12D). But a *weakly-identified* gamma (e.g. a passenger-
        like giant gene) can look bound-limited for a different
        reason: past some data-dependent scale, every sample's
        ``P(present) = 1 - exp(-gamma * mu)`` is already clip-
        saturated (see :func:`_natural_gamma_ceiling`), so the
        likelihood is flat and the posterior mean can sit anywhere
        up near whatever bound is currently active -- expanding
        further doesn't recover more signal, it just gives MCMC a
        bigger flat region to wander, producing a *larger* but not
        *more accurate* number. If True, cap the initial
        ``upper_bound_prior`` at this natural ceiling and refuse
        further bound expansion once ``current_bound`` has reached
        it, marking the result ``likelihood_saturated`` instead of
        continuing to chase the bound outward.

    Returns
    -------
    results : arviz.InferenceData or dict
        Posterior samples (or MAP/MLE estimate) of gamma. For MCMC
        results, ``results.posterior.attrs`` records
        ``final_upper_bound_prior`` and ``bound_expansions`` so
        callers can tell whether (and how much) the bound had to be
        expanded, plus ``likelihood_saturated`` and
        ``natural_gamma_ceiling`` (see ``cap_at_natural_ceiling``)
        -- when ``likelihood_saturated`` is True, treat the estimate
        as "at least this large," not a precise point value.

    Raises
    ------
    RuntimeError
        If sampling fails after all retries.
    ConvergenceError
        If MCMC diagnostics (R-hat, ESS, divergences) still look bad
        after ``max_convergence_retries`` resamples at increasing
        ``target_accept``. Subclasses ``RuntimeError``, so existing
        ``except RuntimeError`` callers (e.g. code that skips known-
        unstable giant non-driver genes) still catch it; catch
        ``ConvergenceError`` specifically to distinguish a hard
        sampling crash from a fit that ran but shouldn't be trusted.

    Notes
    -----
    Uses ``sigmutsel.constants.random_seed`` for reproducibility.
    Left unset (``None``) by default; set it explicitly (e.g. in the
    calling application, before running the pipeline) to make
    sampling reproducible across runs and machines.

    **The mu posterior cut.** Passing 1-D ``mus_yes``/``mus_no``
    treats every tumor's mu as known exactly, so the returned
    interval for gamma reflects only the Bernoulli sampling term
    (how many tumors carry the variant). Since
    ``P = 1 - exp(-gamma * mu)``, a fractional uncertainty in mu
    passes into gamma almost 1:1 -- worst exactly where the
    Bernoulli term looks tightest. Passing 2-D ``(draws, tumors)``
    arrays instead (e.g. from
    :meth:`.models.Model.compute_mu_g_posterior_draws`) fits

        log_mu[tumor] ~ Normal(mean(log(draws)), std(log(draws)))
        gamma ~ Uniform(0, upper_bound_prior)
        P = 1 - exp(-gamma * mu)

    with one ``log_mu`` latent per tumor, informed only by its own
    stage-1 posterior draws (a Normal moment-matched to them) and by
    its single Bernoulli observation here. This is a *cut*, not a
    joint fit: the stage-1 posterior enters as a fixed prior, so no
    information flows back from this Bernoulli likelihood into mu.
    Joint fitting would let driver signal feed back into what is
    meant to be a neutral rate -- the same leak class the production
    vs. evaluation ``r_g`` split (see :mod:`.estimate_rg`) exists to
    avoid.

    Strictly, the latent forms ("independent", "shared") are only
    approximately a cut. A latent's prior is the stage-1 posterior,
    but the presence likelihood still reweights it through each
    value's normalising constant ``int p(y | gamma, mu) p(gamma)
    dgamma``. For a shared scale ``s`` under the Uniform prior that
    constant grows as ``exp(-s)``, which tilts the scale down and
    raises gamma by about ``exp(var(s))`` (by half that under the
    Jeffreys-type prior; not at all under the log-uniform, whose
    constant does not depend on ``s``). ``mu_uncertainty="draws"`` is
    the exact cut: every draw is a whole rate vector and each keeps its
    weight. The "independent" form also averages a shared scale away
    over the tumors, so gamma keeps almost none of its uncertainty.

    """
    mu_uncertainty = _resolve_mu_uncertainty(
        mu_uncertainty, shared_mu_uncertainty
    )
    if gamma_prior not in GAMMA_PRIORS:
        raise ValueError(
            f"gamma_prior must be one of {GAMMA_PRIORS}; got "
            f"{gamma_prior!r}."
        )
    if mu_uncertainty == "draws" and (
        presence_model is not None or draws == 1
    ):
        raise ValueError(
            "mu_uncertainty='draws' needs the built-in link and "
            "posterior draws (draws > 1)."
        )
    shared_mu_uncertainty = mu_uncertainty == "shared"
    mus_yes_arr = np.asarray(mus_yes, dtype=float)
    mus_no_arr = np.asarray(mus_no, dtype=float)
    if mus_yes_arr.ndim not in (1, 2) or mus_no_arr.ndim not in (
        1,
        2,
    ):
        raise ValueError(
            "mus_yes and mus_no must be 1-D (point estimates) or "
            f"2-D (draws, tumors); got ndim {mus_yes_arr.ndim} and "
            f"{mus_no_arr.ndim}."
        )

    use_mu_prior = mus_yes_arr.ndim == 2 or mus_no_arr.ndim == 2
    log_mu_mean = log_mu_sigma = None
    n_zero_rate_absent = 0
    if use_mu_prior:
        if mus_yes_arr.ndim != 2 or mus_no_arr.ndim != 2:
            raise ValueError(
                "mus_yes and mus_no must both be 2-D (draws, "
                "tumors) to use the mu-posterior cut -- got ndim "
                f"{mus_yes_arr.ndim} and {mus_no_arr.ndim}. Pass "
                "both as 1-D point estimates instead if only one "
                "side has posterior draws."
            )
        if mus_yes_arr.shape[0] != mus_no_arr.shape[0]:
            raise ValueError(
                "mus_yes and mus_no must share the same number of "
                f"posterior draws (axis 0); got {mus_yes_arr.shape[0]}"
                f" and {mus_no_arr.shape[0]}."
            )
        # A tumor whose rate is 0 in every draw cannot carry the
        # mutation, so its absence has probability 1 whatever gamma is
        # and dropping it changes nothing -- but log(0) would make the
        # cut's initial log-probability -inf and stop the sampler from
        # starting at all. This happens: a tumor whose fitted
        # signatures emit nothing of a type has mu = 0 for it, and it
        # silently cost TP53 p.R175H its gamma in two TCGA cohorts.
        dead_no = (mus_no_arr <= 0).all(axis=0)
        n_zero_rate_absent = int(dead_no.sum())
        if n_zero_rate_absent:
            mus_no_arr = mus_no_arr[:, ~dead_no]
            if gene_tumor_shape is not None:
                shapes_yes, shapes_no = gene_tumor_shape
                gene_tumor_shape = (
                    shapes_yes,
                    np.asarray(shapes_no, dtype=float).ravel()[
                        ~dead_no
                    ],
                )
        if (mus_yes_arr <= 0).any() or (mus_no_arr <= 0).any():
            raise ValueError(
                "The mu-posterior cut needs positive rates: a tumor "
                "carrying the mutation has rate 0, or a rate is 0 in "
                "only some draws."
            )
        n_yes, n_no = mus_yes_arr.shape[1], mus_no_arr.shape[1]
        log_mu_draws = np.log(
            np.concatenate([mus_yes_arr, mus_no_arr], axis=1)
        )
        log_mu_mean = log_mu_draws.mean(axis=0)
        log_mu_sigma = (
            log_mu_draws.std(axis=0, ddof=1)
            if log_mu_draws.shape[0] > 1
            else np.zeros(log_mu_draws.shape[1])
        )
        log_mu_sigma = np.maximum(log_mu_sigma, _MIN_LOG_MU_SIGMA)
        mus_all = np.exp(log_mu_mean)
        if shared_mu_uncertainty:
            shared = log_mu_draws.mean(axis=1)
            shared = shared - shared.mean()
            shared_sigma = max(
                float(shared.std(ddof=1)) if len(shared) > 1 else 0.0,
                _MIN_LOG_MU_SIGMA,
            )
            residual_sigma = (
                (log_mu_draws - shared[:, None]).std(axis=0, ddof=1)
                if log_mu_draws.shape[0] > 1
                else np.zeros(log_mu_draws.shape[1])
            )
            residual_free = np.flatnonzero(
                residual_sigma > _MIN_LOG_MU_SIGMA
            )
    else:
        n_yes, n_no = len(mus_yes_arr), len(mus_no_arr)
        mus_all = np.concatenate([mus_yes_arr, mus_no_arr])

    shapes_all = None
    if gene_tumor_shape is not None:
        if gene_tumor_dispersion is not None:
            raise ValueError(
                "Pass gene_tumor_dispersion or gene_tumor_shape, not both."
            )
        shapes_yes, shapes_no = gene_tumor_shape
        shapes_all = np.concatenate(
            [
                np.asarray(shapes_yes, dtype=float).ravel(),
                np.asarray(shapes_no, dtype=float).ravel(),
            ]
        )
        if shapes_all.shape[0] != n_yes + n_no:
            raise ValueError(
                "gene_tumor_shape must give one shape per tumor, in the order "
                f"of mus_yes then mus_no; got {shapes_all.shape[0]} for "
                f"{n_yes + n_no} tumors."
            )
        if not np.all(np.isfinite(shapes_all)) or np.any(
            shapes_all <= 0
        ):
            raise ValueError(
                "gene_tumor_shape values must be positive and finite."
            )
        natural_ceiling = _natural_gamma_ceiling_shaped(
            mus_all, shapes_all
        )
    elif gene_tumor_dispersion is not None:
        gene_tumor_dispersion = float(gene_tumor_dispersion)
        if (
            not np.isfinite(gene_tumor_dispersion)
            or gene_tumor_dispersion <= 0
        ):
            raise ValueError(
                "gene_tumor_dispersion must be a positive finite number or "
                f"None; got {gene_tumor_dispersion!r}."
            )
        natural_ceiling = _natural_gamma_ceiling_dispersed(
            mus_all, gene_tumor_dispersion
        )
    else:
        natural_ceiling = _natural_gamma_ceiling(mus_all)

    if not presence_clip:
        if presence_model is not None:
            raise ValueError(
                "presence_clip=False scores the built-in link exactly; "
                "it cannot be combined with a custom presence_model."
            )
        if n_no > 0 and not absence_floor:
            # Unclipped and unfloored, every absent tumor's cost keeps
            # growing with gamma: the likelihood never goes flat.
            natural_ceiling = np.inf

    if kwargs is None:
        kwargs = {}

    attempt = 0
    expansions = 0
    convergence_retries = 0
    likelihood_saturated = False
    current_bound = upper_bound_prior
    if cap_at_natural_ceiling and current_bound > natural_ceiling:
        logger.info(
            "Capping gamma's initial prior bound at %.3g (the "
            "natural numerical ceiling for this data, from a "
            "requested %.3g) -- past this scale every sample's "
            "likelihood term is already clip-saturated and gamma "
            "is not identifiable from the data.",
            natural_ceiling,
            current_bound,
        )
        current_bound = natural_ceiling
    if mu_uncertainty == "draws":
        mu_draws = (
            np.exp(log_mu_draws) if use_mu_prior else mus_all[None, :]
        )
        if max_mu_draws is not None and len(mu_draws) > max_mu_draws:
            mu_draws = mu_draws[
                np.unique(
                    np.linspace(
                        0, len(mu_draws) - 1, max_mu_draws
                    ).astype(int)
                )
            ]
        return _estimate_gamma_on_grid(
            mu_draws,
            n_yes,
            current_bound=current_bound,
            upper_bound_prior=upper_bound_prior,
            natural_ceiling=natural_ceiling,
            gamma_prior=gamma_prior,
            gene_tumor_dispersion=gene_tumor_dispersion,
            shapes_all=shapes_all,
            absence_floor=absence_floor,
            presence_clip=presence_clip,
            n_grid=n_grid,
            draws=draws,
            chains=chains,
            auto_expand_bound=auto_expand_bound,
            max_bound_expansions=max_bound_expansions,
            expand_factor=expand_factor,
            saturation_ratio=saturation_ratio,
            cap_at_natural_ceiling=cap_at_natural_ceiling,
            n_zero_rate_absent=n_zero_rate_absent,
            use_mu_prior=use_mu_prior,
        )
    current_kwargs = dict(kwargs)
    while attempt <= max_retries:
        try:
            with pm.Model():
                gamma = _gamma_prior_rv(
                    gamma_prior, current_bound, upper_bound_prior
                )

                if use_mu_prior:
                    # The cut: log_mu's prior is moment-matched to
                    # the stage-1 posterior draws and is otherwise
                    # only informed by this tumor's own Bernoulli
                    # observation below -- no joint fit, no feedback
                    # from gamma/selection back into mu.
                    if shared_mu_uncertainty:
                        # One latent for what every tumor's draw
                        # shares (scale, r_g), per-tumor latents only
                        # for what varies by tumor.
                        log_mu = log_mu_mean + pm.Normal(
                            name="log_mu_shared",
                            mu=0.0,
                            sigma=shared_sigma,
                        )
                        if residual_free.size:
                            residual = pm.Normal(
                                name="log_mu_residual",
                                mu=0.0,
                                sigma=residual_sigma[residual_free],
                                shape=residual_free.size,
                            )
                            log_mu = tt.inc_subtensor(
                                log_mu[residual_free], residual
                            )
                    else:
                        log_mu = pm.Normal(
                            name="log_mu",
                            mu=log_mu_mean,
                            sigma=log_mu_sigma,
                            shape=len(log_mu_mean),
                        )
                    mu = pm.math.exp(log_mu)
                else:
                    mu = mus_all

                if not presence_clip:
                    # Exact: carriers log(1 - P(absent)), absences
                    # log P(absent), with no clip on either.
                    log_absent = log_absence_probability(
                        gamma,
                        mu,
                        gene_tumor_dispersion=gene_tumor_dispersion,
                        gene_tumor_shape=shapes_all,
                        absence_floor=absence_floor,
                    )
                    pm.Potential(
                        "variants_observed",
                        tt.sum(tt.log1mexp(log_absent[:n_yes]))
                        + tt.sum(log_absent[n_yes:]),
                    )
                else:
                    if presence_model is None:
                        Ps = presence_probability(
                            gamma,
                            mu,
                            gene_tumor_dispersion=gene_tumor_dispersion,
                            gene_tumor_shape=shapes_all,
                            absence_floor=absence_floor,
                        )
                    else:
                        Ps = presence_model(
                            gamma,
                            mu,
                            gene_tumor_dispersion=gene_tumor_dispersion,
                            gene_tumor_shape=shapes_all,
                        )

                    pm.Bernoulli(
                        name="variants_observed",
                        p=Ps,
                        observed=np.concatenate(
                            [np.ones(n_yes), np.zeros(n_no)]
                        ),
                    )

                if draws == 1:
                    results = pm.find_MAP(
                        seed=constants.random_seed, **current_kwargs
                    )
                else:
                    results = pm.sample(
                        draws=int(draws / chains),
                        chains=chains,
                        tune=burn,
                        random_seed=constants.random_seed,
                        **current_kwargs,
                    )

            if draws == 1:
                gamma_point = float(results["gamma"])
            else:
                gamma_point = float(
                    results.posterior["gamma"].values.mean()
                )

            looks_bound_limited = (
                auto_expand_bound
                and gamma_point > saturation_ratio * current_bound
            )

            if (
                looks_bound_limited
                and cap_at_natural_ceiling
                and current_bound >= natural_ceiling
            ):
                likelihood_saturated = True
                logger.warning(
                    "Posterior for gamma (mean/MAP %.3g) looks "
                    "bound-limited, but the current bound %.3g is "
                    "already at or beyond the natural numerical "
                    "ceiling %.3g for this data -- every sample's "
                    "likelihood term is already clip-saturated at "
                    "this scale, so expanding further would add a "
                    "bigger flat region for MCMC to wander, not "
                    "more signal. Treating this as a saturated "
                    '("at least this large") estimate rather than '
                    "expanding again.",
                    gamma_point,
                    current_bound,
                    natural_ceiling,
                )

            elif (
                looks_bound_limited
                and expansions < max_bound_expansions
            ):
                logger.warning(
                    "Posterior for gamma (mean/MAP %.3g) looks "
                    "bound-limited against the prior upper bound "
                    "%.3g -- expanding the bound %sx and refitting "
                    "(expansion %d/%d).",
                    gamma_point,
                    current_bound,
                    expand_factor,
                    expansions + 1,
                    max_bound_expansions,
                )
                current_bound *= expand_factor
                if cap_at_natural_ceiling:
                    current_bound = min(
                        current_bound, natural_ceiling
                    )
                expansions += 1
                continue

            elif looks_bound_limited:
                logger.warning(
                    "Posterior for gamma (mean/MAP %.3g) still looks "
                    "bound-limited against the prior upper bound "
                    "%.3g after %d expansion(s) -- this estimate may "
                    "still be capped. Consider raising "
                    "upper_bound_prior or max_bound_expansions "
                    "manually.",
                    gamma_point,
                    current_bound,
                    expansions,
                )

            rhat = ess_bulk = n_divergent = None
            if draws != 1:
                # round_to="auto" (the default) can hand back a
                # non-numeric value for some summary cells in edge
                # cases; round_to=None keeps r_hat/ess_bulk as plain
                # floats, which is all the comparisons below need.
                summary = az.summary(
                    results, var_names=["gamma"], round_to=None
                )
                try:
                    rhat = float(summary["r_hat"].item())
                    ess_bulk = float(summary["ess_bulk"].item())
                except (TypeError, ValueError):
                    logger.warning(
                        "Could not parse R-hat/ESS from az.summary() "
                        "(got %r/%r) -- treating as unconverged.",
                        summary["r_hat"].item(),
                        summary["ess_bulk"].item(),
                    )
                    rhat, ess_bulk = float("inf"), 0.0
                n_divergent = (
                    results.sample_stats["diverging"].sum().item()
                )
                converged = (
                    rhat <= rhat_threshold
                    and ess_bulk >= ess_threshold
                    and n_divergent == 0
                )

                if (
                    auto_raise_target_accept
                    and not converged
                    and convergence_retries < max_convergence_retries
                ):
                    target_accept = target_accept_schedule[
                        min(
                            convergence_retries,
                            len(target_accept_schedule) - 1,
                        )
                    ]
                    logger.warning(
                        "Poor convergence for gamma (R-hat=%.3f, "
                        "ESS=%.0f, %d divergences) -- resampling "
                        "with target_accept=%.3g (retry %d/%d).",
                        rhat,
                        ess_bulk,
                        n_divergent,
                        target_accept,
                        convergence_retries + 1,
                        max_convergence_retries,
                    )
                    current_kwargs = dict(kwargs)
                    current_kwargs["target_accept"] = target_accept
                    convergence_retries += 1
                    continue

                if auto_raise_target_accept and not converged:
                    raise ConvergenceError(
                        "Gamma sampling did not converge after "
                        f"{max_convergence_retries} target_accept "
                        f"retries (R-hat={rhat:.3f}, "
                        f"ESS={ess_bulk:.0f}, {n_divergent} "
                        "divergences). This looks like a weakly-"
                        "identified parameter (e.g. a gene whose "
                        "mutation count carries little selection "
                        "signal), not a truncated prior bound -- "
                        "raising upper_bound_prior will not fix it."
                    )

                if rhat > 1.01:
                    logger.warning(
                        f"R-hat for gamma "
                        f"is {rhat:.3f} > 1.01. "
                        "Chains may not have converged."
                    )
                if ess_bulk < 200:
                    logger.warning(
                        "Effective sample size (ESS) "
                        f"for gamma "
                        f"is {ess_bulk:.0f} < 200. Increase "
                        "draws or tune steps."
                    )
                if n_divergent > 0:
                    logger.warning(
                        f"{n_divergent} divergent "
                        "transitions were detected. "
                        "Consider reparameterizing or "
                        "increasing target_accept."
                    )

            if save_name is not None:
                if draws == 1:
                    np.savez(save_name, **results)
                else:
                    pm.save_trace(results, save_name, overwrite=True)

            if draws != 1:
                results.posterior.attrs["final_upper_bound_prior"] = (
                    current_bound
                )
                results.posterior.attrs["bound_expansions"] = (
                    expansions
                )
                results.posterior.attrs["convergence_retries"] = (
                    convergence_retries
                )
                results.posterior.attrs["final_target_accept"] = (
                    current_kwargs.get("target_accept", 0.8)
                )
                # netCDF attributes don't support a bool dtype
                # (only S1/i1/u1/.../f8) -- store as int.
                results.posterior.attrs["likelihood_saturated"] = int(
                    likelihood_saturated
                )
                results.posterior.attrs["absence_floor"] = float(
                    absence_floor if presence_model is None else 0.0
                )
                results.posterior.attrs["presence_clip"] = int(
                    presence_clip
                )
                results.posterior.attrs["shared_mu_uncertainty"] = (
                    int(bool(shared_mu_uncertainty) and use_mu_prior)
                )
                results.posterior.attrs["mu_uncertainty"] = (
                    mu_uncertainty if use_mu_prior else "none"
                )
                results.posterior.attrs["gamma_prior"] = gamma_prior
                results.posterior.attrs["natural_gamma_ceiling"] = (
                    natural_ceiling
                )
                results.posterior.attrs[
                    "n_zero_rate_absent_dropped"
                ] = n_zero_rate_absent

            return results

        except (pm.SamplingError, ValueError, KeyError) as e:
            logger.warning(
                "Sampling failed with upper bound "
                f"{current_bound:.1e}: {e}"
            )
            attempt += 1
            current_bound /= factor_of_reduction
            # A raised target_accept was tuned for the previous bound;
            # start fresh at the new (smaller) bound.
            current_kwargs = dict(kwargs)
            convergence_retries = 0

    raise RuntimeError(
        f"Sampling failed after {max_retries} attempts "
        "Try setting a smaller upper_bound_prior manually."
    )


def _estimate_gamma_on_grid(
    mu_draws,
    n_yes,
    *,
    current_bound,
    upper_bound_prior,
    natural_ceiling,
    gamma_prior,
    gene_tumor_dispersion,
    shapes_all,
    absence_floor,
    presence_clip,
    n_grid,
    draws,
    chains,
    auto_expand_bound,
    max_bound_expansions,
    expand_factor,
    saturation_ratio,
    cap_at_natural_ceiling,
    n_zero_rate_absent,
    use_mu_prior,
):
    """The exact cut (``mu_uncertainty="draws"``) of estimate_gamma_from_mus.

    Same bound handling as the MCMC path: the bound starts at the
    natural ceiling's cap, expands while the posterior mean sits above
    ``saturation_ratio`` of it, and a mean still there at the natural
    ceiling marks the result ``likelihood_saturated``.
    """
    f = _compile_draws_loglik(
        n_yes,
        gene_tumor_dispersion,
        shapes_all,
        absence_floor,
        presence_clip,
    )
    rng = np.random.default_rng(constants.random_seed)
    expansions = 0
    likelihood_saturated = False
    while True:
        lower = (
            _loguniform_lower(upper_bound_prior, current_bound)
            if gamma_prior == "loguniform"
            else current_bound * _GRID_FLOOR
        )
        grid, density = _cut_posterior_on_grid(
            f, mu_draws, lower, current_bound, gamma_prior, n_grid
        )
        values = np.exp(
            _sample_grid_density(grid, density, draws, rng)
        )
        gamma_point = float(values.mean())
        looks_bound_limited = (
            auto_expand_bound
            and gamma_point > saturation_ratio * current_bound
        )
        if (
            looks_bound_limited
            and cap_at_natural_ceiling
            and current_bound >= natural_ceiling
        ):
            likelihood_saturated = True
        elif (
            looks_bound_limited and expansions < max_bound_expansions
        ):
            current_bound *= expand_factor
            if cap_at_natural_ceiling:
                current_bound = min(current_bound, natural_ceiling)
            expansions += 1
            continue
        elif looks_bound_limited:
            logger.warning(
                "Posterior for gamma (mean %.3g) still looks "
                "bound-limited against the prior upper bound %.3g "
                "after %d expansion(s).",
                gamma_point,
                current_bound,
                expansions,
            )
        break
    results = _posterior_from_gamma_draws(values, chains)
    attrs = results.posterior.attrs
    attrs["final_upper_bound_prior"] = current_bound
    attrs["bound_expansions"] = expansions
    attrs["convergence_retries"] = 0
    attrs["likelihood_saturated"] = int(likelihood_saturated)
    attrs["absence_floor"] = float(absence_floor)
    attrs["presence_clip"] = int(presence_clip)
    attrs["shared_mu_uncertainty"] = 0
    attrs["mu_uncertainty"] = "draws" if use_mu_prior else "none"
    attrs["gamma_prior"] = gamma_prior
    attrs["grid_points"] = int(n_grid)
    attrs["grid_log_gamma_range"] = [float(grid[0]), float(grid[-1])]
    attrs["mu_draws_used"] = int(mu_draws.shape[0])
    attrs["natural_gamma_ceiling"] = natural_ceiling
    attrs["n_zero_rate_absent_dropped"] = n_zero_rate_absent
    return results


def summarize_gamma(result, prob=0.89):
    """Point estimate and interval of a gamma, or an upper bound only.

    A gamma fitted with no tumor carrying the mutation -- typical of a
    consequence channel of an oncogene, such as the predicted
    loss-of-function channel of NRAS -- is informed only by the tumors that do not carry it.
    Its posterior is then the prior cut down by the absences: it says
    how large gamma can be, not where it is, and its mean depends on
    the prior's shape. Such a gamma is reported as an upper bound
    alone: the ``(1 + prob) / 2`` quantile, the upper end of the
    equal-tailed ``prob`` interval it would otherwise have had.

    Parameters
    ----------
    result : arviz.InferenceData
        A gamma posterior. Whether it had carriers is read from
        ``posterior.attrs["n_tumors_with"]``; a posterior without
        that count (fitted before it was recorded) is summarized as
        an estimate.
    prob : float, default=0.89
        Probability mass of the equal-tailed interval.

    Returns
    -------
    dict
        ``mean``, ``lower`` and ``upper`` of gamma and
        ``upper_bound_only`` (bool). With ``upper_bound_only``,
        ``mean`` and ``lower`` are NaN and ``upper`` is the bound.
    """
    draws = np.asarray(result.posterior["gamma"]).ravel()
    tail = 100 * (1 - prob) / 2
    lower, upper = np.percentile(draws, [tail, 100 - tail])
    n_with = result.posterior.attrs.get("n_tumors_with")
    if n_with is not None and int(n_with) == 0:
        return {
            "mean": np.nan,
            "lower": np.nan,
            "upper": float(upper),
            "upper_bound_only": True,
        }
    return {
        "mean": float(draws.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "upper_bound_only": False,
    }
