"""Selection estimation for somatic variants and genes

This module provides functionality for estimating the selection
intensity parameter γ (gamma) for individual amino acid variants and
for genes. The model assumes that the probability of observing a
variant in a tumor is governed by a Poisson process with rate `γ * μ`,
where `μ` is the estimated mutation rate of the variant in that
sample.

"""

import logging
import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as tt

from . import constants

logger = logging.getLogger(__name__)

# tt.clip floor/ceiling on P(present) inside estimate_gamma_from_mus.
# Keep in sync with the tt.clip(..., _CLIP_FLOOR, 1 - _CLIP_FLOOR)
# call below: this value determines the numerical ceiling computed
# from it (see _natural_gamma_ceiling).
_CLIP_FLOOR = 1e-12

# Floor on the per-tumor sigma of the log-mu cut prior (see
# estimate_gamma_from_mus). Without it, a tumor whose posterior mu
# draws happen to agree to numerical precision would collapse that
# tumor's prior to a point mass -- silently reverting to the pre-cut
# "mu known exactly" behavior for that one tumor, rather than
# reflecting genuinely tight (but nonzero) mu uncertainty.
_MIN_LOG_MU_SIGMA = 1e-3


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


def _renamed_dispersion_args(
    dispersion, shape, old_dispersion, old_shape
):
    """Map the pre-rename ``cell_*`` arguments onto their new names."""
    for old, new, value, current in (
        (
            "cell_dispersion",
            "gene_tumor_dispersion",
            old_dispersion,
            dispersion,
        ),
        ("cell_shape", "gene_tumor_shape", old_shape, shape),
    ):
        if value is None:
            continue
        warnings.warn(
            f"{old} is deprecated; use {new}.",
            DeprecationWarning,
            stacklevel=3,
        )
        if current is not None:
            raise ValueError(f"Pass {new} only, not also {old}.")
    return (
        dispersion if old_dispersion is None else old_dispersion,
        shape if old_shape is None else old_shape,
    )


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
    cell_dispersion=None,
    cell_shape=None,
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
        rate is ``lambda_j ~ Gamma(shape=phi * p_j, rate=phi / M)``
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

    """
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
    else:
        n_yes, n_no = len(mus_yes_arr), len(mus_no_arr)
        mus_all = np.concatenate([mus_yes_arr, mus_no_arr])

    gene_tumor_dispersion, gene_tumor_shape = (
        _renamed_dispersion_args(
            gene_tumor_dispersion,
            gene_tumor_shape,
            cell_dispersion,
            cell_shape,
        )
    )
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
    current_kwargs = dict(kwargs)
    while attempt <= max_retries:
        try:
            with pm.Model():
                gamma = pm.Uniform(
                    name="gamma", lower=0, upper=current_bound
                )

                if use_mu_prior:
                    # The cut: log_mu's prior is moment-matched to
                    # the stage-1 posterior draws and is otherwise
                    # only informed by this tumor's own Bernoulli
                    # observation below -- no joint fit, no feedback
                    # from gamma/selection back into mu.
                    log_mu = pm.Normal(
                        name="log_mu",
                        mu=log_mu_mean,
                        sigma=log_mu_sigma,
                        shape=len(log_mu_mean),
                    )
                    mu = pm.math.exp(log_mu)
                else:
                    mu = mus_all

                if shapes_all is not None:
                    log_absent = -shapes_all * tt.log1p(
                        gamma * mu / shapes_all
                    )
                    Ps = tt.clip(
                        1 - tt.exp(log_absent), 1e-12, 1 - 1e-12
                    )
                elif gene_tumor_dispersion is None:
                    Ps = tt.clip(
                        1 - tt.exp(-gamma * mu), 1e-12, 1 - 1e-12
                    )
                else:
                    m_total = tt.sum(mu)
                    log_absent = (
                        -gene_tumor_dispersion
                        * (mu / m_total)
                        * tt.log1p(
                            gamma * m_total / gene_tumor_dispersion
                        )
                    )
                    Ps = tt.clip(
                        1 - tt.exp(log_absent), 1e-12, 1 - 1e-12
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
                results.posterior.attrs["natural_gamma_ceiling"] = (
                    natural_ceiling
                )

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
