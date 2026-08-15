"""Selection estimation for somatic variants and genes

This module provides functionality for estimating the selection
intensity parameter γ (gamma) for individual amino acid variants and
for genes. The model assumes that the probability of observing a
variant in a tumor is governed by a Poisson process with rate `γ * μ`,
where `μ` is the estimated mutation rate of the variant in that
sample.

"""

import logging

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


class ConvergenceError(RuntimeError):
    """MCMC sampling completed but convergence diagnostics are bad.

    Distinguishes a fit that ran without crashing but produced an
    untrustworthy posterior (high R-hat, low ESS, divergences) from
    an actual sampling crash (``pm.SamplingError`` etc.). Subclasses
    ``RuntimeError`` so existing giant-non-driver-gene skip logic
    (which catches ``RuntimeError``) still catches this too.
    """


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
        Array of mu values for tumors where with the variant(s).

    mus_no : array-like
        Array of mu values for tumors where without the variant(s).

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

    """
    mus_all = np.concatenate([mus_yes, mus_no])
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

                Ps = tt.clip(
                    1 - tt.exp(-gamma * mus_all), 1e-12, 1 - 1e-12
                )

                pm.Bernoulli(
                    name="variants_observed",
                    p=Ps,
                    observed=np.concatenate(
                        [np.ones(len(mus_yes)), np.zeros(len(mus_no))]
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
                summary = az.summary(results, var_names=["gamma"])
                rhat = summary["r_hat"].item()
                ess_bulk = summary["ess_bulk"].item()
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
                results.posterior.attrs["likelihood_saturated"] = (
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
