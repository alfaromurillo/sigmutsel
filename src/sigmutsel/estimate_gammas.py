"""Selection estimation for somatic variants and genes

This module provides functionality for estimating the selection
intensity parameter γ (gamma) for individual amino acid variants and
for genes. The model assumes that the probability of observing a
variant in a tumor is governed by a Poisson process with rate `γ * μ`,
where `μ` is the estimated mutation rate of the variant in that
sample.

"""

import numpy as np

import pymc as pm
import pytensor.tensor as tt
import arviz as az

import logging

from . import constants

logger = logging.getLogger(__name__)


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

    Returns
    -------
    results : arviz.InferenceData or dict
        Posterior samples (or MAP/MLE estimate) of gamma. For MCMC
        results, ``results.posterior.attrs`` records
        ``final_upper_bound_prior`` and ``bound_expansions`` so
        callers can tell whether (and how much) the bound had to be
        expanded.

    Raises
    ------
    RuntimeError
        If sampling fails after all retries.

    Notes
    -----
    Uses ``sigmutsel.constants.random_seed`` for reproducibility.
    Left unset (``None``) by default; set it explicitly (e.g. in the
    calling application, before running the pipeline) to make
    sampling reproducible across runs and machines.

    """
    mus_all = np.concatenate([mus_yes, mus_no])

    if kwargs is None:
        kwargs = {}

    attempt = 0
    expansions = 0
    current_bound = upper_bound_prior
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
                        seed=constants.random_seed, **kwargs
                    )
                else:
                    results = pm.sample(
                        draws=int(draws / chains),
                        chains=chains,
                        tune=burn,
                        random_seed=constants.random_seed,
                        **kwargs,
                    )

            if draws == 1:
                gamma_point = float(results["gamma"])
            else:
                gamma_point = float(
                    results.posterior["gamma"].values.mean()
                )

            if (
                auto_expand_bound
                and expansions < max_bound_expansions
                and gamma_point > saturation_ratio * current_bound
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
                expansions += 1
                continue

            if (
                auto_expand_bound
                and expansions >= max_bound_expansions
                and gamma_point > saturation_ratio * current_bound
            ):
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

            if save_name is not None:
                if draws == 1:
                    np.savez(save_name, **results)
                else:
                    pm.save_trace(results, save_name, overwrite=True)

                    # Warnings for convergence
                    summary = az.summary(results, var_names=["gamma"])
                    rhat = summary["r_hat"].item()
                    ess_bulk = summary["ess_bulk"].item()

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

                    # Check for divergences
                    n_divergent = (
                        results.sample_stats["diverging"].sum().item()
                    )
                    if n_divergent > 0:
                        logger.warning(
                            f"{n_divergent} divergent "
                            "transitions were detected. "
                            "Consider reparameterizing or "
                            "increasing target_accept."
                        )

            if draws != 1:
                results.posterior.attrs["final_upper_bound_prior"] = (
                    current_bound
                )
                results.posterior.attrs["bound_expansions"] = (
                    expansions
                )

            return results

        except (pm.SamplingError, ValueError, KeyError) as e:
            logger.warning(
                "Sampling failed with upper bound "
                f"{current_bound:.1e}: {e}"
            )
            attempt += 1
            current_bound /= factor_of_reduction

    raise RuntimeError(
        f"Sampling failed after {max_retries} attempts "
        "Try setting a smaller upper_bound_prior manually."
    )
