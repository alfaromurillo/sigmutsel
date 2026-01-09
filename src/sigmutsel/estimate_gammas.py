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

from .constants import random_seed

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

    random_seed : int, default=777
        Seed used to control randomness for reproducibility.

    kwargs : dict or None, default=None
        Additional arguments to pass to :func:`pymc.sample` or
        :func:`pymc.find_MAP`.

    max_retries : int, default=5
        Maximum number of retries with reduced prior bound.

    factor_of_reduction : float, default=10
        Factor by which to reduce the prior upper bound if sampling fails.

    Returns
    -------
    results : arviz.InferenceData or dict
        Posterior samples (or MAP/MLE estimate) of gamma.

    Raises
    ------
    RuntimeError
        If sampling fails after all retries.

    """
    mus_all = np.concatenate([mus_yes, mus_no])

    if kwargs is None:
        kwargs = {}

    attempt = 0
    while attempt <= max_retries:
        try:
            with pm.Model():
                gamma = pm.Uniform(
                    name="gamma", lower=0, upper=upper_bound_prior
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
                    results = pm.find_MAP(**kwargs)
                else:
                    results = pm.sample(
                        draws=int(draws / chains),
                        chains=chains,
                        tune=burn,
                        random_seed=random_seed,
                        **kwargs,
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

            return results

        except (pm.SamplingError, ValueError, KeyError) as e:
            logger.warning(
                "Sampling failed with upper bound "
                f"{upper_bound_prior:.1e}: {e}"
            )
            attempt += 1
            upper_bound_prior /= factor_of_reduction

    raise RuntimeError(
        f"Sampling failed after {max_retries} attempts "
        "Try setting a smaller upper_bound_prior manually."
    )
