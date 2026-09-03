"""Per-gene rate correction ``r_g``, shared across consequence channels.

The unified model this implements::

    r_g            ~ Gamma(θ, 1/θ)                    [one per gene]
    μ_g^(syn,j)    = μ̄_g^(syn,j)    × exp(c·x_g) × r_g
    μ_g^(nonsyn,j) = μ̄_g^(nonsyn,j) × exp(c·x_g) × r_g
    L = ∏_{g∈G} ∏_j Poisson(N_g^(syn,j);    μ_g^(syn,j))
      × ∏_{g∈P} ∏_j Poisson(N_g^(nonsyn,j); μ_g^(nonsyn,j))

``G`` is every gene (silent channel, drivers included), ``P`` the
passenger genes (non-synonymous channel). ``r_g`` is **shared** across
the two channels, and that sharing is the mechanism, not a
simplification: a driver gene's non-synonymous counts are
selection-contaminated by construction, so they could never identify a
channel-specific ``r_g``; the shared one lets the driver's *silent*
counts — clean and neutral in any gene — correct its own local rate,
which then also scales its non-synonymous prediction.

`r_g` is marginalized out analytically
--------------------------------------
Given ``r_g``, gene ``g``'s Poisson terms depend on the data only
through the per-gene totals ``S_g = Σ_j N_g^j`` and
``A_g = Σ_j μ̄_g^j``, so the Gamma integral has a closed form::

    log ∫ ∏_j Poisson(N_j; m_j r) Gamma(r; θ, 1/θ) dr
        = Σ_j [N_j log m_j - log N_j!]
          + lnΓ(θ + S) - lnΓ(θ) + θ log θ - (θ + S) log(θ + M)

with ``m_j = μ̄_j e^{c·x}``, ``S = Σ_j N_j``, ``M = Σ_j m_j``. (Verified
against numerical integration to ~1e-14; see
``tests/test_estimate_rg.py``.) This is the Negative Binomial the deck
describes — i.e. dNdScv's NB regression — done jointly and Bayesianly
on top of our per-tumor ``μ̄_g^j``, which dNdScv never had.

Three things follow, all of them practical:

1. The fit is over ``(c, θ)`` only. No ~17,000 latent ``r_g``
   parameters, and no MAP pathology from a Gamma prior whose density
   diverges at 0 when ``θ < 1``.
2. The whole likelihood reduces to four per-gene vectors
   (``S^(syn)``, ``S^(nonsyn)``, ``A^(syn)``, ``A^(nonsyn)``) rather
   than genes × samples matrices.
3. ``r_g``'s posterior is exactly ``Gamma(θ + S_g, θ + M_g)``, so its
   mean ``(θ + S_g)/(θ + M_g)`` is a closed form rather than a
   sampling problem — which is what lets the production and evaluation
   variants below be two small, obviously-different functions.

Production vs evaluation
------------------------
These are **separate functions with different signatures**, not one
function with a flag, because getting them confused would silently
reproduce the leakage that killed the first ``r_g`` attempt:

* :func:`r_g_production` takes both channels' statistics — the fully
  Bayesian use of everything known about a gene. These are the
  numbers to publish.
* :func:`r_g_silent_only_for_evaluation` takes **only** the silent
  channel's, and cannot be handed non-silent data at all: there is no
  argument for it. Scoring the resulting ``μ^(nonsyn)`` against
  held-out non-silent counts therefore cannot be predicting the
  target from the target.

Both take ``theta`` and the covariate-scaled expectations from the
same joint fit, so they differ *only* in which channel informs each
gene's own correction.
"""

import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc.sampling.jax as pmjax
import pytensor.tensor as tt

from . import constants

logger = logging.getLogger(__name__)


def r_g_production(
    counts_silent: pd.Series,
    counts_non_silent: pd.Series,
    expected_silent: pd.Series,
    expected_non_silent: pd.Series,
    theta: float,
) -> pd.Series:
    """Posterior mean of ``r_g`` using **both** channels (production).

    The number to publish: each gene's rate correction informed by
    every mutation observed in it, silent and non-silent alike. This
    is not leakage, it is the model using what it has -- but it must
    never be the ``r_g`` behind a reported R², which is what
    :func:`r_g_silent_only_for_evaluation` is for.

    Parameters
    ----------
    counts_silent, counts_non_silent : pandas.Series
        Per-gene observed totals ``Σ_j N_g^j`` for each channel,
        summed over exactly the samples the fit used. Genes outside a
        channel's gene set contribute 0.
    expected_silent, expected_non_silent : pandas.Series
        Per-gene covariate-scaled expectations ``Σ_j μ̄_g^j e^{c·x_g}``
        for each channel, over the same samples.
    theta : float
        The fitted Gamma shape, shared across genes.

    Returns
    -------
    pandas.Series
        ``(θ + S_g) / (θ + M_g)``, indexed like the inputs. Mean 1
        under the prior; > 1 means the gene mutates faster than its
        covariates predict.
    """
    counts = counts_silent.add(counts_non_silent, fill_value=0.0)
    expected = expected_silent.add(
        expected_non_silent, fill_value=0.0
    )
    return (theta + counts) / (theta + expected)


def r_g_silent_only_for_evaluation(
    counts_silent: pd.Series,
    expected_silent: pd.Series,
    theta: float,
) -> pd.Series:
    """Posterior mean of ``r_g`` from the silent channel **only**.

    The evaluation-discipline variant: this function has no argument
    through which non-silent data could reach it, so a ``μ^(nonsyn)``
    scaled by its output and scored against non-silent counts is
    scored on a target it has never seen. Same discipline as
    ``dnds_comparison``'s closed-form leakage control, Bayesian
    instead.

    ``theta`` and ``expected_silent`` still come from the joint fit
    (same ``c``, same ``θ``); only which channel informs *this gene's
    own* correction changes.

    Parameters
    ----------
    counts_silent : pandas.Series
        Per-gene observed silent totals ``Σ_j N_g^(syn,j)``.
    expected_silent : pandas.Series
        Per-gene silent expectations ``Σ_j μ̄_g^(syn,j) e^{c·x_g}``.
    theta : float
        The fitted Gamma shape, shared across genes.

    Returns
    -------
    pandas.Series
        ``(θ + S_g^(syn)) / (θ + M_g^(syn))``, indexed like the inputs.
    """
    return (theta + counts_silent) / (theta + expected_silent)


def channel_rg_log_likelihood(
    eta,
    theta,
    counts_silent,
    counts_non_silent,
    baseline_silent,
    baseline_non_silent,
):
    """Marginal log-likelihood of the two channels with ``r_g`` integrated out.

    Written against pytensor tensors (for the PyMC model) but valid
    for plain numpy arrays too, which is how the tests check it
    against numerical integration.

    All six arguments are aligned, per-gene, over the **silent**
    channel's gene set ``G``. A gene outside the non-synonymous set
    ``P`` carries ``counts_non_silent = 0`` and
    ``baseline_non_silent = 0``, which drops its non-synonymous term
    exactly.

    The ``Σ_j [N log μ̄ - log N!]`` part of the Poisson terms does not
    depend on ``c`` or ``θ`` and is omitted -- an additive constant
    that shifts the objective without moving its argmax.
    """
    scale = tt.exp(eta) if hasattr(eta, "type") else np.exp(eta)
    expected_silent = scale * baseline_silent
    expected_non_silent = scale * baseline_non_silent

    counts = counts_silent + counts_non_silent
    expected = expected_silent + expected_non_silent

    gammaln = tt.gammaln if hasattr(eta, "type") else _np_gammaln
    log = tt.log if hasattr(eta, "type") else np.log

    return (counts * eta).sum() + (
        gammaln(theta + counts)
        - gammaln(theta)
        + theta * log(theta)
        - (theta + counts) * log(theta + expected)
    ).sum()


def _np_gammaln(x):
    from scipy.special import gammaln

    return gammaln(x)


def estimate_channel_rg_effect(
    counts_silent: np.ndarray,
    baseline_silent: np.ndarray,
    counts_non_silent: np.ndarray,
    baseline_non_silent: np.ndarray,
    cov_matrix: np.ndarray,
    draws: int = 4000,
    lower_bounds_c: float | np.ndarray | None = -2,
    upper_bounds_c: float | np.ndarray = 2,
    log_theta_bounds: tuple[float, float] = (-5.0, 10.0),
    burn: int = 1000,
    chains: int = 4,
    save_path: str | Path | None = None,
    kwargs: dict | None = None,
) -> az.InferenceData | dict:
    """Fit shared ``c`` and ``θ`` with ``r_g`` marginalized out.

    All arrays are per-gene and aligned over the silent channel's gene
    set ``G`` (see :func:`channel_rg_log_likelihood`): pass zeros in
    ``counts_non_silent``/``baseline_non_silent`` for the genes
    excluded from the non-synonymous channel, which is how driver
    genes enter through their silent channel alone.

    Parameters
    ----------
    counts_silent, counts_non_silent : ndarray, shape (n_genes,)
        Per-gene observed totals for each channel.
    baseline_silent, baseline_non_silent : ndarray, shape (n_genes,)
        Per-gene ``Σ_j μ̄_g^j`` for each channel, **before** the
        covariate scaling (which the fit applies).
    cov_matrix : ndarray, shape (n_genes, n_covariates)
        Gene covariates; a column of ones is prepended for the
        intercept.
    draws, lower_bounds_c, upper_bounds_c, burn, chains, save_path, kwargs
        As in :func:`estimate_covariates_effect.estimate_covariates_effect`.
    log_theta_bounds : (float, float), default (-5, 10)
        Uniform prior bounds on ``log θ``. Fitted in log space because
        θ spans orders of magnitude across cohorts (dNdScv reports
        values from ~1 to ~250 on this data). The default range,
        θ ∈ [0.0067, 22026], brackets that comfortably; a fitted θ at
        either bound means the data wants no overdispersion at all
        (upper) or unbounded overdispersion (lower), and is worth
        investigating rather than reporting.

    Returns
    -------
    arviz.InferenceData | dict
        As elsewhere: a dict with keys ``c`` and ``log_theta`` for
        ``draws == 1`` (MAP), otherwise posterior samples.

    Notes
    -----
    θ is a single per-cohort hyperparameter. Subdividing it (say by
    gene-length bucket) is deliberately *not* offered as a first pass:
    each cohort's fit already has thousands of genes to estimate one θ
    from, and subdividing adds identifiability risk with no evidence
    yet that it is needed.
    """
    if kwargs is None:
        kwargs = {}

    n_genes = counts_silent.shape[0]
    for name, arr in (
        ("baseline_silent", baseline_silent),
        ("counts_non_silent", counts_non_silent),
        ("baseline_non_silent", baseline_non_silent),
    ):
        if arr.shape[0] != n_genes:
            raise ValueError(
                f"{name} has {arr.shape[0]} genes but counts_silent "
                f"has {n_genes}; all per-gene arrays must be aligned "
                "over the silent channel's gene set."
            )
    if cov_matrix.shape[0] != n_genes:
        raise ValueError(
            f"cov_matrix has {cov_matrix.shape[0]} rows but there "
            f"are {n_genes} genes."
        )

    n_in_non_silent = int((baseline_non_silent > 0).sum())
    logger.info(
        f"r_g mode: {n_genes} genes in the silent channel, "
        f"{n_in_non_silent} also in the non-silent channel, "
        f"{cov_matrix.shape[1]} covariate(s)"
    )

    ones = np.ones((n_genes, 1), dtype="float64")
    cov_ext = np.concatenate(
        [ones, np.asarray(cov_matrix, dtype="float64")], axis=1
    )
    n_coeffs = cov_ext.shape[1]

    if lower_bounds_c is None:
        lower_bounds_c = -upper_bounds_c

    with pm.Model():
        c = pm.Uniform(
            name="c",
            lower=lower_bounds_c,
            upper=upper_bounds_c,
            shape=n_coeffs,
        )
        log_theta = pm.Uniform(
            name="log_theta",
            lower=log_theta_bounds[0],
            upper=log_theta_bounds[1],
        )
        theta = tt.exp(log_theta)

        cov32 = pm.Data("cov_ext", cov_ext)
        eta = tt.dot(cov32, c)

        pm.Potential(
            "channel_rg_marginal",
            channel_rg_log_likelihood(
                eta=eta,
                theta=theta,
                counts_silent=pm.Data(
                    "counts_silent", counts_silent.astype("float64")
                ),
                counts_non_silent=pm.Data(
                    "counts_non_silent",
                    counts_non_silent.astype("float64"),
                ),
                baseline_silent=pm.Data(
                    "baseline_silent",
                    np.clip(
                        baseline_silent.astype("float64"),
                        1e-12,
                        np.inf,
                    ),
                ),
                baseline_non_silent=pm.Data(
                    "baseline_non_silent",
                    baseline_non_silent.astype("float64"),
                ),
            ),
        )

        if draws == 1:
            logger.info(
                f"Finding MAP estimate for {n_coeffs} coefficient(s) "
                "plus log_theta"
            )
            results = pm.find_MAP(
                seed=constants.random_seed, **kwargs
            )
            logger.info("MAP optimization completed")
        else:
            logger.info(
                f"Sampling posterior: {draws} draws across "
                f"{chains} chains ({int(draws / chains)} per chain), "
                f"{burn} tuning steps"
            )
            results = pmjax.sample_numpyro_nuts(
                draws=int(draws / chains),
                chain_method="parallel",
                tune=burn,
                chains=chains,
                target_accept=0.9,
                random_seed=constants.random_seed,
                **kwargs,
            )
            logger.info("MCMC sampling completed")

    if save_path is not None:
        base_path = Path(save_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        if draws == 1:
            np.savez(f"{base_path}.npz", **results)
        else:
            results.to_netcdf(f"{base_path}.nc")

    return results
