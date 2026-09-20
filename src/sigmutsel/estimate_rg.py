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

Restrictions of this one likelihood
-----------------------------------
This model is the top of a nested ladder, and the rungs below it are
reached by *removing* parameters from the same likelihood rather than
by writing a second model -- which is the only way the difference
between two rungs can be read as "what this addition buys" instead of
"what these two implementations disagree about".

``fit_rg=False`` drops the Gamma marginalisation and leaves the plain
two-channel Poisson (``theta=None`` in
:func:`channel_rg_log_likelihood`); ``fit_intercept=False`` pins the
shared intercept at 0 rather than fitting it, which a zero-column
``cov_matrix`` does **not** do, since the ones column is prepended
unconditionally; and ``use_silent_channel=False`` drops the silent
channel's terms, leaving a single-channel Poisson GLM with offset
``log mu_bar^(nonsyn)``. See each argument's entry on
:func:`estimate_channel_rg_effect`.
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


def r_g_draws_production(
    counts_silent: pd.Series,
    counts_non_silent: pd.Series,
    expected_silent: pd.Series,
    expected_non_silent: pd.Series,
    theta: float,
    n_draws: int,
    rng=None,
) -> pd.DataFrame:
    """Draws from ``r_g``'s exact posterior, **both** channels.

    ``r_g``'s posterior given ``θ``, ``S_g`` and ``M_g`` is exactly
    ``Gamma(θ + S_g, θ + M_g)`` (shape, rate) -- see the module
    docstring -- so these are drawn directly, no MCMC needed. Same
    production-vs-evaluation split as :func:`r_g_production` vs.
    :func:`r_g_silent_only_for_evaluation`: this variant partly
    absorbs selection itself (it is informed by the same non-silent
    counts gamma is estimated from), so feeding it into
    :func:`.estimate_gammas.estimate_gamma_from_mus`'s mu posterior
    would bias gamma downward. Use
    :func:`r_g_draws_for_evaluation` instead for that purpose.

    Parameters
    ----------
    counts_silent, counts_non_silent, expected_silent,
    expected_non_silent : pandas.Series
        As in :func:`r_g_production`.
    theta : float
        The fitted Gamma shape, shared across genes.
    n_draws : int
        Number of draws per gene.
    rng : numpy.random.Generator or None, default None
        Source of randomness. A fresh default_rng() if None.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_draws, n_genes)``, columns indexed like the
        inputs.
    """
    counts = counts_silent.add(counts_non_silent, fill_value=0.0)
    expected = expected_silent.add(
        expected_non_silent, fill_value=0.0
    )
    return _r_g_gamma_draws(counts, expected, theta, n_draws, rng)


def r_g_draws_for_evaluation(
    counts_silent: pd.Series,
    expected_silent: pd.Series,
    theta: float,
    n_draws: int,
    rng=None,
) -> pd.DataFrame:
    """Draws from ``r_g``'s exact posterior, silent channel **only**.

    The evaluation-discipline variant of :func:`r_g_draws_production`
    -- see :func:`r_g_silent_only_for_evaluation` for why. This is
    the defensible choice (along with ``r_g_variant="none"``) for
    feeding gamma's mu posterior cut; the production variant is not,
    since it would bias gamma downward.

    Parameters
    ----------
    counts_silent, expected_silent : pandas.Series
        As in :func:`r_g_silent_only_for_evaluation`.
    theta : float
        The fitted Gamma shape, shared across genes.
    n_draws : int
        Number of draws per gene.
    rng : numpy.random.Generator or None, default None
        Source of randomness. A fresh default_rng() if None.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_draws, n_genes)``, columns indexed like
        ``counts_silent``.
    """
    return _r_g_gamma_draws(
        counts_silent, expected_silent, theta, n_draws, rng
    )


def _r_g_gamma_draws(counts, expected, theta, n_draws, rng):
    if rng is None:
        rng = np.random.default_rng()
    shape = (theta + counts).to_numpy()
    rate = (theta + expected).to_numpy()
    draws = rng.gamma(shape, 1.0 / rate, size=(n_draws, len(shape)))
    return pd.DataFrame(draws, columns=counts.index)


def channel_rg_log_likelihood(
    eta_silent,
    theta,
    counts_silent,
    counts_non_silent,
    baseline_silent,
    baseline_non_silent,
    eta_non_silent=None,
    use_silent=True,
):
    """Marginal log-likelihood of the two channels with ``r_g`` integrated out.

    Written against pytensor tensors (for the PyMC model) but valid
    for plain numpy arrays too, which is how the tests check it
    against numerical integration.

    All per-gene arguments are aligned over the **silent** channel's
    gene set ``G``. A gene outside the non-synonymous set ``P``
    carries ``counts_non_silent = 0`` and ``baseline_non_silent =
    0``, which drops its non-synonymous term exactly.

    ``eta_non_silent`` defaults to ``eta_silent``, which is the
    shared-``c`` model. Passing a different linear predictor gives
    the separate-``c`` model, in which the two channels get their
    own covariate coefficients while still sharing ``r_g`` and
    ``θ``. The shared model is nested inside the separate one at
    ``c^(syn) = c^(nonsyn)``, so the two log-likelihoods are
    directly comparable and their difference is a likelihood-ratio
    statistic on ``n_coeffs`` degrees of freedom.

    Note ``r_g`` stays shared even when ``c`` is not: only the
    covariate-driven part of the rate is allowed to differ by
    consequence, and the per-gene rate correction still has to be
    identified from the silent channel for driver genes.

    ``theta = None`` drops ``r_g`` altogether and leaves the plain
    two-channel Poisson,
    ``Σ nᵍʸⁿ ηˢʸⁿ + Σ nⁿᵒⁿ ηⁿᵒⁿ - Σ (μ̄ˢʸⁿ e^ηˢʸⁿ + μ̄ⁿᵒⁿ e^ηⁿᵒⁿ)``.
    That is the ``θ → ∞`` limit of the expression below, written
    out rather than approached: ``lnGamma(θ+n) - lnGamma(θ) → n log θ``
    and ``-(θ+n) log(θ+M) → -n log θ - M``, so the ``log θ`` terms
    cancel and ``-M`` is what survives. Arms 0-3 of the nested ladder
    are this restriction; arm 4 is the full expression.

    ``use_silent = False`` drops the **silent** channel's terms
    entirely -- not by zeroing ``baseline_silent`` (which
    :func:`estimate_channel_rg_effect` clips away from 0 for the log),
    but by leaving them out of ``counts``, ``expected`` and
    ``linear``. Combined with ``theta = None`` and a gene set
    restricted to ``P`` this is a single-channel Poisson GLM with log
    link and offset ``log μ̄ⁿᵒⁿ_g`` -- the ladder's 1p/2ap arms, which
    are December's data (non-synonymous, passengers) under a Poisson
    rather than a Bernoulli. Their likelihood is over different data
    from the two-channel arms', so the two are **not** comparable as
    likelihoods; compare them on a common held-out target instead.

    The ``Σ_j [N log μ̄ - log N!]`` part of the Poisson terms does not
    depend on ``c`` or ``θ`` and is omitted -- an additive constant
    that shifts the objective without moving its argmax. It is the
    same constant for every arm that keeps the same gene sets and
    channels, which is what makes arms 0-4 directly comparable.
    """
    if eta_non_silent is None:
        eta_non_silent = eta_silent

    is_tensor = hasattr(eta_silent, "type")
    exp = tt.exp if is_tensor else np.exp
    gammaln = tt.gammaln if is_tensor else _np_gammaln
    log = tt.log if is_tensor else np.log

    expected_non_silent = exp(eta_non_silent) * baseline_non_silent
    linear = (counts_non_silent * eta_non_silent).sum()

    if use_silent:
        expected_silent = exp(eta_silent) * baseline_silent
        counts = counts_silent + counts_non_silent
        expected = expected_silent + expected_non_silent
        linear = linear + (counts_silent * eta_silent).sum()
    else:
        counts = counts_non_silent
        expected = expected_non_silent

    if theta is None:
        return linear - expected.sum()

    return (
        linear
        + (
            gammaln(theta + counts)
            - gammaln(theta)
            + theta * log(theta)
            - (theta + counts) * log(theta + expected)
        ).sum()
    )


def _c_mode_label(separate_c):
    """Human-readable name for the three nested c parameterisations."""
    if separate_c is True:
        return "separate c"
    if separate_c == "intercept":
        return "separate intercept, shared slopes"
    return "shared c"


def _np_gammaln(x):
    from scipy.special import gammaln

    return gammaln(x)


def _drop_intercept_bound(bound, n_coeffs):
    """Drop a per-coefficient bound's intercept entry.

    ``lower_bounds_c``/``upper_bounds_c`` may be a scalar (one bound
    for every coefficient, nothing to drop) or an array sized for the
    full ``c`` including the prepended intercept. Only the latter
    needs trimming when the intercept is pinned rather than fitted.
    """
    arr = np.asarray(bound)
    if arr.ndim == 0:
        return bound
    if arr.shape[-1] != n_coeffs:
        raise ValueError(
            f"A per-coefficient bound must have {n_coeffs} entries "
            f"(intercept included); got shape {arr.shape}."
        )
    return arr[..., 1:]


def estimate_channel_rg_effect(
    counts_silent: np.ndarray,
    baseline_silent: np.ndarray,
    counts_non_silent: np.ndarray,
    baseline_non_silent: np.ndarray,
    cov_matrix: np.ndarray,
    separate_c: bool = False,
    draws: int = 4000,
    lower_bounds_c: float | np.ndarray | None = -2,
    upper_bounds_c: float | np.ndarray = 2,
    log_theta_bounds: tuple[float, float] = (-5.0, 10.0),
    burn: int = 1000,
    chains: int = 4,
    fit_rg: bool = True,
    fit_intercept: bool = True,
    use_silent_channel: bool = True,
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
    separate_c : bool | str, default False
        Three nested models, in increasing order of freedom:

        * ``False`` -- both channels share one coefficient vector.
        * ``"intercept"`` -- shared slopes, but the non-synonymous
          channel gets its own intercept (one extra parameter,
          returned as ``delta_intercept``).
        * ``True`` -- each channel gets its own full vector; ``c``
          comes back with shape ``(2, n_coeffs)``, row 0 synonymous.

        ``r_g`` and ``θ`` stay shared in all three. The ladder
        matters because the intercept absorbs a *calibration*
        offset between the two channels -- any systematic mismatch
        in the opportunity split lands there -- while the slopes
        carry the biological question. Comparing ``True`` against
        ``"intercept"`` tests the slopes alone; comparing
        ``"intercept"`` against ``False`` tests only the offset. A
        test of ``True`` against ``False`` conflates the two and
        will look overwhelmingly significant on the strength of the
        offset.
    draws, lower_bounds_c, upper_bounds_c, burn, chains, save_path, kwargs
        As in :func:`estimate_covariates_effect.estimate_covariates_effect`.
    fit_rg : bool, default True
        ``False`` drops ``r_g`` and ``θ`` and fits the plain
        two-channel Poisson (``theta=None`` in
        :func:`channel_rg_log_likelihood`). The returned dict/posterior
        then has **no** ``log_theta`` entry. Ladder arms 0-3.
    fit_intercept : bool, default True
        ``False`` pins the intercept at 0 instead of fitting it: the
        ones column is still prepended (so ``c`` always comes back
        with ``1 + n_covariates`` entries and every downstream
        consumer of ``cov_effects`` keeps working unchanged), but its
        coefficient is held at 0 rather than given a prior. This is
        the only way to reach ladder arm 0 -- ``μ = μ̄`` with nothing
        fitted -- since a zero-column ``cov_matrix`` still leaves the
        prepended intercept free and so gives arm 1, not arm 0.
        With no covariate columns, no ``δ`` and ``fit_rg=False`` there
        is nothing left to fit at all; the function then skips PyMC
        entirely and returns the fixed ``c = 0`` vector, which is the
        honest representation of "nothing fitted" and avoids
        ``find_MAP`` failing on a model with no free variables.
    use_silent_channel : bool, default True
        ``False`` drops the silent channel from the likelihood
        (:func:`channel_rg_log_likelihood`'s ``use_silent``). Pass a
        gene set already restricted to ``P`` with it -- the caller
        owns the gene set, this argument only owns the likelihood.
        With ``fit_rg=False`` this is the ladder's 1p/2ap arms: a
        single-channel Poisson GLM with offset
        ``log μ̄ⁿᵒⁿ_g``. ``separate_c`` must be ``False`` here, since
        ``δ`` is an offset *between* channels and only one is left.
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
        ``log_theta`` is absent when ``fit_rg=False``, and a dict is
        returned regardless of ``draws`` when there is nothing to fit
        (see ``fit_intercept``).

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

    if separate_c not in (True, False, "intercept"):
        raise ValueError(
            f"separate_c must be False, True or 'intercept'; got "
            f"{separate_c!r}."
        )
    if not use_silent_channel and separate_c is not False:
        raise ValueError(
            "separate_c is an offset between the two channels; with "
            "use_silent_channel=False there is only one channel, so "
            f"separate_c must be False (got {separate_c!r})."
        )

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
        f"r_g mode ({_c_mode_label(separate_c)}): "
        f"{n_genes} genes in the "
        f"{'silent' if use_silent_channel else 'non-silent'} channel, "
        f"{n_in_non_silent} in the non-silent channel, "
        f"{cov_matrix.shape[1]} covariate(s), "
        f"r_g {'fitted' if fit_rg else 'off'}, "
        f"intercept {'fitted' if fit_intercept else 'pinned at 0'}"
    )

    ones = np.ones((n_genes, 1), dtype="float64")
    cov_ext = np.concatenate(
        [ones, np.asarray(cov_matrix, dtype="float64")], axis=1
    )
    n_coeffs = cov_ext.shape[1]

    if lower_bounds_c is None:
        lower_bounds_c = -upper_bounds_c

    # The ones column stays in `cov_ext` even when the intercept is
    # not fitted, so `c` keeps its `1 + n_covariates` length for every
    # arm and nothing downstream has to know which arm produced it;
    # only the *prior* is withheld. An array-valued bound was sized
    # for the full `c`, so drop its intercept entry to match.
    n_free_c = n_coeffs if fit_intercept else n_coeffs - 1
    if not fit_intercept:
        lower_bounds_c = _drop_intercept_bound(
            lower_bounds_c, n_coeffs
        )
        upper_bounds_c = _drop_intercept_bound(
            upper_bounds_c, n_coeffs
        )

    n_free = n_free_c * (2 if separate_c is True else 1)
    n_free += 1 if separate_c == "intercept" else 0
    n_free += 1 if fit_rg else 0
    if n_free == 0:
        # Ladder arm 0: mu = mu_bar, nothing fitted. There is no
        # optimisation to run and no posterior to sample -- find_MAP
        # on a model with no free variables raises -- so return the
        # fixed coefficient vector directly. `draws` is ignored on
        # purpose: the "posterior" of a model with no parameters is a
        # point mass, and pretending otherwise would invite a caller
        # to average over it.
        logger.info(
            "No free parameters (ladder arm 0): returning the fixed "
            "c = 0 vector without fitting."
        )
        results = {"c": np.zeros(n_coeffs, dtype="float64")}
        if save_path is not None:
            base_path = Path(save_path)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f"{base_path}.npz", **results)
        return results

    with pm.Model():
        if fit_intercept:
            c = pm.Uniform(
                name="c",
                lower=lower_bounds_c,
                upper=upper_bounds_c,
                shape=(
                    (2, n_coeffs) if separate_c is True else n_coeffs
                ),
            )
        elif n_free_c == 0:
            # No covariates and no fitted intercept: `c` is the zero
            # vector, but something else (delta or theta) is still
            # free, so the PyMC model is not degenerate.
            c = pm.Deterministic(
                "c",
                tt.zeros(
                    (2, n_coeffs) if separate_c is True else n_coeffs
                ),
            )
        else:
            c_slopes = pm.Uniform(
                name="c_slopes",
                lower=lower_bounds_c,
                upper=upper_bounds_c,
                shape=(
                    (2, n_free_c) if separate_c is True else n_free_c
                ),
            )
            c = pm.Deterministic(
                "c",
                tt.concatenate(
                    [
                        tt.zeros(
                            (2, 1) if separate_c is True else (1,)
                        ),
                        c_slopes,
                    ],
                    axis=-1,
                ),
            )
        if fit_rg:
            log_theta = pm.Uniform(
                name="log_theta",
                lower=log_theta_bounds[0],
                upper=log_theta_bounds[1],
            )
            theta = tt.exp(log_theta)
        else:
            theta = None

        cov32 = pm.Data("cov_ext", cov_ext)
        if separate_c is True:
            eta_silent = tt.dot(cov32, c[0])
            eta_non_silent = tt.dot(cov32, c[1])
        elif separate_c == "intercept":
            delta = pm.Uniform(
                name="delta_intercept",
                lower=lower_bounds_c,
                upper=upper_bounds_c,
            )
            eta_silent = tt.dot(cov32, c)
            eta_non_silent = eta_silent + delta
        else:
            eta_silent = tt.dot(cov32, c)
            eta_non_silent = None

        pm.Potential(
            "channel_rg_marginal",
            channel_rg_log_likelihood(
                eta_silent=eta_silent,
                eta_non_silent=eta_non_silent,
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
                use_silent=use_silent_channel,
            ),
        )

        if draws == 1:
            logger.info(
                f"Finding MAP estimate for {n_free} free "
                "parameter(s)"
            )
            results, opt = pm.find_MAP(
                seed=constants.random_seed,
                return_raw=True,
                **kwargs,
            )
            # The MAP analogue of r_hat. A MAP fit has no chains, so
            # no convergence diagnostic was ever recorded for it --
            # `pca_nc_production_sweep.py` selected every cohort's nc
            # from MAP fits and stored theta and an R2, with nothing
            # saying the optimiser had actually reached a mode. The
            # gradient norm and scipy's own success flag say exactly
            # that, cost nothing, and travel with the result.
            jac = getattr(opt, "jac", None)
            grad_norm = (
                float(
                    np.linalg.norm(np.asarray(jac, dtype="float64"))
                )
                if jac is not None
                else float("nan")
            )
            success = bool(getattr(opt, "success", False))
            results = dict(results)
            results["map_grad_norm"] = grad_norm
            results["map_success"] = success
            results["map_nit"] = int(getattr(opt, "nit", -1))
            if not success or not np.isfinite(grad_norm):
                logger.warning(
                    "MAP did NOT converge: success=%s, ||grad||=%.4g, "
                    "message=%r. Treat the coefficients as a stopping "
                    "point, not a mode.",
                    success,
                    grad_norm,
                    getattr(opt, "message", ""),
                )
            else:
                logger.info(
                    "MAP optimization completed: success=True, "
                    "||grad||=%.4g after %d iterations",
                    grad_norm,
                    results["map_nit"],
                )
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
