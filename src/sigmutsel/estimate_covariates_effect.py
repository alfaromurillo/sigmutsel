"""Estimate covariate effects on mutation rates.

This module provides Bayesian inference tools to estimate how
genomic covariates (gene expression, chromatin state, replication
timing, etc.) affect mutation rates. It uses PyMC for probabilistic
modeling and can handle both single-signature and multi-signature
scenarios.

The main function fits a log-linear model where mutation rates are
modulated by exponential covariate effects: μ = μ_base * exp(X @ β),
where X is the covariate matrix and β are the effect coefficients.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pymc as pm
import pymc.sampling.jax as pmjax
import pytensor.tensor as tt
import arviz as az

import logging

from .estimate_presence import filter_passenger_genes_ensembl

from .constants import random_seed

logger = logging.getLogger(__name__)


def estimate_covariates_effect(
    mus: np.ndarray | dict[int | str, np.ndarray],
    presence_matrix: np.ndarray,
    cov_matrix: np.ndarray,
    draws: int = 4000,
    lower_bounds_c: float | np.ndarray | None = -2,
    upper_bounds_c: float | np.ndarray = 2,
    burn: int = 1000,
    chains: int = 4,
    save_path: str | Path | None = None,
    kwargs: dict | None = None,
) -> az.InferenceData | dict:
    """Estimate covariate effects (with intercept) in a Bernoulli model.

    Builds a per-gene linear predictor:
    ``eta_g = dot(c, cov_ext[g, :])``
    where
    ``cov_ext = [1, cov_1, …, cov_K]``
    (a leading column of ones is prepended for the intercept). Base
    rates ``mus`` are scaled by ``exp(eta_g)``, and a Bernoulli
    likelihood is fit to ``presence_matrix``. Implemented with PyMC
    (>=5.23) and PyTensor.

    Parameters
    ----------
    mus : ndarray | dict[int | str, ndarray]
        **Signature independent mode:**
            Array of shape (n_tumors, n_genes) with per-tumor,
            per-gene mutation-rate components (nonnegative).

        **Multi-signature mode:**
            Dictionary mapping signature identifiers to arrays, each
            shaped (n_tumors, n_genes). The model fits separate
            covariate effects ``c[sigma]`` per signature and sums
            their contributions:
            ``mus_full = sum_sigma(mus[sigma] * exp(eta[sigma]))``.
            As produced by :func:`compute_mu_g_per_tumor` with
            signature-separated input.

    presence_matrix : ndarray, shape (n_tumors, n_genes)
        Binary presence indicators (0/1) per tumor and gene.
    cov_matrix : ndarray, shape (n_genes, n_covariates)
        Gene-level covariates. A column of ones is added internally to
        model the intercept, so the coefficient vector length is
        ``n_covariates + 1``.
    draws : int, default 4000
        Total posterior draws **across all chains**. If ``draws == 1``,
        perform MAP/MLE via ``pm.find_MAP`` (uniform priors make MAP = MLE).
    upper_bounds_c : float | array-like, default 2
        Upper bound(s) for the Uniform prior over ``c``. Formats:
          * **Scalar**: Applied to all coefficients.
          * **1D array** (length ``n_covariates + 1``): Per-coefficient
            bounds in signature independent mode. Element 0 is intercept,
            remaining elements are covariates.
          * **2D array** (shape ``(n_signatures, n_covariates + 1)``):
            Per-signature, per-coefficient bounds in multi-signature
            mode. Row i corresponds to signature i in ``mus.keys()``
            order.

    lower_bounds_c : float | array-like | None, default -1
        Lower bound(s) for the Uniform prior over ``c``.
        Accepts same formats as ``upper_bounds_c``.

        **Special behavior**: If ``None``, automatically set to
        ``-upper_bounds_c`` (negated element-wise), creating
        symmetric bounds around zero.
    burn : int, default 1000
        Number of tuning (warm-up) steps **per chain**.
    chains : int, default 4
        Number of MCMC chains.
    save_path : str | Path | None, default None
        If provided, must be the full path (without extension) where
        results should be written. ``.npz`` is appended for MAP mode
        (``draws == 1``) via ``np.savez``; ``.nc`` (NetCDF) is appended
        for posterior samples via ``InferenceData.to_netcdf``. The
        parent directory is created automatically.
    kwargs : dict | None, default None
        Extra keyword arguments forwarded to the sampler
        (``pmjax.sample_numpyro_nuts``) or to ``pm.find_MAP`` when
        ``draws == 1``.

    Returns
    -------
    arviz.InferenceData | dict
        If ``draws > 1``: an ArviZ ``InferenceData`` with posterior
        samples. Variable ``c`` has shape ``(n_coeffs,)`` in signature
        independent mode or ``(n_signatures, n_coeffs)`` in
        multi-signature mode.
        If ``draws == 1``: a dict containing the MAP/MLE. Key ``c``
        maps to an array of shape ``(n_coeffs,)`` or
        ``(n_signatures, n_coeffs)`` respectively.

    Raises
    ------
    ValueError
        If ``cov_matrix`` does not have ``n_genes`` rows, or if
        multi-signature arrays have inconsistent shapes.

    Notes
    -----
    * Coefficients ``c`` have shape ``(n_covariates + 1,)`` in signature
      independent mode or ``(n_signatures, n_covariates + 1)`` in
      multi-signature mode; ``c[0]`` or ``c[:, 0]`` is the intercept.
    * **Signature independent:** ``eta = cov_ext @ c`` (shape ``(n_genes,)``),
      broadcast to ``(n_tumors, n_genes)``. The Bernoulli parameter is
      ``p = 1 - exp(-mus * exp(eta))``.
    * **Multi-signature (vectorized):** Coefficients ``c`` has shape
      ``(n_signatures, n_coeffs)``. Computation uses batched
      operations: ``eta[s,:] = cov_ext @ c[s,:]`` for each signature
      s, then ``mus_full = sum_s(mus[s] * exp(eta[s]))``. This is
      faster than looping and allows PyTensor to optimize the full
      computation graph.
    * Inputs are cast to ``float32`` (and presence to ``uint8``) for
      performance.
    * When ``draws > 1``, each chain receives ``draws // chains``
      samples.

    """
    if kwargs is None:
        kwargs = {}

    # Detect multi-signature mode
    if isinstance(mus, dict):
        signatures = list(mus.keys())
        n_tumors, n_genes = next(iter(mus.values())).shape
        # Validate all have same shape
        for sigma in signatures:
            if mus[sigma].shape != (n_tumors, n_genes):
                raise ValueError(
                    f"All mus arrays must have shape "
                    f"({n_tumors}, {n_genes}), but mus[{sigma}] "
                    f"has shape {mus[sigma].shape}"
                )
        logger.info(
            f"Multi-signature mode: {len(signatures)} signatures, "
            f"{n_tumors} tumors, {n_genes} genes"
        )
    else:
        signatures = None
        n_tumors, n_genes = mus.shape
        logger.info(
            f"Signature-independent mode: "
            f"{n_tumors} tumors, {n_genes} genes"
        )

    if cov_matrix.shape[0] != n_genes:
        raise ValueError("cov_matrix must have n_genes rows.")

    # Build cov_extended: [1, cov1, cov2, ...]
    ones = np.ones((n_genes, 1), dtype=np.float32)
    cov_ext = np.concatenate(
        [ones, np.asarray(cov_matrix, dtype=np.float32)], axis=1
    )

    n_coeffs = cov_ext.shape[1]

    if lower_bounds_c is None:
        lower_bounds_c = -upper_bounds_c

    # Validate bounds shapes in multi-signature mode
    if signatures is not None:
        for bounds_name, bounds in [
            ("upper_bounds_c", upper_bounds_c),
            ("lower_bounds_c", lower_bounds_c),
        ]:
            if isinstance(bounds, np.ndarray) and bounds.ndim == 2:
                if bounds.shape != (len(signatures), n_coeffs):
                    raise ValueError(
                        f"{bounds_name} has shape {bounds.shape} but "
                        f"expected ({len(signatures)}, {n_coeffs}) for "
                        f"multi-signature mode with {len(signatures)} "
                        f"signatures and {n_coeffs} coefficients"
                    )

    with pm.Model():
        cov32 = pm.Data("cov_ext", cov_ext.astype("float32"))
        pres8 = pm.Data("pres", presence_matrix.astype("uint8"))

        if signatures is None:
            # ──────── Signature independent mode ────────
            c = pm.Uniform(
                name="c",
                lower=lower_bounds_c,
                upper=upper_bounds_c,
                shape=n_coeffs,
            )

            mus32 = pm.Data(
                "mus", np.clip(mus.astype("float32"), 1e-12, np.inf)
            )

            eta_gene = tt.dot(cov32, c)
            eta = eta_gene.dimshuffle("x", 0)
            mus_full = mus32 * tt.exp(eta)

        else:
            # ──────── Multi-signature mode (vectorized) ────────
            # Stack all signature baselines into 3D array
            # Shape: (n_signatures, n_tumors, n_genes)
            mus_stacked = np.stack(
                [mus[sigma] for sigma in signatures], axis=0
            )
            mus_data = pm.Data(
                "mus",
                np.clip(mus_stacked.astype("float32"), 1e-12, np.inf),
            )

            # Batched coefficient matrix
            # Shape: (n_signatures, n_coeffs)
            c = pm.Uniform(
                name="c",
                lower=lower_bounds_c,
                upper=upper_bounds_c,
                shape=(len(signatures), n_coeffs),
            )

            # Batched matrix multiply: eta[s,g] = sum_k(cov[g,k] * c[s,k])
            # Use einsum: 'sk,gk->sg' (s=sigs, g=genes, k=coeffs)
            eta_gene = tt.tensordot(c, cov32, axes=[[1], [1]])
            # Shape: (n_signatures, n_genes)

            # Broadcast to (n_signatures, n_tumors, n_genes)
            eta = eta_gene.dimshuffle(0, "x", 1)

            # Element-wise scale: (n_signatures, n_tumors, n_genes)
            mus_scaled = mus_data * tt.exp(eta)

            # Sum over signatures axis to get total rate
            mus_full = mus_scaled.sum(axis=0)
            # Shape: (n_tumors, n_genes)

        # Bernoulli likelihood (shared between modes)
        Ps = tt.clip(1.0 - tt.exp(-mus_full), 1e-10, 1.0 - 1e-10)
        pm.Bernoulli(name="genes_observed", p=Ps, observed=pres8)

        if draws == 1:
            logger.info(
                f"Finding MAP estimate for {n_coeffs} coefficient(s)"
            )
            results = pm.find_MAP(**kwargs)
            logger.info("MAP optimization completed")
        else:
            logger.info(
                f"Sampling posterior: {draws} draws across "
                f"{chains} chains ({int(draws/chains)} per chain), "
                f"{burn} tuning steps"
            )
            results = pmjax.sample_numpyro_nuts(
                draws=int(draws / chains),
                chain_method="parallel",
                tune=burn,
                chains=chains,
                target_accept=0.9,
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


def estimate_all_cov_effects(
    mus: pd.DataFrame | dict[int | str, pd.DataFrame],
    presence_matrix: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    sample: int | str = "MAP",
    subset_sizes: int | list[int] | str = "all",
    sequential: bool = False,
    column_restriction: str | list[str] | None = None,
    restrict_to_passenger: bool = True,
    lower_bounds_c: (
        float
        | np.ndarray
        | dict[tuple[str, ...], float | np.ndarray]
        | None
    ) = -1,
    upper_bounds_c: (
        float | np.ndarray | dict[tuple[str, ...], float | np.ndarray]
    ) = 1.5,
    save_results: str | bool = True,
    results_dir: str | Path | None = None,
) -> dict[tuple[str, ...], np.ndarray | az.InferenceData]:
    """Estimate covariate-effect over all (or selected) covariates.

    For each chosen subset_size `k` and each `k`-combination of
    covariate columns in `cov_matrix`, this function:
      1) filters genes to those with non-missing values for that
         combination in cov_matrix, present in both `mus` and
         `presence_matrix` (optionally passenger-only),
      2) calls `estimate_covariates_effect(...)` to fit the Bernoulli model,
      3) stores either the MAP coefficients (if `sample='MAP'`) or the full
         posterior object (if `sample` is an int or `'full'`).

    Parameters
    ----------
    mus : pd.DataFrame | dict[int | str, pd.DataFrame]
        **Signature independent mode:**
            Genes × tumors baseline mutation-rate components. Index
            must be Ensembl gene IDs; columns are tumor/sample IDs.
        **Multi-signature mode:**
            Dict mapping signature identifiers to DataFrames (genes ×
            tumors). As returned by :func:`compute_mu_g_per_tumor`
            with signature-separated input.
    presence_matrix : pd.DataFrame
        Genes × tumors binary matrix (variant/gene presence). Index must match
        `mus`.
    cov_matrix : pd.DataFrame
        Genes × covariates values. Columns are covariate names to be combined.
    sample : {"MAP", "full"} | int, default "MAP"
        Sampling mode:
          - "MAP" (case-insensitive): run MAP only (i.e., `draws=1`).
          - "full": MCMC with `draws=4000` total (split across chains by the
            estimator).
          - int N: MCMC with `draws=4000`, but **randomly subsample N genes**
            for each combination to keep runtime manageable.
        Note: the subsampling uses pandas `.sample(N)` without a fixed seed.
    subset_sizes : {"all"} | int | list[int], default "all"
        Which subset sizes to try. 'all' means 1..len(covariates).
        You can pass a single integer (e.g., 2) or a list like [1, 3].
    sequential : bool, default False
        If True, run nested models using the column order in
        `cov_matrix`: {cov1}, {cov1,cov2}, ..., {cov1,..,covn}. This is
        particularly useful when columns are PCs from a PCA.
    column_restriction : str | list[str] | None, default None
        Restrict the set of tested models to those that **include**
        specific covariate(s). If a single string is provided, only
        combinations that contain that covariate are evaluated. If a
        list is provided, only combinations that contain **all**
        listed covariates are evaluated.  When ``None``, no
        restriction is applied.
    restrict_to_passenger : bool, default True
        If True, further restrict `genes_to_consider` via
        func:`estimate_presence.filter_passenger_genes_ensembl`.

    upper_bounds_c : float | array-like | dict, default 1.5
        Upper bound(s) for the Uniform prior over coefficient vector ``c``.

        **Formats**:
          * **Scalar**: Applied to all coefficients across all models.
          * **Per-model dict**: Keys are covariate tuples (e.g.,
            ``('mrt',)``), values are scalars or arrays for that model.
          * **Template dict**: Keys specify defaults and overrides:
            - ``'c0'``: intercept bound (default for all models)
            - Covariate names (e.g., ``'mrt'``): bound for that covariate
            - ``'cs'`` or ``'other'``: fallback for unlisted covariates

        **Multi-signature mode**: Use tuple keys ``(item, signature)``
        for signature-specific bounds. Lookup order per signature:
        ``(item, sig)`` → ``item`` → ``'cs'/'other'`` → default.

        **Examples**:
          * Simple: ``2`` (all coefficients bounded at 2)
          * Per-covariate: ``{'c0': 3, 'mrt': 2, 'other': 1}``
          * Per-signature: ``{'c0': 2, ('mrt', 'SBS5'): 1.5,
            ('mrt', 'SBS18'): 2, 'other': 1}`` sets intercept=2 for
            all, mrt=1.5 for SBS5, mrt=2 for SBS18, other
            covariates=1.

    lower_bounds_c : float | array-like | dict | None, default -1
        Lower bound(s) for the Uniform prior over ``c``.
        Accepts the same formats as ``upper_bounds_c`` (scalar, dict,
        per-model, template, tuple keys for multi-signature).

        **Special behavior**: If ``None``, automatically set to
        ``-upper_bounds_c`` (negated element-wise). This creates
        symmetric bounds around zero.
    save_results : bool | str, default True
        Controls whether and how each model fit is saved by
        `estimate_covariates_effect`:
          • True
            Save each fit using an auto name:
            'cov_effect_' + '+'.join(covariates).
          • False
            Do not save intermediate results.
          • str
            Use this string as a tag inserted after 'cov_effect_' and before
            the covariate combo (e.g., 'cov_effect_COAD_cov1+cov2').
        Requires `results_dir` to be provided when saving.
    results_dir : str | Path | None, default None
        Directory where result files should be written when
        `save_results` is truthy. The directory is created if it does
        not exist. Ignored when `save_results` is False.

    Returns
    -------
    results : dict[tuple[str, ...], Any]
        Mapping from covariate-name tuple (e.g., `('mrt',)` or
        `('loglog1p_gtex','mrt')`) to:

          - **Signature independent mode, MAP case**: 1D NumPy array of
            coefficients with shape `(n_coeffs,)`.
          - **Signature independent mode, MCMC case**: ArviZ InferenceData
            posterior.
          - **Multi-signature mode, MAP case**: 2D NumPy array with
            shape `(n_signatures, n_coeffs)`. Row i corresponds to
            signature i in the order of `mus.keys()`.
          - **Multi-signature mode, MCMC case**: ArviZ InferenceData
            with variable `c` of shape `(n_signatures, n_coeffs)`.

        Coefficients are ordered `[c_intercept, c_cov1, c_cov2, ...]`
        as defined by `estimate_covariates_effect`.

    Notes
    -----
    - This function expects `estimate_covariates_effect` to accept:
        mus.T  (tumors × genes) or dict of arrays
        presence.T
        cov_matrix_for_genes (genes × covariates)
      and to name the coefficient vector `'c'` for the MAP return dict.
    - In multi-signature mode, coefficients are estimated jointly for
      all signatures, with each signature getting its own coefficient
      vector while sharing the same likelihood.
    - Make sure `cov_matrix` column names match the covariates you
      intend to test.
    - MCMC hyperparameters (e.g., chains, tune) are those defined
      inside `estimate_covariates_effect`.

    """
    from itertools import combinations

    results = {}
    genes_considered = {}

    # Detect multi-signature mode
    if isinstance(mus, dict):
        is_multi_sig = True
        signatures = list(mus.keys())
        # Use first signature's index as reference
        mus_index = next(iter(mus.values())).index
    else:
        is_multi_sig = False
        signatures = None
        mus_index = mus.index

    if isinstance(sample, int) or sample.lower() == "full":
        draws = 4000
    elif sample.lower() == "map":
        draws = 1

    covs = list(cov_matrix.columns)

    if column_restriction is None:
        required = set()
    elif isinstance(column_restriction, str):
        required = {column_restriction}
    else:
        required = set(column_restriction)

    # To avoid typos, check that required covariates exist
    missing_required = [c for c in required if c not in covs]
    if missing_required:
        raise KeyError(
            "column_restriction contains unknown "
            f"covariates: {missing_required}"
        )

    if sequential:
        order = tuple(covs)
        # Only sizes that can be built sequentially
        if subset_sizes == "all":
            sizes_list = [n for n in range(1, len(order) + 1)]
        elif isinstance(subset_sizes, int):
            sizes_list = [subset_sizes]
        else:
            sizes_list = list(subset_sizes)
        size_iter = sizes_list
    else:
        if subset_sizes == "all":
            sizes_list = [n for n in range(1, len(covs) + 1)]
        elif isinstance(subset_sizes, int):
            sizes_list = [subset_sizes]
        else:
            sizes_list = list(subset_sizes)
        size_iter = sizes_list

    for size in size_iter:
        if sequential:
            combo_iter = [tuple(order[:size])]
        else:
            combo_iter = combinations(covs, size)
        for combo in combo_iter:
            if required and not required.issubset(set(combo)):
                continue
            logger.info(f"Running estimation for covariates: {combo}")
            restrict_matrix = cov_matrix.loc[:, combo]
            restrict_matrix = restrict_matrix[
                restrict_matrix.notna().all(axis=1)
            ]
            genes_to_consider = mus_index.intersection(
                presence_matrix.index
            ).intersection(restrict_matrix.index)

            if restrict_to_passenger:
                genes_to_consider = filter_passenger_genes_ensembl(
                    genes_to_consider
                )

            if draws > 1 and isinstance(sample, int):
                # then we restrict to only `sample` random genes
                genes_to_consider = pd.Index(
                    genes_to_consider.to_series().sample(
                        sample, random_state=random_seed
                    )
                )

            if save_results:
                if results_dir is None:
                    raise ValueError(
                        "results_dir must be provided when save_results is True."
                    )
                save_name = "cov_effect_"
                if (
                    isinstance(save_results, str)
                    and save_results is not True
                ):
                    save_name = f"{save_name}{save_results}_"
                save_name = save_name + "+".join(combo)
                save_path = Path(results_dir) / save_name
            else:
                save_path = None

            # Resolve lower/upper bounds for this combo
            def _resolve_template(
                bounds,
                combo,
                default_scalar,
                sign=+1,
                signatures=None,
            ):
                # scalar/array passthrough
                if not isinstance(bounds, dict):
                    return bounds
                # per-combo dict passthrough
                if combo in bounds:
                    return bounds[combo]

                # Multi-signature mode: build 2D array (n_signatures, n_coeffs)
                if signatures is not None:
                    tmpl = dict(bounds)
                    if "other" in tmpl and "cs" not in tmpl:
                        tmpl["cs"] = tmpl["other"]

                    result = []
                    for sigma in signatures:
                        # Build coefficient vector for this signature
                        vec = []

                        # Intercept: check ('c0', sigma), then 'c0', then default
                        c0_key = ("c0", sigma)
                        if c0_key in tmpl:
                            c0_val = tmpl[c0_key]
                        elif "c0" in tmpl:
                            c0_val = tmpl["c0"]
                        else:
                            c0_val = default_scalar
                        vec.append(float(sign * float(c0_val)))

                        # Covariates: check (cov, sigma), then cov, then 'cs'/'other', then default
                        cs_val = tmpl.get("cs", default_scalar)
                        for cov in combo:
                            cov_sig_key = (cov, sigma)
                            if cov_sig_key in tmpl:
                                val = tmpl[cov_sig_key]
                            elif cov in tmpl:
                                val = tmpl[cov]
                            elif isinstance(cs_val, dict):
                                if cs_val:
                                    fallback = next(
                                        iter(cs_val.values())
                                    )
                                else:
                                    fallback = default_scalar
                                val = cs_val.get(cov, fallback)
                            else:
                                val = cs_val
                            vec.append(float(sign * float(val)))

                        result.append(vec)

                    return np.asarray(result, dtype=float)

                # Signature independent mode: template dict (original logic)
                tmpl = dict(bounds)
                if "other" in tmpl and "cs" not in tmpl:
                    tmpl["cs"] = tmpl["other"]
                c0_val = tmpl.get("c0", default_scalar)
                cs_val = tmpl.get("cs", default_scalar)
                vec = [float(sign * float(c0_val))]
                for cov in combo:
                    if cov in tmpl:
                        vec.append(float(sign * float(tmpl[cov])))
                    elif isinstance(cs_val, dict):
                        if cs_val:
                            fallback = next(iter(cs_val.values()))
                        else:
                            fallback = default_scalar
                        vec.append(
                            float(
                                sign
                                * float(cs_val.get(cov, fallback))
                            )
                        )
                    else:
                        vec.append(float(sign * float(cs_val)))
                return np.asarray(vec, dtype=float)

            # Upper default scalar baseline uses provided scalar if any
            if isinstance(upper_bounds_c, (int, float)):
                _ub_default = float(upper_bounds_c)
            else:
                _ub_default = 1.0

            upp_bound = _resolve_template(
                upper_bounds_c,
                combo,
                _ub_default,
                +1,
                signatures=signatures if is_multi_sig else None,
            )

            # Lower bounds: mirror upper if None; apply same template rules
            if lower_bounds_c is None:
                low_bound = _resolve_template(
                    upper_bounds_c,
                    combo,
                    _ub_default,
                    -1,
                    signatures=signatures if is_multi_sig else None,
                )
            else:
                if isinstance(lower_bounds_c, (int, float)):
                    _lb_default = float(lower_bounds_c)
                else:
                    _lb_default = -_ub_default
                low_bound = _resolve_template(
                    lower_bounds_c,
                    combo,
                    _lb_default,
                    +1,
                    signatures=signatures if is_multi_sig else None,
                )

            # Prepare mus input based on mode
            if is_multi_sig:
                # Multi-signature: pass dict with transposed arrays
                mus_input = {
                    sigma: mus[sigma].loc[genes_to_consider].T.values
                    for sigma in signatures
                }
            else:
                # Signature independent mode: transpose DataFrame to array
                mus_input = mus.loc[genes_to_consider].T.values

            estimation = estimate_covariates_effect(
                mus_input,
                presence_matrix.loc[genes_to_consider].T.values,
                restrict_matrix.loc[genes_to_consider].values,
                draws=draws,
                lower_bounds_c=low_bound,
                upper_bounds_c=upp_bound,
                save_path=save_path,
            )
            if draws == 1:
                results[combo] = estimation["c"]
            else:
                results[combo] = estimation
            genes_considered[combo] = genes_to_consider
            print("")

    return results


def load_or_estimate_all_cov_effects(
    mus: pd.DataFrame | dict[int | str, pd.DataFrame],
    presence_matrix: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    sample: int | str = "MAP",
    subset_sizes: int | list[int] | str = "all",
    sequential: bool = False,
    column_restriction: str | list[str] | None = None,
    restrict_to_passenger: bool = True,
    lower_bounds_c: (
        float
        | np.ndarray
        | dict[tuple[str, ...], float | np.ndarray]
        | None
    ) = -1,
    upper_bounds_c: (
        float | np.ndarray | dict[tuple[str, ...], float | np.ndarray]
    ) = 1.5,
    save_results: str | bool = True,
    results_dir: str | Path | None = None,
    force_generation: bool = False,
) -> dict[tuple[str, ...], np.ndarray | az.InferenceData]:
    """
    Load or estimate covariate effects with caching support.

    Wraps :func:`estimate_all_cov_effects` with load/save logic.
    If `force_generation=False` and `save_results` is not False,
    checks if all expected result files exist and loads them.
    Otherwise, calls `estimate_all_cov_effects` to generate.

    Parameters are identical to :func:`estimate_all_cov_effects`
    except for the additional `force_generation` parameter.

    Parameters
    ----------
    results_dir : str | Path | None, default None
        Directory containing cached result files. Required when
        `save_results` is truthy. Created automatically if missing
        when generating new results.
    force_generation : bool, default False
        If True, regenerate results even if saved files exist.
        If False, loads previously saved results when available.

    Returns
    -------
    results : dict[tuple[str, ...], Any]
        Same as :func:`estimate_all_cov_effects`. Mapping from
        covariate tuples to coefficient arrays or posteriors.

    See Also
    --------
    estimate_all_cov_effects : Core estimation function.

    Notes
    -----
    - For MAP mode (sample="MAP"), expects .npz files
    - For MCMC mode (sample > 1 or "full"), expects .nc files
    - File paths determined by `results_dir`, `save_results`, and
      covariate names.
    """
    from itertools import combinations
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    # Determine if we can/should try loading
    can_load = save_results is not False and not force_generation

    if save_results is not False and results_dir is None:
        raise ValueError(
            "results_dir must be provided when save_results is enabled."
        )

    if not can_load:
        # Generate without loading attempt
        return estimate_all_cov_effects(
            mus=mus,
            presence_matrix=presence_matrix,
            cov_matrix=cov_matrix,
            sample=sample,
            subset_sizes=subset_sizes,
            sequential=sequential,
            column_restriction=column_restriction,
            restrict_to_passenger=restrict_to_passenger,
            lower_bounds_c=lower_bounds_c,
            upper_bounds_c=upper_bounds_c,
            save_results=save_results,
            results_dir=results_dir,
        )

    # Build expected file names
    covs = list(cov_matrix.columns)

    if subset_sizes == "all":
        sizes = range(1, len(covs) + 1)
    elif isinstance(subset_sizes, int):
        sizes = [subset_sizes]
    else:
        sizes = subset_sizes

    # Generate expected combinations
    expected_combos = []
    if sequential:
        for k in sizes:
            expected_combos.append(tuple(covs[:k]))
    else:
        for k in sizes:
            expected_combos.extend(combinations(covs, k))

    # Apply column restriction if provided
    if column_restriction is not None:
        if isinstance(column_restriction, str):
            restriction_set = {column_restriction}
        else:
            restriction_set = set(column_restriction)

        expected_combos = [
            combo
            for combo in expected_combos
            if restriction_set.issubset(set(combo))
        ]

    # Determine file extension based on sample mode
    if isinstance(sample, int) or (
        isinstance(sample, str) and sample.lower() == "full"
    ):
        file_ext = ".nc"
        is_mcmc = True
    else:
        file_ext = ".npz"
        is_mcmc = False

    # Build file paths
    base_dir = Path(results_dir) if results_dir is not None else None
    file_paths = []
    for combo in expected_combos:
        save_name = "cov_effect_"
        if isinstance(save_results, str):
            save_name = f"{save_name}{save_results}_"
        save_name = save_name + "+".join(combo)
        file_path = base_dir / f"{save_name}{file_ext}"
        file_paths.append((combo, file_path))

    # Check if all files exist
    all_exist = all(fp.exists() for _, fp in file_paths)

    if all_exist:
        # Load results
        logger.info(
            f"Loading covariate effects from saved {file_ext} "
            f"files..."
        )
        results = {}
        for combo, file_path in file_paths:
            if is_mcmc:
                results[combo] = az.from_netcdf(str(file_path))
            else:
                data = np.load(file_path)
                results[combo] = data["c"]
        logger.info("... done loading covariate effects.")
        return results
    else:
        # Generate results
        return estimate_all_cov_effects(
            mus=mus,
            presence_matrix=presence_matrix,
            cov_matrix=cov_matrix,
            sample=sample,
            subset_sizes=subset_sizes,
            sequential=sequential,
            column_restriction=column_restriction,
            restrict_to_passenger=restrict_to_passenger,
            lower_bounds_c=lower_bounds_c,
            upper_bounds_c=upper_bounds_c,
            save_results=save_results,
            results_dir=results_dir,
        )
