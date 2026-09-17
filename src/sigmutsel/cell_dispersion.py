"""Deprecated alias of :mod:`sigmutsel.gene_tumor_dispersion`."""

import warnings

from .gene_tumor_dispersion import (
    DEFAULT_STRATA,
    _fit_trend,
    _log_rising,
    dirichlet_multinomial_logpmf,
    fit_gene_tumor_dispersion_trend,
    fit_phi,
    phi_for_count,
)

fit_cell_dispersion_trend = fit_gene_tumor_dispersion_trend

__all__ = [
    "DEFAULT_STRATA",
    "_fit_trend",
    "_log_rising",
    "dirichlet_multinomial_logpmf",
    "fit_cell_dispersion_trend",
    "fit_gene_tumor_dispersion_trend",
    "fit_phi",
    "phi_for_count",
]

warnings.warn(
    "sigmutsel.cell_dispersion is deprecated; use "
    "sigmutsel.gene_tumor_dispersion.",
    DeprecationWarning,
    stacklevel=2,
)
