"""sigmutsel: Signature based mutation rate and selection estimation in cancer.

This package provides tools for estimating mutation rates and
inferring selection coefficients from tumor sequencing data using
signature decomposition.

"""

# Version is managed by setuptools-scm
try:
    from sigmutsel._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("sigmutsel")
    # Any lookup failure means "version unknown", not a real error;
    # narrowing to PackageNotFoundError would miss other broken-
    # install cases (e.g. corrupt metadata) that should fall back
    # the same way.
    except Exception:  # noqa: BLE001
        __version__ = "unknown"

from sigmutsel import locations
from sigmutsel.cross_validation import gene_cv_passenger_r2
from sigmutsel.models import Model, MutationDataset

__all__ = [
    "Model",
    "MutationDataset",
    "gene_cv_passenger_r2",
    "locations",
]
