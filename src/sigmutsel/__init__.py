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
    except Exception:
        __version__ = "unknown"

from sigmutsel.models import MutationDataset, Model
from sigmutsel import locations

__all__ = [
    "MutationDataset",
    "Model",
    "locations",
]
