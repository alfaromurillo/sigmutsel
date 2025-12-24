"""sigmutsel: Signature based mutation rate and selection estimation in cancer.

This package provides tools for estimating mutation rates and
inferring selection coefficients from tumor sequencing data using
signature decomposition.

"""

__version__ = "0.1.0"

from sigmutsel.models import MutationDataset, Model
from sigmutsel import locations

__all__ = [
    "MutationDataset",
    "Model",
    "locations",
]
