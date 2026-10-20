"""ArviZ's posterior container type, on arviz 0.2x and 1.x alike.

arviz 1.x replaced ``arviz.InferenceData`` by xarray's ``DataTree``
and warns (``MigrationWarning``) whenever the old name is looked up.
The name is read from the module's own namespace, which does not go
through arviz's warning ``__getattr__``, so neither version warns.
"""

import arviz as az

InferenceData = vars(az).get("InferenceData")
if InferenceData is None:  # arviz >= 1.0
    import xarray as xr

    InferenceData = xr.DataTree

__all__ = ["InferenceData"]
