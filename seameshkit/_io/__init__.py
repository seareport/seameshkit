from __future__ import annotations

# Import all adapters to register them
from . import ww3
from . import telemac
from . import schism
from . import ugrid
from . import gmsh

# Define __all__ to expose only what's needed
__all__ = [
    "ww3",
    "telemac",
    "schism",
    "ugrid",
    "gmsh",
]
