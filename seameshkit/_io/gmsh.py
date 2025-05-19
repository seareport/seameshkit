import xarray as xr

import uuid
import gmsh
import numpy as np
import pandas as pd

import typing as T
import pathlib

from seameshkit._core import MeshAdapterRegistry
from seameshkit._core import MeshAdapter
from seameshkit._mappings import MESH_FORMATS
from seameshkit._utils import standardize_dataset
from seameshkit._utils import format_specific_dataset

import logging

logger = logging.getLogger(__name__)

if not gmsh.is_initialized():
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)


@MeshAdapterRegistry.register(MESH_FORMATS.GMSH)
class GMSH(MeshAdapter):
    @classmethod
    def can_read(cls, path: T.Union[str, pathlib.Path]) -> bool:
        return str(path).endswith(".msh")

    @classmethod
    def read(cls, path: T.Union[str, pathlib.Path]) -> xr.Dataset:
        gmsh.model.add(str(uuid.uuid4()))
        gmsh.open(str(path))
        tri_i, tri_n = gmsh.model.mesh.getElementsByType(2)
        tri_n = tri_n.reshape([-1, 3])

        node_i, nodes, _ = gmsh.model.mesh.getNodes()
        nodes = nodes.reshape([-1, 3])
        x = nodes[:, 0]
        y = nodes[:, 1]
        z = nodes[:, 2]
        element = np.subtract(tri_n, 1)

        ds = xr.Dataset(
            {
                "depth": ("node", z),
                "face_node_connectivity": (("face", "nmax_face"), element),
            },
            coords={
                "node_x": ("node", x),
                "node_y": ("node", y),
                "time": [pd.Timestamp.now()],
            },
        )

        gmsh.model.remove()
        std_ds = standardize_dataset(ds, "GMSH")
        return std_ds

    @classmethod
    def write(cls, std_ds: xr.Dataset, path: T.Union[str, pathlib.Path]) -> None:
        pass
