import pandas as pd
import xarray as xr
import numpy as np
import pathlib

import typing as T

from seameshkit._core import MeshAdapterRegistry
from seameshkit._core import MeshAdapter
from seameshkit._mappings import MESH_FORMATS
from seameshkit._utils import standardize_dataset
from seameshkit._utils import format_specific_dataset

import logging

logger = logging.getLogger(__name__)


@MeshAdapterRegistry.register(MESH_FORMATS.WW3)
class WW3(MeshAdapter):
    @classmethod
    def can_read(cls, path: T.Union[str, pathlib.Path]) -> bool:
        return str(path).endswith(".ww3")

    @classmethod
    def read(cls, path: T.Union[str, pathlib.Path]) -> xr.Dataset:
        # https://github.com/NOAA-EMC/WW3-tools/blob/develop/unst_msh_gen/plot_msh.py
        # purpose: this function reads a gmsh file and returns node and element information
        # additional information about the gmsh file format can be found
        # in many places including here: http://gmsh.info/dev/doc/texinfo/gmsh.pdf
        # The gmsh format is what the WW3 model uses to define unstructured grids

        # input:
        # filename - name of gmsh file

        # output:
        # xy    -  x/y or lon/lat of nodes
        # depth - depth value at node points
        # ect   - element connection table
        # bnd   - list of boundary nodes

        with open(path, "r") as f:
            # Skip mesh format lines
            for _ in range(4):  # Skip 4 lines directly (including $Nodes)
                next(f)

            # Read number of nodes
            nn = int(next(f).strip())

            # Initialize node coordinate and depth arrays
            xy = np.zeros((nn, 2), dtype=np.double)
            depth = np.zeros(nn, dtype=np.double)

            # Read coordinates and depths
            for i in range(nn):
                line = next(f).split()
                idx = int(line[0]) - 1
                x_coord = float(line[1])
                xy[idx, 0] = x_coord - 360 if x_coord > 180 else x_coord
                xy[idx, 1] = float(line[2])
                depth[idx] = float(line[3])

            # Skip '$EndNodes' and '$Elements'
            next(f)
            next(f)

            # Read number of elements
            ne = int(next(f).strip())

            # Initialize temporary arrays to read in element info
            ecttemp = np.zeros((ne, 3), dtype=np.int32)
            bndtemp = []

            elem_count = 0
            for _ in range(ne):
                line = next(f).split()
                eltype = int(line[1])
                if eltype == 15:
                    bndtemp.append(int(line[5]) - 1)
                else:
                    ecttemp[elem_count, :] = [
                        int(line[6]) - 1,
                        int(line[7]) - 1,
                        int(line[8]) - 1,
                    ]
                    elem_count += 1

            # Trim the ect array to the actual number of elements
            ect = ecttemp[:elem_count, :]
            bnd = np.array(bndtemp, dtype=np.int32)

        logger.info("WW3 mesh read successfully")
        ds = xr.Dataset(
            {
                "depth": (("time", "node"), [depth]),
                "face_node_connectivity": (("face", "nmax_face"), ect),
            },
            coords={
                "node_x": ("node", xy[:, 0]),
                "node_y": ("node", xy[:, 1]),
                "time": [pd.Timestamp.now()],
            },
        )
        if len(bnd) > 0:
            ds["boundary_node_connectivity"] = (("boundary"), bnd)

        std_ds = standardize_dataset(ds, "WW3")
        return std_ds

    @classmethod
    def write(cls, std_ds: xr.Dataset, path: T.Union[str, pathlib.Path]) -> None:
        ds = format_specific_dataset(std_ds, "WW3")
        node_data = np.vstack(
            (ds.node_x.values, ds.node_y.values, ds.isel(time=0).depth.values)
        ).T
        tri = (
            ds.face_node_connectivity.values.astype(int) - 1
        )  # Convert to 0-based indexing
        num_nodes = node_data.shape[0]

        with open(path, "w") as fileID:
            fileID.write("$MeshFormat\n")
            fileID.write("2 0 8\n")
            fileID.write("$EndMeshFormat\n")
            fileID.write("$Nodes\n")
            fileID.write(str(num_nodes) + "\n")

            for i in range(num_nodes):
                fileID.write(
                    f"{i + 1}  {node_data[i, 0]:5.5f} {node_data[i, 1]:5.5f} {node_data[i, 2]:5.5f}\n"
                )

            fileID.write("$EndNodes\n")
            fileID.write("$Elements\n")
            num_elements = len(tri)
            fileID.write(str(num_elements) + "\n")

            m = 0
            for i in range(len(tri)):
                m += 1
                fileID.write(f"{m} 2 3 0 {i+1} 0 {tri[i][0]} {tri[i][1]} {tri[i][2]}\n")

            fileID.write("$EndElements\n")
        logger.info("WW3 mesh written successfully")
