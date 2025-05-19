import pathlib
import collections
import io
import itertools
import os
import typing as T
import numpy as np
import pandas as pd
import xarray as xr

from seameshkit._core import MeshAdapterRegistry
from seameshkit._core import MeshAdapter
from seameshkit._mappings import MESH_FORMATS
from seameshkit._utils import standardize_dataset
from seameshkit._utils import format_specific_dataset

import logging

logger = logging.getLogger(__name__)


def _readline(fd: bytes) -> bytes:
    return fd.readline().split(b"=")[0].split(b"!")[0].strip()


def parse_hgrid(
    path: os.PathLike[str] | str,
    include_boundaries: bool = False,
    sep: str | None = None,
) -> dict[str, T.Any]:
    """
    Parse an hgrid.gr3 file.

    The function is also able to handle fort.14 files, too, (i.e. ADCIRC)
    but the boundary parsing is not keeping all the available information.
    """
    rvalue: dict[str, T.Any] = {}
    with open(path, "rb") as fd:
        _ = fd.readline()  # skip line
        no_elements, no_points = map(int, fd.readline().strip().split(b"!")[0].split())
        nodes_buffer = io.BytesIO(b"\n".join(itertools.islice(fd, 0, no_points)))
        nodes = np.loadtxt(nodes_buffer, delimiter=sep, usecols=(1, 2, 3))
        elements_buffer = io.BytesIO(b"\n".join(itertools.islice(fd, 0, no_elements)))
        elements = np.loadtxt(
            elements_buffer, delimiter=sep, usecols=(2, 3, 4), dtype=int
        )
        elements -= 1  # 0-based index for the nodes
        rvalue["nodes"] = nodes
        rvalue["elements"] = elements
        # boundaries
        if include_boundaries:
            boundaries = collections.defaultdict(list)
            no_open_boundaries = int(_readline(fd))
            total_open_boundary_nodes = int(_readline(fd))
            for i in range(no_open_boundaries):
                no_nodes_in_boundary = int(_readline(fd))
                boundary_nodes = np.loadtxt(fd, delimiter=sep, usecols=(0,), dtype=int)
                boundaries["open"].append(boundary_nodes - 1)  # 0-based index
            # closed boundaries
            no_closed_boundaries = int(_readline(fd))
            total_closed_boundary_nodes = int(_readline(fd))
            for _ in range(no_closed_boundaries):
                # Sometimes it seems that the closed boundaries don't have a "type indicator"
                # For example: Test_COSINE_SFBay/hgrid.gr3
                # In this cases we assume that boundary type is 0 (i.e. land in schism)
                # XXX Maybe check the source code?
                parsed = _readline(fd).split(b" ")
                if len(parsed) == 1:
                    no_nodes_in_boundary = int(parsed[0])
                    boundary_type = 0
                else:
                    no_nodes_in_boundary, boundary_type = map(
                        int, (p for p in parsed if p)
                    )
                boundary_nodes = np.genfromtxt(
                    fd,
                    delimiter=sep,
                    usecols=(0,),
                    max_rows=no_nodes_in_boundary,
                    dtype=int,
                )
                boundary_nodes -= 1  # 0-based-index
                boundaries[boundary_type].append(boundary_nodes)
            rvalue["boundaries"] = boundaries
    return rvalue


@MeshAdapterRegistry.register(MESH_FORMATS.SCHISM)
@MeshAdapterRegistry.register(MESH_FORMATS.GR3)
class SCHISM(MeshAdapter):
    @classmethod
    def can_read(cls, path: T.Union[str, pathlib.Path]) -> bool:
        return str(path).endswith(".gr3")

    @classmethod
    def read(cls, path: T.Union[str, pathlib.Path]) -> xr.Dataset:
        logger.info(f"read mesh file {path}")
        try:
            mesh_dic = parse_hgrid(path, include_boundaries=True)
        except Exception as e:
            logging.error(f"couldn't read {path}: {e}")
            mesh_dic = parse_hgrid(path)

        schism_ds = xr.Dataset(
            {
                "depth": ("node", mesh_dic["nodes"][:, 2]),
                "face_node_connectivity": (["face", "nmax_face"], mesh_dic["elements"]),
            },
            coords={
                "node_x": ("node", mesh_dic["nodes"][:, 0]),
                "node_y": ("node", mesh_dic["nodes"][:, 1]),
                "time": [pd.Timestamp.now()],
            },
        )
        if "boundary" in mesh_dic.keys():
            schism_ds["boundary_node_connectivity"] = (
                (("boundary"), mesh_dic["boundary"]),
            )

        std_ds = standardize_dataset(schism_ds, "SCHISM")
        logger.info("SCHISM mesh read successfully")
        return std_ds

    @classmethod
    def write(cls, std_ds: xr.Dataset, path: T.Union[str, pathlib.Path]) -> None:
        ds = format_specific_dataset(std_ds, "SCHISM")
        nn = ds.SCHISM_hgrid_node_x.size
        n3e = ds.nSCHISM_hgrid_face.size
        logging.info("writing SCHISM file", path)
        with open(path, "w") as f:
            f.write("\t uniform.gr3\n")
            f.write("\t {} {}\n".format(n3e, nn))

        q = ds[["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y", "depth"]].to_dataframe()

        q.index = np.arange(1, len(q) + 1)

        q.to_csv(
            path,
            index=True,
            sep="\t",
            header=None,
            mode="a",
            float_format="%.10f",
            columns=["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y", "depth"],
        )

        e = pd.DataFrame(
            ds.SCHISM_hgrid_face_nodes.dropna(dim="nMaxSCHISM_hgrid_face_nodes").values,
            columns=["a", "b", "c"],
        )

        e["nv"] = e.apply(lambda row: row.dropna().size, axis=1)

        e.index = np.arange(1, len(e) + 1)

        e = e.dropna(axis=1).astype(int)

        e.loc[:, ["a", "b", "c"]] = (
            e.loc[:, ["a", "b", "c"]] + 1
        )  # convert to fortran (index starts from 1)

        e.to_csv(
            path,
            index=True,
            sep="\t",
            header=None,
            mode="a",
            columns=["nv", "a", "b", "c"],
        )

        bs = ds[["node", "id", "type"]].to_dataframe()

        # open boundaries
        number_of_open_boundaries = bs.loc[bs.type == "open"].id
        if not number_of_open_boundaries.empty:
            number_of_open_boundaries = number_of_open_boundaries.max()
        else:
            number_of_open_boundaries = 0
        number_of_open_boundaries_nodes = bs.loc[bs.type == "open"].shape[0]

        if number_of_open_boundaries > 0:
            with open(path, "a") as f:
                f.write(
                    "{} = Number of open boundaries\n".format(number_of_open_boundaries)
                )
                f.write(
                    "{} = Total number of open boundary nodes\n".format(
                        number_of_open_boundaries_nodes
                    )
                )

                for i in range(1, number_of_open_boundaries + 1):
                    dat = bs.loc[bs.id == i, "node"] + 1  # fortran
                    f.write(
                        "{} = Number of nodes for open boundary {}\n".format(
                            dat.size, i
                        )
                    )
                    dat.to_csv(f, index=None, header=False)

        else:
            with open(path, "a") as f:
                f.write("{} = Number of open boundaries\n".format(0))
                f.write("{} = Total number of open boundary nodes\n".format(0))

        # land boundaries

        number_of_land_boundaries = bs.loc[bs.type == "land"].id
        if not number_of_land_boundaries.empty:
            number_of_land_boundaries = number_of_land_boundaries.max() - 1000
        else:
            number_of_land_boundaries = 0
        number_of_land_boundaries_nodes = bs.loc[bs.type == "land"].shape[0]

        number_of_island_boundaries = bs.loc[bs.type == "island"].id
        if not number_of_island_boundaries.empty:
            number_of_island_boundaries = number_of_island_boundaries.min()
        else:
            number_of_island_boundaries = 0
        number_of_island_boundaries_nodes = bs.loc[bs.type == "island"].shape[0]

        nlb = number_of_land_boundaries - number_of_island_boundaries
        nlbn = number_of_land_boundaries_nodes + number_of_island_boundaries_nodes

        if nlb > 0:
            with open(path, "a") as f:
                f.write("{} = Number of land boundaries\n".format(nlb))
                f.write("{} = Total number of land boundary nodes\n".format(nlbn))
                ik = 1
        else:
            with open(path, "a") as f:
                f.write("{} = Number of land boundaries\n".format(0))
                f.write("{} = Total number of land boundary nodes\n".format(0))

        if number_of_land_boundaries > 0:
            with open(path, "a") as f:
                for i in range(1001, 1000 + number_of_land_boundaries + 1):
                    dat_ = bs.loc[bs.id == i]
                    dat = dat_.node + 1  # fortran

                    f.write(
                        "{} {} = Number of nodes for land boundary {}\n".format(
                            dat.size, 0, ik
                        )
                    )
                    dat.to_csv(f, index=None, header=False)
                    ik += 1

        if number_of_island_boundaries < 0:
            with open(path, "a") as f:
                for i in range(-1, number_of_island_boundaries - 1, -1):
                    dat_ = bs.loc[bs.id == i]
                    dat = dat_.node + 1  # fortran

                    f.write(
                        "{} {} = Number of nodes for land boundary {}\n".format(
                            dat.size, 1, ik
                        )
                    )
                    dat.to_csv(f, index=None, header=False)
                    ik += 1
        logger.info("SCHISM mesh written successfully")
