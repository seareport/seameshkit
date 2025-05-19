import xarray as xr
import numpy as np
from xarray_selafin.xarray_backend import SelafinAccessor

import typing as T
import pathlib

from seameshkit._core import MeshAdapterRegistry
from seameshkit._core import MeshAdapter
from seameshkit._mappings import MESH_FORMATS
from seameshkit._utils import standardize_dataset
from seameshkit._utils import format_specific_dataset

import logging

logger = logging.getLogger(__name__)


@MeshAdapterRegistry.register(MESH_FORMATS.SELAFIN)
@MeshAdapterRegistry.register(MESH_FORMATS.TELEMAC)
class TELEMAC(MeshAdapter):
    @classmethod
    def can_read(cls, path: T.Union[str, pathlib.Path]) -> bool:
        result = str(path).endswith(".slf") or str(path).endswith(".srf")
        return result

    @classmethod
    def read(cls, path: T.Union[str, pathlib.Path]) -> xr.Dataset:
        ds = xr.open_dataset(path, engine="selafin")
        ds["face_node_connectivity"] = (("face", "nmax_face"), ds.attrs["ikle2"] - 1)
        std_ds = standardize_dataset(ds, "TELEMAC")
        logger.info("TELEMAC mesh read successfully")
        return std_ds

    @classmethod
    def write(cls, std_ds: xr.Dataset, path: T.Union[str, pathlib.Path]) -> None:
        ds = format_specific_dataset(std_ds, "TELEMAC")
        ds.selafin.write(path)
        logger.info("TELEMAC mesh written successfully")
