from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import xarray as xr

from ._mappings import get_mappings_for_format
from ._mappings import get_transforms_for_format
from ._mappings import get_attrs_for_format
from ._mappings import get_suppress_vars_for_format
from .extract_contours import get_contour_info

logger = logging.getLogger(__name__)


def standardize_dataset(ds: xr.Dataset, format_name: str) -> xr.Dataset:
    """
    Convert a format-specific dataset to the standard internal representation.

    Args:
        ds: Input dataset in format-specific variables
        format_name: Name of the format

    Returns:
        Dataset with standardized variable names
    """
    mappings = get_mappings_for_format(format_name)
    transforms = get_transforms_for_format(format_name)
    std_ds = ds.copy()

    for var_name, (to_std_fn, _) in transforms.items():
        if var_name in std_ds:
            std_ds[var_name] = to_std_fn(std_ds)

    rename_dict = {
        fmt_var: std_var for fmt_var, std_var in mappings.items() if fmt_var in std_ds
    }

    if rename_dict:
        std_ds = std_ds.rename(rename_dict)

    if "node_x" in std_ds and "node_y" in std_ds:
        std_ds = std_ds.assign_coords(
            {"node_x": std_ds.node_x, "node_y": std_ds.node_y}
        )

    if "time" not in std_ds and "depth" in std_ds:
        if "node" in std_ds.dims and len(std_ds.depth.dims) == 1:
            std_ds["depth"] = (("time", "node"), [std_ds.depth.values])
            std_ds["time"] = [pd.Timestamp.now()]

    if (
        "boundary_node_connectivity" not in std_ds
        and "node_x" in std_ds
        and "node_y" in std_ds
        and "face_node_connectivity" in std_ds
    ):
        try:
            type_, id_, bnd_ = get_contour_info(std_ds)
            std_ds["boundary_node_connectivity"] = (("boundary"), bnd_)
            std_ds["id"] = ("boundary", id_)
            std_ds["type"] = ("boundary", type_)
        except Exception as e:

            print(f"Could not extract boundary info: {e}")
            logger.warning(f"Could not extract boundary info: {e}")

    # Add synthetic coordinate variables for dimensions without coordinates
    if "node" in std_ds.dims and "node" not in std_ds.coords:
        std_ds = std_ds.assign_coords(node=("node", np.arange(std_ds.sizes["node"])))

    if "face" in std_ds.dims and "face" not in std_ds.coords:
        std_ds = std_ds.assign_coords(face=("face", np.arange(std_ds.sizes["face"])))

    if "nmax_face" in std_ds.dims and "nmax_face" not in std_ds.coords:
        std_ds = std_ds.assign_coords(
            nmax_face=("nmax_face", np.arange(std_ds.sizes["nmax_face"]))
        )

    if "boundary" in std_ds.dims and "boundary" not in std_ds.coords:
        std_ds = std_ds.assign_coords(
            boundary=("boundary", np.arange(std_ds.sizes["boundary"]))
        )

    return std_ds


def format_specific_dataset(ds: xr.Dataset, format_name: str) -> xr.Dataset:
    """
    Convert a standardized dataset to a format-specific representation.

    Args:
        ds: Input dataset with standard variable names
        format_name: Target format name

    Returns:
        Dataset with format-specific variable names and transformations
    """
    mappings = get_mappings_for_format(format_name)
    transforms = get_transforms_for_format(format_name)
    format_attrs = get_attrs_for_format(format_name)
    suppress_vars = get_suppress_vars_for_format(format_name)

    reverse_mappings = {std_var: fmt_var for fmt_var, std_var in mappings.items()}
    fmt_ds = ds.copy()

    rename_dict = {
        std_var: fmt_var
        for std_var, fmt_var in reverse_mappings.items()
        if std_var in fmt_ds
    }

    if rename_dict:
        fmt_ds = fmt_ds.rename(rename_dict)
    for std_var, (_, from_std_fn) in transforms.items():
        reverse_var = reverse_mappings.get(std_var, std_var)
        if std_var in ds:
            fmt_ds[reverse_var] = from_std_fn(ds)

    for attr_name, attr_fn in format_attrs.items():
        fmt_ds.attrs[attr_name] = attr_fn(ds)

    existing_suppress_vars = [v for v in suppress_vars if v in fmt_ds]
    if existing_suppress_vars:
        fmt_ds = fmt_ds.drop_vars(existing_suppress_vars)

    return fmt_ds
