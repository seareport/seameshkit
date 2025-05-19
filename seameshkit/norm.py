from __future__ import annotations

import typing as T

import pathlib
import logging

from ._core import MeshAdapterRegistry
from ._mappings import normalize_format
from ._mappings import MESH_FORMATS
from . import _io

import xarray as xr

logger = logging.Logger(__name__, level=logging.INFO)


def read(path: T.Union[str, pathlib.Path], force: bool = False) -> xr.Dataset:
    """
    Read a mesh file in any supported format.

    This function attempts to read the mesh using all registered adapters
    until one succeeds or all fail.

    Args:
        path: Path to the mesh file

    Returns:
        xarray Dataset in standard format

    Raises:
        ValueError: If the mesh cannot be read
    """
    path = pathlib.Path(path) if isinstance(path, str) else path
    logger.debug(f"Trying to open: {path}")

    # Try to determine format from file extension
    extension = path.suffix.lower()
    format_candidates = []
    errors = {}

    for format_type in MeshAdapterRegistry.list_formats():
        adapter_cls = MeshAdapterRegistry.get_adapter(format_type)
        try:
            if adapter_cls.can_read(path):
                format_candidates.append(adapter_cls)
        except Exception:
            if force:
                format_candidates.append(adapter_cls)

    for adapter_cls in format_candidates:
        try:
            ds = adapter_cls.read(path)
            return ds
        except Exception as e:
            errors[adapter_cls.__name__] = str(e)

    # If we get here, all adapters failed
    error_msg = "\n".join([f"{fmt}: {err}" for fmt, err in errors.items()])
    raise ValueError(f"Failed to read mesh file {path}. Errors:\n{error_msg}")


def write(
    ds: xr.Dataset, path: T.Union[str, pathlib.Path], fmt: T.Union[MESH_FORMATS, str]
) -> None:
    """
    Write a mesh dataset to a file in the specified format.

    Args:
        ds: xarray Dataset in standard format
        path: Path to write the mesh file
        fmt: Format to write the mesh in

    Raises:
        UnsupportedFormatError: If the format is not supported
        MeshError: If the mesh cannot be written
    """
    fmt = normalize_format(fmt)
    adapter_cls = MeshAdapterRegistry.get_adapter(fmt)
    if not adapter_cls:
        raise ValueError(f"Unsupported mesh format: {fmt}")
    adapter_cls.write(ds, path)


def convert(
    input_path: T.Union[str, pathlib.Path],
    output_path: T.Union[str, pathlib.Path],
    output_format: T.Union[MESH_FORMATS, str],
) -> None:
    """
    Convert a mesh file from one format to another.

    Args:
        input_path: Path to the input mesh file
        output_path: Path to write the output mesh file
        output_format: Format to write the output mesh in

    Raises:
        MeshError: If the conversion fails
    """
    try:
        ds = read(input_path)
        write(ds, output_path, output_format)
        logger.info(
            f"Converted {input_path} to {output_path} in {output_format} format"
        )

    except Exception as e:
        raise ValueError(f"Failed to convert mesh: {e}")


def list_supported_formats() -> T.List[str]:
    return [fmt.name for fmt in MeshAdapterRegistry.list_formats()]


# def read(path: str | pathlib.Path) -> xr.Dataset:
#     logger.debug("Trying to open: %s", path)
#     errors = {}
#     errors["success"] = False
#     try:
#         ds = ww3.read_gmsh(path)
#         logger.info("WW3 mesh read successfully")
#         errors["success"] = True
#         return ds
#     except Exception as e:
#         errors["ww3"] = str(e)
#     try:
#         ds = telemac.read_selafin(path)
#         ds = ds.rename({
#             "x": "node_x",
#             "y": "node_y",
#             "B": "depth"
#             })
#         ds["face_node_connectivity"] = (("face", "nmax_face"), ds.attrs['ikle2'] - 1)
#         # get boundaries
#         type_, id_, bnd_ = get_contour_info(ds)
#         ds["boundary_node_connectivity"] = (("boundary"), bnd_)
#         ds["id"] = ("boundary", id_)
#         ds["type"] = ("boundary", type_)
#         ds = ds.assign({"depth": -ds.depth})
#         logger.info("Telemac mesh read successfully")
#         errors["success"] = True
#         return ds
#     except Exception as e:
#         errors["telemac"] = str(e)
#     try:
#         ds = schism.read_gr3(path)
#         ds["boundary_node_connectivity"] = (("bnodes"), ds.node.values)
#         ds = ds.drop_vars("node")
#         ds = ds.rename({
#             "nSCHISM_hgrid_node": "node",
#             "SCHISM_hgrid_node_x": "node_x",
#             "SCHISM_hgrid_node_y": "node_y",
#             "nSCHISM_hgrid_face": "face",
#             "nMaxSCHISM_hgrid_face_nodes": "nmax_face",
#             "SCHISM_hgrid_face_nodes": "face_node_connectivity",
#         })
#         # get boundaries
#         type_, id_, bnd_ = get_contour_info(ds)
#         ds["boundary_node_connectivity"] = (("boundary"), bnd_)
#         ds["id"] = ("boundary", id_)
#         ds["type"] = ("boundary", type_)
#         ds["depth"] = (("time", "node"), [ds.depth])
#         ds["time"] = [pd.Timestamp.now()]
#         ds = ds.drop_vars("bnodes")
#         logger.info("Schism mesh read successfully")
#         errors["success"] = True
#         return ds
#     except Exception as e:
#         errors["schism"] = str(e)
#     try:
#         ds = ugrid.read_ugrid(path)
#         logger.info("UGRID mesh read successfully")
#         errors["success"] = True
#         return ds
#     except Exception as e:
#         errors["ugrid"] = str(e)
#     try:
#         mesh = meshio.read(path)
#         ds = xr.Dataset(
#             coords={
#                 "node_x": (["node"], mesh.points[:,0]),
#                 "node_y": (["node"], mesh.points[:,1]),
#                 "time":[pd.Timestamp.now()],
#             },
#             data_vars={
#                 "face_node_connectivity": (["face", "nmax_face"], mesh.cells),
#                 "depth": (["node"], mesh.points[:,2]),
#             },
#         )
#         type_, id_, bnd_ = get_contour_info(ds)
#         ds["boundary_node_connectivity"] = (("boundary"), bnd_)
#         ds["id"] = ("boundary", id_)
#         ds["type"] = ("boundary", type_)
#         logger.info("MESHIO mesh read successfully")
#         errors["success"] = True
#         return ds

#     except Exception as e:
#         errors["meshio"] = str(e)
#     try:
#         ds = gmsh.read_msh(path)
#         ds = ds.rename({
#             "nSCHISM_hgrid_node": "node",
#             "SCHISM_hgrid_node_x": "node_x",
#             "SCHISM_hgrid_node_y": "node_y",
#             "nSCHISM_hgrid_face": "face",
#             "nMaxSCHISM_hgrid_face_nodes": "nmax_face",
#             "SCHISM_hgrid_face_nodes": "face_node_connectivity",
#         })
#         ds = ds.assign_coords({"node_x": ds.node_x, "node_y": ds.node_y})
#         # get boundaries
#         type_, id_, bnd_ = get_contour_info(ds)
#         ds["boundary_node_connectivity"] = (("boundary"), bnd_)
#         ds["id"] = ("boundary", id_)
#         ds["type"] = ("boundary", type_)
#         ds["depth"] = (("time", "node"), [ds.depth])
#         ds["time"] = [pd.Timestamp.now()]
#         logger.info("GMSH mesh read successfully")
#         errors["success"] = True
#         return ds
#     except Exception as e:
#         errors["gmsh"] = str(e)

#     if not errors["success"]:
#         logger.error("Unable to read mesh from %s: %s", path, errors)


# def write(ds: xr.Dataset, path: str | pathlib.Path, fmt: MESH_FORMATS) -> None:
#     fmt = normalize_format(fmt)
#     if fmt == MESH_FORMATS.WW3:
#         ww3.write_gmsh_mesh(ds, path)
#     elif fmt == MESH_FORMATS.SCHISM:
#         ds = ds.rename({
#             "node_x": "SCHISM_hgrid_node_x",
#             "node_y": "SCHISM_hgrid_node_y",
#             "node":"nSCHISM_hgrid_node",
#             "face_node_connectivity": "SCHISM_hgrid_face_nodes",
#             "nmax_face": "nMaxSCHISM_hgrid_face_nodes",
#             "face": "nSCHISM_hgrid_face",
#         })
#         idx = ds["boundary_node_connectivity"].values
#         ds = ds.drop_vars("boundary_node_connectivity")
#         ds["id"] = ("bnodes", [-1]*(len(idx)))
#         ds["node"] = ("bnodes", idx)
#         ds["type"] = ("bnodes", ["island"]*len(idx))
#         schism.write_gr3(ds, path)
#         logger.info("SCHISM mesh written successfully")
#     elif fmt == MESH_FORMATS.TELEMAC or fmt == MESH_FORMATS.SELAFIN:
#         ds = ds.rename({
#             "node_x": "x",
#             "node_y": "y",
#             "depth": "B",
#         })
#         ds = ds.assign_coords({"x": ds.x, "y": ds.y})
#         ds = ds.assign({"B": -ds.B})
#         ds.attrs['ikle2'] = ds.face_node_connectivity.values + 1
#         ds = ds.drop_vars("face_node_connectivity")
#         ds = ds.drop_vars("boundary_node_connectivity")
#         ds = ds.drop_vars("type")
#         ds = ds.drop_vars("id")
#         telemac.to_slf(ds, path)
#     elif fmt == MESH_FORMATS.UGRID:
#         ugrid.write_ugrid(ds, path)
#     else:
#         print(f"Unsupported mesh format: {fmt}, choose between {MESH_FORMATS}")
#         raise ValueError("Unsupported mesh format")
