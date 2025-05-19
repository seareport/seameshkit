from __future__ import annotations

import enum
import typing as T


class MESH_FORMATS(enum.Enum):
    UNKNOWN = "UNKNOWN"
    MESHIO = "MESHIO"
    UGRID = "UGRID"
    GMSH = "GMSH"
    SCHISM = "SCHISM"
    GR3 = "GR3"
    TELEMAC = "TELEMAC"
    SELAFIN = "SELAFIN"
    WW3 = "WW3"


STANDARD_VARS = {
    "node_x": "X-coordinates of mesh nodes",
    "node_y": "Y-coordinates of mesh nodes",
    "node": "Node indices",
    "face": "Face indices",
    "nmax_face": "Maximum number of nodes per face",
    "face_node_connectivity": "Connectivity between faces and nodes",
    "boundary_node_connectivity": "Nodes that form the boundary",
    "depth": "Depth values at nodes",
    "id": "Boundary segment IDs",
    "type": "Boundary segment types",
    "time": "Time dimension",
}

VAR_MAPPINGS = {
    "SCHISM": {
        "nSCHISM_hgrid_node": "node",
        "SCHISM_hgrid_node_x": "node_x",
        "SCHISM_hgrid_node_y": "node_y",
        "nSCHISM_hgrid_face": "face",
        "nMaxSCHISM_hgrid_face_nodes": "nmax_face",
        "SCHISM_hgrid_face_nodes": "face_node_connectivity",
        "node": "boundary_node_connectivity",  # When in boundary context
    },
    "TELEMAC": {
        "x": "node_x",
        "y": "node_y",
        "B": "depth",
    },
    "GMSH": {},
    "WW3": {},
    "UGRID": {},
    "MESHIO": {},
}

VAR_TRANSFORMS = {
    "TELEMAC": {
        "depth": (
            lambda ds: -ds.B,  # To standard (negate B to get depth)
            lambda ds: -ds.depth,  # From standard (negate depth to get B)
        ),
        "face_node_connectivity": (
            lambda ds: (
                ("face", "nmax_face"),
                ds.attrs["ikle2"] - 1,
            ),  # To standard (convert 1-indexed to 0-indexed)
            lambda ds: ds.face_node_connectivity
            + 1,  # From standard (convert 0-indexed to 1-indexed)
        ),
    },
    "SCHISM": {},
    "GMSH": {},
}

# Additional attributes needed for specific formats
FORMAT_ATTRS = {
    "TELEMAC": {"ikle2": lambda ds: ds.face_node_connectivity.values + 1},
}

SUPPRESS_VARS = {
    "TELEMAC": [
        "face_node_connectivity",
        "boundary_node_connectivity",
        "id",
        "type",
        "boundary",
        "node_x",
        "node_y",
    ]
}


def normalize_format(fmt) -> MESH_FORMATS:
    """Convert string format name to MESH_FORMATS enum."""
    if isinstance(fmt, MESH_FORMATS):
        return fmt
    elif isinstance(fmt, str):
        try:
            return MESH_FORMATS[fmt.upper()]
        except KeyError:
            pass
    raise ValueError(
        f"Unsupported mesh format: {fmt}, choose between {[f.name for f in MESH_FORMATS]}"
    )


def get_mappings_for_format(fmt: T.Union[MESH_FORMATS, str]) -> T.Dict[str, str]:
    fmt = normalize_format(fmt) if isinstance(fmt, str) else fmt
    return VAR_MAPPINGS.get(fmt.name, {})


def get_transforms_for_format(fmt: T.Union[MESH_FORMATS, str]) -> T.Dict[str, tuple]:
    fmt = normalize_format(fmt) if isinstance(fmt, str) else fmt
    return VAR_TRANSFORMS.get(fmt.name, {})


def get_attrs_for_format(fmt: T.Union[MESH_FORMATS, str]) -> T.Dict[str, T.Any]:
    """Get required attributes for a specific format."""
    fmt = normalize_format(fmt) if isinstance(fmt, str) else fmt
    return FORMAT_ATTRS.get(fmt.name, {})


def get_suppress_vars_for_format(fmt: T.Union[MESH_FORMATS, str]) -> T.Dict[str, T.Any]:
    """Get required attributes for a specific format."""
    fmt = normalize_format(fmt) if isinstance(fmt, str) else fmt
    return SUPPRESS_VARS.get(fmt.name, {})
