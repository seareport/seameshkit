import xarray as xr
from collections import defaultdict, OrderedDict

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt


def sort_contours(contours):
    """
    Generate a dictionary of sorted contours where the key is the index of an
    outer contour, and the values are its corresponding inner contours.

    @param contours (list) List of contours.

    @return dict Dictionary of inner contour indices for each outer contour.
    """
    # Initialize an empty dictionary to store the result
    sorted_contours = {}
    all_inner_contours = []

    # Loop over all contours
    for i, contour in enumerate(contours):
        # Create a Path object for the outer contour
        outer_contour = Path(np.array(contour))

        # Initialize an empty list to store the indices of the inner contours
        inner_contours = []

        # Loop over all other contours
        for j, other_contour in enumerate(contours):
            # Skip if it's the same contour
            if i == j:
                continue

            # Create a Path object for the potential inner contour
            inner_contour = Path(np.array(other_contour))

            # Check if the outer contour contains the inner contour
            if outer_contour.contains_path(inner_contour):
                # If it does, add the index of the inner contour to the list
                # and remove it from the list of sorted contours
                inner_contours.append(j)

        all_inner_contours.extend(inner_contours)
        sorted_contours[i] = inner_contours

    for i in all_inner_contours:
        sorted_contours.pop(i, None)

    return sorted_contours


def signed_area(contour):
    """
    Calculate the signed area of a polygon using the Shoelace theorem.

    @param contour (list) List of vertex coordinates forming the polygon.
    @return float The signed area of the polygon.
    """
    x = np.array(contour)[:, 0]
    y = np.array(contour)[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def extract_boundaries(triangles):
    """
    Extract the boundaries of a mesh defined by a list of triangles.

    @param triangles (np.ndarray) A 1D array of triangles with each triangle
    being an array of 3 node IDs.

    @return np.ndarray A 1D array of edges.
    """
    # Extract and sort all edges
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]]),
        axis=1,
    ).astype(np.int64)

    # Use the Cantor pairing function to create edge IDs
    k1, k2 = edges[:, 0], edges[:, 1]
    edge_ids = (k1 + k2) * (k1 + k2 + 1) // 2 + k2

    # Extract unique edges
    edge_ids, indices, counts = np.unique(
        edge_ids, return_index=True, return_counts=True
    )
    indices = indices[counts == 1]
    return edges[indices]


def get_first_point(ds, bnd_points):
    """
    Determine the southwest points among points given trough their indexes

    @param tel (TelemacFile) a Telemac file
    @param bnd_points (numpy array of shape (number of points)) contains indexes
    of points
    @return first_bnd_pt_index (integer) index of the southwest point
    """

    x_plus_y = ds.node_x[bnd_points] + ds.node_y[bnd_points]

    southwest_bnd_pts_index = bnd_points[np.where(x_plus_y == x_plus_y.min())[0]]

    if southwest_bnd_pts_index.shape[0] == 1:
        first_bnd_pt_index = southwest_bnd_pts_index[0]
    else:
        first_bnd_pt_index = southwest_bnd_pts_index[
            np.where(
                ds.node_x[southwest_bnd_pts_index]
                == ds.node_x[southwest_bnd_pts_index].min()
            )[0][0]
        ]

    return first_bnd_pt_index


def extract_contour(ds: xr.Dataset):
    """
    Extract the contour of a given mesh.

    @param tel_file_path (str) Path to the mesh file

    @returns (list) A list of polygons, one for each domain.
    """
    vertices = np.vstack((ds.node_x, ds.node_y)).T

    # Extract boundary edges and nodes
    boundary_edges = extract_boundaries(ds.face_node_connectivity.values)
    boundary_nodes = set(np.unique(np.hstack(boundary_edges)))

    # Create a list of the neighbours of each node for quick search
    node_neighbours = defaultdict(set)
    for edge in boundary_edges:
        node_neighbours[edge[0]].add(edge[1])
        node_neighbours[edge[1]].add(edge[0])

    # Group boundary nodes coordinates into contours
    contours = []
    contours_idx = []
    while len(boundary_nodes) > 0:
        next_vertex = get_first_point(ds, np.array(list(boundary_nodes)))
        boundary_nodes.remove(next_vertex)
        contour_idx = [next_vertex]
        contour = [vertices[next_vertex].tolist()]
        while True:
            neighbours = node_neighbours[next_vertex].intersection(boundary_nodes)
            if len(neighbours) == 0:
                break
            next_vertex = neighbours.pop()
            boundary_nodes.remove(next_vertex)
            contour.append(vertices[next_vertex].tolist())
            contour_idx.append(next_vertex)

        # Ensure the contour is closed and append it to the list of contours
        contour.append(contour[0])
        contour_idx.append(contour_idx[0])
        contours.append(contour)
        contours_idx.append(contour_idx)

    # Build the list of domains while ensuring proper contours orientation
    sorted_contours = sort_contours(contours)
    domains = []
    domains_idx = []
    for outer, inners in sorted_contours.items():
        contour = contours[outer]
        contour_idx = contours_idx[outer]
        area = signed_area(contour)
        if area < 0:
            contour = contour[::-1]
            contour_idx = contour_idx[::-1]
        domains.append(contour)
        domains_idx.append(contour_idx)
        if len(inners) > 0:
            for i in inners:
                contour = contours[i]
                contour_idx = contours_idx[i]
                area = signed_area(contour)
                if area > 0:
                    contour = contour[::-1]
                    contour_idx = contour_idx[::-1]
                domains.append(contour)
                domains_idx.append(contour_idx)

    return domains, domains_idx


def get_contour_info(ds: xr.Dataset):
    contours, contours_idx = extract_contour(ds)
    type_, id_, nodes_ = [], [], []
    for ii, idx2 in enumerate(contours_idx):
        idx = np.array(idx2)
        nodes_.extend(idx)
        type_.extend(["island"] * len(idx))
        id_.extend([-1 - ii] * len(idx))
    return type_, id_, nodes_


def format_value(value, width, precision=3, is_float=False):
    if is_float:
        return f"{value:{5}.{precision}f}"
    else:
        return f"{value:{2}}"


def get_boundary_settings(boundary_type, glo_node, bnd_node):
    settings = {
        "lihbor": 5 if boundary_type == "open" else 2,
        "liubor": 6 if boundary_type == "open" else 2,
        "livbor": 6 if boundary_type == "open" else 2,
        "hbor": 0.0,
        "ubor": 0.0,
        "vbor": 0.0,
        "aubor": 0.0,
        "litbor": 5 if boundary_type == "open" else 2,
        "tbor": 0.0,
        "atbor": 0.0,
        "btbor": 0.0,
        "nbor": glo_node + 1,
        "k": bnd_node + 1,
    }
    return settings


def export_cli(ds: xr.Dataset, filename: str, tel_module: str = "telemac2d"):
    """
    (This function is a modification of the existing extract_contour() function
    in scripts/python3/pretel/extract_contour.py of the TELEMAC scripts)

    Generic function for extraction of contour from a mesh (with our without
    boundary file)

    @param inTel (str) Path to the mesh file
    @param ds (xr.Dataset) xarray Dataset of the mesh file (used to extract the boundary types)
    @param outCli (str) Path to the output contour file

    @returns (list) List of polygons
    """
    node_to_type = dict(zip(ds.boundary_node_connectivity.values, ds.type.values))
    domains_bnd = []
    lines = []
    bnd_node = 0
    contours, contours_idx = extract_contour(ds)
    for bnd in contours_idx:
        poly_bnd = []
        coord_bnd = []
        for i, glo_node in enumerate(
            bnd[:-1]
        ):  # not taking the last node (not repeating)
            x, y = ds.node_x[glo_node], ds.node_y[glo_node]
            coord_bnd.append((x, y))
            # Determine boundary type for the current node
            boundary_type = node_to_type.get(glo_node, "Unknown")
            if boundary_type == "open":
                # Get previous and next node indices in a circular manner
                prev_node = (
                    bnd[i - 1] if i > 0 else bnd[-2]
                )  # -2 to skip the repeated last node
                next_node = (
                    bnd[i + 1] if i < len(bnd) - 2 else bnd[0]
                )  # Wrapping around to the first node
                # Get boundary types for previous and next nodes
                prev_boundary_type = node_to_type.get(prev_node, "Unknown")
                next_boundary_type = node_to_type.get(next_node, "Unknown")
                # If both adjacent nodes are not 'open', then bnd is closed
                if prev_boundary_type != "open" and next_boundary_type != "open":
                    boundary_type = "Unknown"
            boundary_settings = get_boundary_settings(boundary_type, glo_node, bnd_node)

            keys_order = [
                "lihbor",
                "liubor",
                "livbor",
                "hbor",
                "ubor",
                "vbor",
                "aubor",
                "litbor",
                "tbor",
                "atbor",
                "btbor",
                "nbor",
                "k",
            ]
            if tel_module == "telemac2d":
                line = " ".join(str(boundary_settings[key]) for key in keys_order)
                lines.append(line)
            bnd_node += 1
        plt.plot(*zip(*coord_bnd))
        poly_bnd.append((coord_bnd, bnd))
    domains_bnd.append(poly_bnd)

    plt.show()
    # Writing to file
    with open(filename, "w") as f:
        for line in lines:
            formatted_line = " ".join(
                format_value(value, 3, is_float=isinstance(value, float))
                for value in line.split()
            )
            f.write(f"{formatted_line}\n")

    return domains_bnd
