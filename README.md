# seameshkit

A mesh format converter that transforms various ocean, coastal, and hydrodynamic model mesh formats into standardized UGRID convention within xarray datasets.
The converter tries to adhere to the latest CF (Climate and Forecast) conventions

##  Supported Formats

Currently supports conversion from the following formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| SCHISM | `.gr3` | Semi-implicit Cross-scale Hydroscience Integrated System Model format |
| TELEMAC | `.slf`, `.tel` | TELEMAC-MASCARET modeling system format (Selafin) |
| WAVEWATCH III | `.msh`, `.ww3` | NOAA wave model mesh format |
| GMSH | `.msh` | General-purpose finite element mesh generator format |

All formats are converted to UGRID conventions within xarray Dataset structure.

## Installation

```bash
git clone https://github.com/seareport/seameshkit.git
cd seameshkit
pip install ./
```

## Basic Usage

```python
from seameshkit import norm
from seameshkit.extract_contours import export_cli

file_msh = "tests/data/out_azov_0.3.msh"
ds = norm.read(file_msh)
```
returns:
```
<xarray.Dataset> Size: 1MB
Dimensions:                     (node: 12180, face: 21855, nmax_face: 3,
                                 time: 1, boundary: 2516)
Coordinates:
    node_x                      (node) float64 97kB 36.86 36.87 ... 37.54 36.88
    node_y                      (node) float64 97kB 45.45 45.44 ... 47.07 45.27
  * time                        (time) datetime64[ns] 8B 2025-05-19T06:58:03....
  * node                        (node) int64 97kB 0 1 2 3 ... 12177 12178 12179
  * face                        (face) int64 175kB 0 1 2 3 ... 21852 21853 21854
  * nmax_face                   (nmax_face) int64 24B 0 1 2
  * boundary                    (boundary) int64 20kB 0 1 2 3 ... 2513 2514 2515
Data variables:
    depth                       (node) float64 97kB 0.0 0.0 0.0 ... 0.0 0.0 0.0
    face_node_connectivity      (face, nmax_face) uint64 525kB 6696 ... 9317
    boundary_node_connectivity  (boundary) int64 20kB 1533 1534 ... 2451 2452
    id                          (boundary) int64 20kB -1 -1 -1 -1 ... -5 -5 -5
    type                        (boundary) <U6 60kB 'island' ... 'island'
```

## Export
### To TELEMAC
```python
file_tel = "tests/data/out_azov_0.3.slf"
file_cli = "tests/data/out_azov_0.3.cli"
norm.write(ds, file_tel, 'telemac')
export_cli(ds, file_cli)
```
### To SCHISM
```python
file_gr3 = "tests/data/out_azov_0.3.gr3"
norm.write(ds, file_gr3, 'schism')
```
