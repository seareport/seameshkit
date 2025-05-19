import pytest
from seameshkit import norm

FILES = pytest.mark.parametrize(
    "file",
    [
        pytest.param("tests/data/uglo_100km_BlkS_360.ww3", id="ww3"),
    ],
)


@FILES
def test_read_ww3(tmp_path, file):
    ds = norm.read(file)
    out = tmp_path / "test.ww3"
    fmt = "ww3"
    norm.write(ds, out, fmt)


@FILES
def test_circular(file):
    ds = norm.read(file)
    out = "test.slf"
    fmt = "selafin"
    norm.write(ds, out, fmt)
    ds = norm.read(out)
    out = "test.gr3"
    fmt = "schism"
    norm.write(ds, out, fmt)
    ds = norm.read(out)
    out = "test.ugrid"
    fmt = "ugrid"
    norm.write(ds, out, fmt)
    ds = norm.read(out)
    out = "test.ww3"
    fmt = "ww3"
    norm.write(ds, out, fmt)
    ds = norm.read(out)
    ds_ref = norm.read(file)
    assert ds.equals(ds_ref)
