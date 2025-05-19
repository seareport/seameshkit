import xarray as xr


def read_ugrid(filename):
    return xr.open_dataset(filename, engine="netcdf4")


def write_ugrid(ds, filename):
    ds.to_netcdf(filename, mode="w", format="NETCDF4")
