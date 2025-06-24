"""MERRA2 provider loaders."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_merra2(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the MERRA2 data.

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the NWP data
    """
    ds = open_zarr_paths(zarr_path)
    da = ds["combined_variable"]
    init_time = da.coords["time"].values[0]
    da = da.expand_dims({"init_time_utc": [init_time]})
    da = da.rename({
        "feature": "channel",
        "time": "step",
        "lat": "latitude",
        "lon": "longitude",
    })
    da["step"] = da["step"] - init_time
    da = da.assign_coords(channel=ds["feature"].values)
    da = da.transpose("init_time_utc", "step", "channel", "latitude", "longitude")

    ds_new = da.to_dataset(name="aerosol_forecast")

    check_time_unique_increasing(ds_new.init_time_utc)
    ds_new = make_spatial_coords_increasing(ds_new, x_coord="longitude", y_coord="latitude")
    ds_new = ds_new.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    print(ds_new)

    return get_xr_data_array_from_xr_dataset(ds_new)


