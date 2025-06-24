import numpy as np
from xarray import DataArray
#import pandas as pd

from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_ecmwf(nwp_ecmwf_zarr_path):
    da = open_nwp(zarr_path=nwp_ecmwf_zarr_path, provider="ecmwf")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert np.issubdtype(da.dtype, np.number)


def test_load_icon_eu(icon_eu_zarr_path):
    da = open_nwp(zarr_path=icon_eu_zarr_path, provider="icon-eu")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (2, 78, 2, 100, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_cloudcasting(nwp_cloudcasting_zarr_path):
    da = open_nwp(zarr_path=nwp_cloudcasting_zarr_path, provider="cloudcasting")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")
    assert "area" in da.attrs
    assert da.shape == (2, 12, 3, 100, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_merra2(merra2_zarr_path):

    da = open_nwp(merra2_zarr_path, provider="merra2")

    # Check shape and dimensions
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")

    assert da.shape[0] == 1
    assert da.shape[2] == 4
    assert da.shape[3] == 34
    assert da.shape[4] == 55

    # Check coordinate types
    assert np.issubdtype(da["step"].dtype, np.timedelta64)

    # Check channel names match expected
    expected_channels = ["ALBEDO", "DUSMASS", "TOTANGSTR", "TOTEXTTAU"]
    assert list(da.channel.values) == expected_channels

    # Check no NaNs or invalid values
    assert not np.isnan(da.values).all()

