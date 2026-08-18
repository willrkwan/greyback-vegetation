from dataclasses import dataclass
import xarray as xr

@dataclass
class ClassifiedRasterResult:
    raster: xr.DataArray
    ndvi: xr.DataArray