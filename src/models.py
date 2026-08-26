from dataclasses import dataclass
from typing import Optional

import pystac_client
import xarray as xr

@dataclass
class ClassifiedRasterResult:
    raster: xr.DataArray
    ndvi: xr.DataArray

@dataclass
class ChangeRasterResult:
    ndvi_diff: xr.DataArray
    base_classified: ClassifiedRasterResult
    target_classified: ClassifiedRasterResult


@dataclass(frozen=True)
class BandProfile:
    name: str
    bands: tuple[str, ...]
    kind: str
    apply_cloud_mask: bool = True
    numerator_band: str | None = None
    denominator_band: str | None = None


@dataclass(frozen=True)
class SceneSummary:
    scene_id: str
    acquired_at: str
    cloud_cover: Optional[float]
    platform: Optional[str]
    available_bands: frozenset[str]

@dataclass(frozen=True)
class RasterConfig:
    catalog: pystac_client.Client
    collection_id: str

    center_lat: float
    center_lon: float

    half_side_km: float = 5.0
    aoi_geometry: dict | None = None

    season_start_month: int = 7
    season_start_day: int = 1
    duration_days: int = 31

    max_cloud: float = 10.0

    resolution: int = 30
    epsg: int = 32611

    ndvi_num_band: str = "nir08"
    ndvi_den_band: str = "red"
    nbr_num_band: str = "nir08"
    nbr_den_band: str = "swir22"
    red_band: str = "red"
    green_band: str = "green"
    blue_band: str = "blue"
    qa_band: str = "qa_pixel"
    cloud_bits: tuple[int, ...] = (1, 3, 4)

    reducer: str = "median"

    thresholds: tuple[float, ...] = (-0.05, 0.10, 0.20, 0.35, 0.50)
    class_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    compute: bool = True