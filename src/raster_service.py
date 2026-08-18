from .ingest import (
    get_bounding_box_geojson, 
    get_season_date_ranges, 
    search_stac_items, 
    stack_items, 
    build_landsat_cloud_mask
)
from .indices import compute_normalized_difference
from .classification import classify_by_thresholds
from .models import ClassifiedRasterResult

def build_classified_raster(
    year: int,
    *,
    catalog,
    collection_id: str,
    center_lat: float,
    center_lon: float,
    half_side_km: float = 5.0,
    season_start_month: int = 7,
    season_start_day: int = 1,
    duration_days: int = 31,
    max_cloud: float = 10,
    resolution: int = 30,
    epsg: int = 32611,
    ndvi_num_band: str = "nir08",
    ndvi_den_band: str = "red",
    qa_band: str = "qa_pixel",
    cloud_bits: tuple[int, ...] = (1, 3, 4),
    reducer: str = "median",
    thresholds: tuple[float, ...] = (-0.05, 0.10, 0.20, 0.35, 0.50),
    class_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    compute: bool = True,
) -> ClassifiedRasterResult:
    """Get a classified raster and its index for a given year and location."""
    footprint = get_bounding_box_geojson(center_lat, center_lon, half_side_km)
    date_range = get_season_date_ranges(year, season_start_month, season_start_day, duration_days)[year]

    items = search_stac_items(catalog, collection_id, footprint, date_range, max_cloud)

    data = stack_items(items, footprint, bands=[ndvi_num_band, ndvi_den_band, qa_band], resolution=resolution, epsg=epsg)

    cloud_mask = build_landsat_cloud_mask(data, qa_band=qa_band, cloud_bits=cloud_bits)

    ndvi_index = compute_normalized_difference(data, num_band=ndvi_num_band, den_band=ndvi_den_band, mask=cloud_mask, reducer=reducer, compute=compute)

    classified_raster = classify_by_thresholds(ndvi_index, thresholds=thresholds, class_values=class_values)

    return classified_raster, ndvi_index

def build_change_raster(base_year, target_year):
    """Get a change raster between two years for a given location."""
    raise NotImplementedError("Change raster functionality is not yet implemented.")