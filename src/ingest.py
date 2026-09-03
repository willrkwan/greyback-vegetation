from datetime import date, timedelta

import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask
from shapely.geometry import Point, box, mapping, shape

import stackstac


def get_bounding_box_geojson(lat, lon, half_side_km=0.5):
    """Return a square bounding box in GeoJSON format around a point (lat, lon) with a given half side length in kilometers."""
    half_side_m = half_side_km * 1000

    wgs84_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_m = wgs84_point.to_crs(epsg=3857).iloc[0]

    bbox_m = box(
        point_m.x - half_side_m,
        point_m.y - half_side_m,
        point_m.x + half_side_m,
        point_m.y + half_side_m,
    )

    bbox_wgs84 = gpd.GeoSeries([bbox_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    return mapping(bbox_wgs84)

def get_season_date_ranges(years, season_start_month, season_start_day, duration_days):
    """Return {year: 'YYYY-MM-DD/YYYY-MM-DD'} for one year or an iterable of years."""
    if isinstance(years, int):
        years = [years]

    season_date_ranges = {}
    for year in years:
        start_dt = date(year, season_start_month, season_start_day)
        end_dt = start_dt + timedelta(days=duration_days - 1)
        season_date_ranges[year] = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    return season_date_ranges

def search_stac_items(catalog, collection_id, footprint, date_range, max_cloud=10):
    """Search a STAC catalog for items matching a footprint, date range, and cloud threshold.

    Returns a `pystac_client.ItemCollection`.
    """
    return catalog.search(
        collections=[collection_id],
        intersects=footprint,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        datetime=date_range,
    ).item_collection()

def stack_items(items, footprint, bands, resolution=10, epsg=32611, aoi_geometry=None):
    """Stack STAC item collection into a lazy xarray DataArray and mask it to an AOI."""
    footprint_geom = shape(footprint)
    data = stackstac.stack(
        items,
        assets=list(bands),
        resolution=resolution,
        bounds_latlon=footprint_geom.bounds,
        epsg=epsg,
    )

    if aoi_geometry is None:
        return data

    aoi_geometry_projected = gpd.GeoSeries(
        [shape(aoi_geometry)], crs="EPSG:4326"
    ).to_crs(epsg=epsg).iloc[0]
    mask = geometry_mask(
        [mapping(aoi_geometry_projected)],
        out_shape=(data.sizes["y"], data.sizes["x"]),
        transform=data.transform,
        invert=True,
    )
    aoi_mask = xr.DataArray(
        mask,
        dims=("y", "x"),
        coords={"y": data.y, "x": data.x},
    )
    return data.where(aoi_mask)

def build_landsat_cloud_mask(raw_stack, qa_band="qa_pixel", cloud_bits=(1, 3, 4)):
    """Create a boolean mask where True means clear-sky pixel. Bands 1, 3, and 4 of the 
    Landsat QA_PIXEL band indicate cloud, cloud shadow, and snow/ice respectively.
    """
    qa = raw_stack.sel(band=qa_band).astype("uint16")
    bitmask = 0
    for bit in cloud_bits:
        bitmask |= qa & (1 << bit)
    return bitmask == 0