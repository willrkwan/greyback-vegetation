
import geopandas as gpd
from shapely.geometry import Point, box, mapping
from datetime import date, timedelta

def get_bounding_box_geojson(lat, lon, half_side_km=0.5):
    """ Return a square bounding box in GeoJSON format around a point (lat, lon) with a given half side length in kilometers."""
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