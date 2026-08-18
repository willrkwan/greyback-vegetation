
import geopandas as gpd
from shapely.geometry import Point, box, mapping

def get_bounding_box_geojson(lat, lon, half_side_km=0.5):
    half_side_m = half_side_km * 1000

    # Point in WGS84 (lon/lat), then project to meter-based CRS
    wgs84_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_m = wgs84_point.to_crs(epsg=3857).iloc[0]

    # Build square bounding box in meters around the projected point
    bbox_m = box(
        point_m.x - half_side_m,
        point_m.y - half_side_m,
        point_m.x + half_side_m,
        point_m.y + half_side_m,
    )

    # Reproject back to WGS84 and return GeoJSON geometry
    bbox_wgs84 = gpd.GeoSeries([bbox_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    return mapping(bbox_wgs84)