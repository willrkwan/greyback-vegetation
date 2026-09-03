from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AOI_PATH = PROJECT_ROOT / "data" / "aoi" / "greyback_assessment_aoi.geojson"
AOI_CRS = "EPSG:4326"


def load_aoi_geometry(path: Path = AOI_PATH) -> dict:
    """Load the configured local AOI and return its GeoJSON geometry."""
    if not path.exists():
        raise FileNotFoundError(
            f"AOI file not found: {path}. Run notebooks/watersheds.ipynb first."
        )

    aoi = gpd.read_file(path)
    if aoi.empty:
        raise ValueError(f"AOI file contains no features: {path}")
    if aoi.crs is None:
        raise ValueError(f"AOI file has no CRS: {path}")

    aoi = aoi.to_crs(AOI_CRS)
    return mapping(aoi.geometry.iloc[0])
