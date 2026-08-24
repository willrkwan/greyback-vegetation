import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from src import raster_service
from src.models import ChangeRasterResult, ClassifiedRasterResult, RasterConfig
from src.raster_service import NoClearPixelsError, NoScenesFoundError


def make_config() -> RasterConfig:
    return RasterConfig(
        catalog=object(),
        collection_id="test-collection",
        center_lat=49.5,
        center_lon=-119.5,
        half_side_km=2.0,
    )


class TestRasterService(unittest.TestCase):
    def test_build_classified_raster_smoke(self):
        service = raster_service.RasterService(make_config())
        footprint = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        ndvi = xr.DataArray(np.array([[0.10, 0.20], [0.30, 0.40]]), dims=("y", "x"))
        classified = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=("y", "x"))

        with (
            patch.object(raster_service, "get_bounding_box_geojson", return_value=footprint),
            patch.object(raster_service, "get_season_date_ranges", return_value={2023: "2023-07-01/2023-07-31"}),
            patch.object(raster_service, "search_stac_items", return_value=object()),
            patch.object(raster_service, "stack_items", return_value=object()),
            patch.object(
                raster_service,
                "build_landsat_cloud_mask",
                return_value=xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x")),
            ),
            patch.object(raster_service, "compute_normalized_difference", return_value=ndvi),
            patch.object(raster_service, "classify_by_thresholds", return_value=classified),
        ):
            result = service.build_classified_raster(2023)

        self.assertIsInstance(result, ClassifiedRasterResult)
        self.assertTrue(result.ndvi.equals(ndvi))
        self.assertTrue(result.raster.equals(classified))

    def test_build_change_raster_contract(self):
        service = raster_service.RasterService(make_config())
        base_ndvi = xr.DataArray(np.array([0.10, 0.20]), dims=("x",))
        target_ndvi = xr.DataArray(np.array([0.50, 0.60]), dims=("x",))
        base_raster = xr.DataArray(np.array([1, 2]), dims=("x",))
        target_raster = xr.DataArray(np.array([3, 4]), dims=("x",))

        base_result = ClassifiedRasterResult(raster=base_raster, ndvi=base_ndvi)
        target_result = ClassifiedRasterResult(raster=target_raster, ndvi=target_ndvi)

        with patch.object(
            service,
            "build_classified_raster",
            side_effect=lambda year: {
                2013: base_result,
                2023: target_result,
            }[year],
        ):
            result = service.build_change_raster(2013, 2023)

        self.assertIsInstance(result, ChangeRasterResult)
        self.assertIs(result.base_classified, base_result)
        self.assertIs(result.target_classified, target_result)
        np.testing.assert_allclose(result.ndvi_diff.values, target_ndvi.values - base_ndvi.values)

    def test_build_classified_raster_raises_when_no_scenes(self):
        service = raster_service.RasterService(make_config())

        with (
            patch.object(raster_service, "get_bounding_box_geojson", return_value={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}),
            patch.object(raster_service, "get_season_date_ranges", return_value={2023: "2023-07-01/2023-07-31"}),
            patch.object(raster_service, "search_stac_items", return_value=[]),
        ):
            with self.assertRaises(NoScenesFoundError):
                service.build_classified_raster(2023)

    def test_build_classified_raster_raises_when_no_clear_pixels(self):
        service = raster_service.RasterService(make_config())
        data = object()

        with (
            patch.object(raster_service, "get_bounding_box_geojson", return_value={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}),
            patch.object(raster_service, "get_season_date_ranges", return_value={2023: "2023-07-01/2023-07-31"}),
            patch.object(raster_service, "search_stac_items", return_value=[object()]),
            patch.object(raster_service, "stack_items", return_value=data),
            patch.object(raster_service, "build_landsat_cloud_mask", return_value=xr.DataArray(np.zeros((2, 2), dtype=bool), dims=("y", "x"))),
        ):
            with self.assertRaises(NoClearPixelsError):
                service.build_classified_raster(2023)


if __name__ == "__main__":
    unittest.main()
