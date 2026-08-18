from .ingest import (
    get_bounding_box_geojson, 
    get_season_date_ranges, 
    search_stac_items, 
    stack_items, 
    build_landsat_cloud_mask,
)
from .indices import compute_normalized_difference
from .classification import classify_by_thresholds
from .models import (
    ClassifiedRasterResult, 
    RasterConfig,
    ChangeRasterResult,
)


class RasterService:
    def __init__(self, config: RasterConfig):
        self.config = config

    def build_classified_raster(self, year: int) -> ClassifiedRasterResult:
        """Get a classified raster and its index for a given year and location."""
        footprint = get_bounding_box_geojson(
            self.config.center_lat, 
            self.config.center_lon, 
            self.config.half_side_km
        )
        date_range = get_season_date_ranges(
            year, 
            self.config.season_start_month, 
            self.config.season_start_day, 
            self.config.duration_days
        )[year]

        items = search_stac_items(
            self.config.catalog, 
            self.config.collection_id, 
            footprint, 
            date_range, 
            self.config.max_cloud
        )

        data = stack_items(
            items, 
            footprint, 
            bands=[self.config.ndvi_num_band, self.config.ndvi_den_band, self.config.qa_band], 
            resolution=self.config.resolution, 
            epsg=self.config.epsg
        )

        cloud_mask = build_landsat_cloud_mask(
            data, 
            qa_band=self.config.qa_band, 
            cloud_bits=self.config.cloud_bits
        )

        ndvi = compute_normalized_difference(
            data, 
            num_band=self.config.ndvi_num_band, 
            den_band=self.config.ndvi_den_band, 
            mask=cloud_mask, 
            reducer=self.config.reducer, 
            compute=self.config.compute
        )

        classified_raster = classify_by_thresholds(
            ndvi, 
            thresholds=self.config.thresholds, 
            class_values=self.config.class_values
        )

        return ClassifiedRasterResult(raster=classified_raster, ndvi=ndvi)


    def build_change_raster(self, base_year: int, target_year: int) -> ChangeRasterResult:
        """Get a change raster between two years for a given location."""
        base_result = self.build_classified_raster(base_year)
        target_result = self.build_classified_raster(target_year)

        ndvi_diff = target_result.ndvi - base_result.ndvi
       
        return ChangeRasterResult(ndvi_diff=ndvi_diff, base_classified=base_result, target_classified=target_result)