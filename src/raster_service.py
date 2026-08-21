from .classification import classify_by_thresholds
from .indices import compute_normalized_difference
from .ingest import (
    get_bounding_box_geojson, 
    get_season_date_ranges, 
    search_stac_items, 
    stack_items, 
    build_landsat_cloud_mask,
)
from .models import (
    ClassifiedRasterResult, 
    RasterConfig,
    ChangeRasterResult,
)


class RasterServiceError(RuntimeError):
    """Base class for raster service data availability errors."""


class NoScenesFoundError(RasterServiceError):
    """Raised when a selected year has no matching STAC scenes."""


class NoClearPixelsError(RasterServiceError):
    """Raised when cloud masking removes all pixels for a given scene set."""


class RasterService:
    def __init__(self, config: RasterConfig):
        self.config = config

    def build_classified_raster(self, year: int) -> ClassifiedRasterResult:
        """Get a classified raster and its index for a given year."""
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
        if not items:
            raise NoScenesFoundError(
                f"No STAC scenes found for year {year} in the selected seasonal window. "
                "Try a different year or a wider date range."
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
        if cloud_mask is None or not bool(cloud_mask.any()):
            raise NoClearPixelsError(
                f"No usable clear-sky pixels remained for year {year} after masking. "
                "Try a higher cloud threshold or a different season."
            )

        ndvi = compute_normalized_difference(
            data, 
            num_band=self.config.ndvi_num_band, 
            den_band=self.config.ndvi_den_band, 
            mask=cloud_mask, 
            reducer=self.config.reducer, 
            compute=self.config.compute
        )
        if ndvi is None or ndvi.size == 0:
            raise NoClearPixelsError(
                f"The NDVI result for year {year} contained no valid pixels after filtering."
            )

        classified_raster = classify_by_thresholds(
            ndvi, 
            thresholds=self.config.thresholds, 
            class_values=self.config.class_values
        )

        return ClassifiedRasterResult(raster=classified_raster, ndvi=ndvi)


    def build_change_raster(self, base_year: int, target_year: int) -> ChangeRasterResult:
        """Get a change raster comparing NDVI between two years, and each year's classified raster and index."""
        base_result = self.build_classified_raster(base_year)
        target_result = self.build_classified_raster(target_year)

        ndvi_diff = target_result.ndvi - base_result.ndvi
       
        return ChangeRasterResult(ndvi_diff=ndvi_diff, base_classified=base_result, target_classified=target_result)