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
    SceneSummary,
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

    def _build_footprint(self):
        if self.config.aoi_geometry is not None:
            return self.config.aoi_geometry

        return get_bounding_box_geojson(
            self.config.center_lat, 
            self.config.center_lon, 
            self.config.half_side_km
        )

    def _build_date_range(self, year: int) -> str:
        return get_season_date_ranges(
            year, 
            self.config.season_start_month, 
            self.config.season_start_day, 
            self.config.duration_days
        )[year]

    def _search_items_for_year(self, year: int):
        footprint = self._build_footprint()
        date_range = self._build_date_range(year)

        items = search_stac_items(
            self.config.catalog, 
            self.config.collection_id, 
            footprint, 
            date_range, 
            self.config.max_cloud
        )
        return footprint, items

    def list_scene_summaries(self, year: int) -> list[SceneSummary]:
        """List available scenes for a year using STAC metadata only."""
        _, items = self._search_items_for_year(year)
        summaries: list[SceneSummary] = []

        for item in items:
            assets = set(item.assets.keys())
            cloud_cover = item.properties.get("eo:cloud_cover")
            summaries.append(
                SceneSummary(
                    scene_id=item.id,
                    acquired_at=str(item.datetime) if item.datetime is not None else "",
                    cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
                    platform=item.properties.get("platform"),
                    has_ndvi_bands=self.config.ndvi_num_band in assets and self.config.ndvi_den_band in assets and self.config.qa_band in assets,
                    has_rgb_bands=self.config.ndvi_den_band in assets and self.config.green_band in assets and self.config.blue_band in assets,
                )
            )

        return summaries

    @staticmethod
    def _select_scene(items, scene_id: str | None):
        if scene_id is None:
            return items
        return [item for item in items if item.id == scene_id]

    @staticmethod
    def _filter_items_by_assets(items, required_assets: set[str]):
        return [item for item in items if required_assets.issubset(set(item.assets.keys()))]

    def build_ndvi_raster(self, year: int, scene_id: str | None = None):
        """Build an NDVI raster for a year or a selected scene within that year."""
        footprint, items = self._search_items_for_year(year)
        items = self._select_scene(items, scene_id)
        required_assets = {self.config.ndvi_num_band, self.config.ndvi_den_band, self.config.qa_band}
        items = self._filter_items_by_assets(items, required_assets)
        if not items:
            raise NoScenesFoundError(
                f"No NDVI-eligible scenes found for year {year} in the selected seasonal window "
                f"for scene_id={scene_id!r}. Try a different selection or a wider date range."
            )

        data = stack_items(
            items, 
            footprint, 
            bands=[self.config.ndvi_num_band, self.config.ndvi_den_band, self.config.qa_band], 
            resolution=self.config.resolution, 
            epsg=self.config.epsg,
            aoi_geometry=self.config.aoi_geometry,
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

        return ndvi

    def build_rgb_raster(self, year: int, scene_id: str | None = None, apply_cloud_mask: bool = False):
        """Build an RGB raster for a year or a selected scene within that year.

        RGB visualization keeps cloudy pixels by default so the scene footprint remains visible.
        """
        footprint, items = self._search_items_for_year(year)
        items = self._select_scene(items, scene_id)
        required_assets = {self.config.ndvi_den_band, self.config.green_band, self.config.blue_band}
        if apply_cloud_mask:
            required_assets.add(self.config.qa_band)
        items = self._filter_items_by_assets(items, required_assets)
        if not items:
            raise NoScenesFoundError(
                f"No RGB-eligible scenes found for year {year} in the selected seasonal window "
                f"for scene_id={scene_id!r}. Try a different selection or a wider date range."
            )

        stack_bands = [self.config.ndvi_den_band, self.config.green_band, self.config.blue_band]
        if apply_cloud_mask:
            stack_bands.append(self.config.qa_band)

        data = stack_items(
            items,
            footprint,
            bands=stack_bands,
            resolution=self.config.resolution,
            epsg=self.config.epsg,
            aoi_geometry=self.config.aoi_geometry,
        )

        rgb = data.sel(band=[self.config.ndvi_den_band, self.config.green_band, self.config.blue_band])
        if apply_cloud_mask:
            cloud_mask = build_landsat_cloud_mask(
                data,
                qa_band=self.config.qa_band,
                cloud_bits=self.config.cloud_bits,
            )
            if cloud_mask is None or not bool(cloud_mask.any()):
                raise NoClearPixelsError(
                    f"No usable clear-sky pixels remained for year {year} after masking. "
                    "Try a higher cloud threshold or a different season."
                )
            rgb = rgb.where(cloud_mask)

        rgb = rgb.median(dim="time")
        return rgb.compute() if self.config.compute else rgb

    def build_classified_raster(self, year: int, scene_id: str | None = None) -> ClassifiedRasterResult:
        """Get a classified raster and NDVI index for a given year or selected scene."""
        ndvi = self.build_ndvi_raster(year, scene_id=scene_id)

        classified_raster = classify_by_thresholds(
            ndvi, 
            thresholds=self.config.thresholds, 
            class_values=self.config.class_values
        )

        return ClassifiedRasterResult(raster=classified_raster, ndvi=ndvi)


    def build_change_raster(
        self,
        base_year: int,
        target_year: int,
        base_scene_id: str | None = None,
        target_scene_id: str | None = None,
    ) -> ChangeRasterResult:
        """Get a change raster comparing NDVI between two years, and each year's classified raster and index."""
        base_result = self.build_classified_raster(base_year, scene_id=base_scene_id)
        target_result = self.build_classified_raster(target_year, scene_id=target_scene_id)

        ndvi_diff = target_result.ndvi - base_result.ndvi
       
        return ChangeRasterResult(ndvi_diff=ndvi_diff, base_classified=base_result, target_classified=target_result)