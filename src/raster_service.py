from .indices import compute_normalized_difference
from .preprocessing import temporal_composite
from .ingest import (
    get_bounding_box_geojson, 
    get_season_date_ranges, 
    search_stac_items, 
    stack_items, 
    build_landsat_cloud_mask,
)
from .models import IndexChangeResult, RasterConfig, SceneSummary


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
        """Return a GeoJSON footprint for the configured AOI, either from the AOI geometry
         or a bounding box around the center point."""
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
                    available_bands=frozenset(assets),
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

    def _load_stack(
        self,
        year: int,
        bands: tuple[str, ...],
        scene_id: str | None = None,
        apply_cloud_mask: bool = True,
    ):
        footprint, items = self._search_items_for_year(year)
        items = self._select_scene(items, scene_id)

        requested_bands = tuple(dict.fromkeys(bands))
        if not requested_bands:
            raise ValueError("At least one imagery band is required.")

        stack_bands = list(requested_bands)
        if apply_cloud_mask and self.config.qa_band not in stack_bands:
            stack_bands.append(self.config.qa_band)

        items = self._filter_items_by_assets(items, set(stack_bands))
        if not items:
            raise NoScenesFoundError(
                f"No scenes found for year {year} with bands {requested_bands!r} "
                f"in the selected seasonal window for scene_id={scene_id!r}."
            )

        data = stack_items(
            items,
            footprint,
            bands=stack_bands,
            resolution=self.config.resolution,
            epsg=self.config.epsg,
            aoi_geometry=self.config.aoi_geometry,
        )

        mask = None
        if apply_cloud_mask:
            mask = build_landsat_cloud_mask(
                data,
                qa_band=self.config.qa_band,
                cloud_bits=self.config.cloud_bits,
            )
            if mask is None or not bool(mask.any()):
                raise NoClearPixelsError(
                    f"No usable clear-sky pixels remained for year {year} after masking. "
                    "Try a higher cloud threshold or a different season."
                )

        return data, requested_bands, mask

    def build_composite(
        self,
        year: int,
        bands: tuple[str, ...],
        scene_id: str | None = None,
        apply_cloud_mask: bool = True,
    ):
        """Build a temporal composite from any requested imagery bands."""
        data, requested_bands, mask = self._load_stack(
            year,
            bands,
            scene_id=scene_id,
            apply_cloud_mask=apply_cloud_mask,
        )

        return temporal_composite(
            data,
            bands=requested_bands,
            mask=mask,
            reducer=self.config.reducer,
            compute=self.config.compute,
        )

    def build_normalized_difference(
        self,
        year: int,
        numerator_band: str,
        denominator_band: str,
        scene_id: str | None = None,
        apply_cloud_mask: bool = True,
    ):
        """Build a normalized-difference index from any two imagery bands."""
        data, _, mask = self._load_stack(
            year,
            bands=(numerator_band, denominator_band),
            scene_id=scene_id,
            apply_cloud_mask=apply_cloud_mask,
        )
        index = compute_normalized_difference(
            data,
            num_band=numerator_band,
            den_band=denominator_band,
            mask=mask,
            reducer=self.config.reducer,
            compute=self.config.compute,
        )
        if index is None or index.size == 0:
            raise NoClearPixelsError(
                f"The normalized-difference result for year {year} contained no valid pixels."
            )
        return index

    def build_ndvi_raster(self, year: int, scene_id: str | None = None):
        """Build an NDVI raster using the configured numerator and denominator bands."""
        return self.build_normalized_difference(
            year,
            numerator_band=self.config.ndvi_num_band,
            denominator_band=self.config.ndvi_den_band,
            scene_id=scene_id,
        )

    def build_nbr_raster(self, year: int, scene_id: str | None = None):
        """Build an NBR raster using the configured numerator and denominator bands."""
        return self.build_normalized_difference(
            year,
            numerator_band=self.config.nbr_num_band,
            denominator_band=self.config.nbr_den_band,
            scene_id=scene_id,
        )

    def build_rgb_raster(self, year: int, scene_id: str | None = None, apply_cloud_mask: bool = False):
        """Build an RGB raster for a year or a selected scene within that year.

        RGB visualization keeps cloudy pixels by default so the scene footprint remains visible. """
        return self.build_composite(
            year,
            bands=(
                self.config.red_band,
                self.config.green_band,
                self.config.blue_band,
            ),
            scene_id=scene_id,
            apply_cloud_mask=apply_cloud_mask,
        )

    def build_index_change(
        self,
        base_year: int,
        target_year: int,
        numerator_band: str,
        denominator_band: str,
        base_scene_id: str | None = None,
        target_scene_id: str | None = None,
    ) -> IndexChangeResult:
        """Compare a normalized-difference index between two years."""
        baseline = self.build_normalized_difference(
            base_year,
            numerator_band=numerator_band,
            denominator_band=denominator_band,
            scene_id=base_scene_id,
        )
        comparison = self.build_normalized_difference(
            target_year,
            numerator_band=numerator_band,
            denominator_band=denominator_band,
            scene_id=target_scene_id,
        )
        return IndexChangeResult(
            ndvi_diff=comparison - baseline,
            baseline=baseline,
            comparison=comparison,
        )