# Greyback Lake vegetation analysis

This project compares seasonal vegetation-index patterns around Greyback Lake, British Columbia. It uses Landsat Collection 2 Level-2 imagery from the Microsoft Planetary Computer STAC API and displays the results in a Streamlit application.

## Implemented workflow

1. Fetch the FWA assessment watershed containing the configured Greyback Lake coordinate (`49.630227703275324, -119.42400064714786`) from the British Columbia WFS service.
2. Use the assessment watershed geometry as the area of interest (AOI). The current match is the Penticton Creek watershed (`WATERSHED_FEATURE_ID=12515`) and covers approximately 8,289 hectares.
3. Query Landsat scenes for the AOI and a selected year and seasonal date range. The available year range is 1984 through 2025.
4. Use the STAC `eo:cloud_cover` field to filter scenes, then retain scenes with the required Landsat assets.
5. Stack the red, near-infrared, QA, and RGB assets on a 30 m grid in UTM Zone 11N (`EPSG:32611`), using the AOI bounds for scene discovery and masking pixels outside the watershed.
6. Apply a Landsat `QA_PIXEL` bit mask to the red and near-infrared data, calculate NDVI, and take the temporal median across all eligible scenes in the seasonal window.
7. Group the composite NDVI values with fixed thresholds of `-0.05`, `0.10`, `0.20`, `0.35`, and `0.50`.
8. Compare two seasonal composites by subtracting the baseline NDVI from the comparison-year NDVI. Missing values remain `NaN` instead of being assigned to a vegetation class.

## Assessment watershed AOI

The AOI is retrieved from the BC Data Catalogue layer `WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY` through its WFS endpoint:

`https://openmaps.gov.bc.ca/geo/pub/WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY/ows`

The lookup uses the Greyback Lake point to select one polygon. The resulting geometry is cached by Streamlit for 24 hours. For local inspection, the notebook exports the selected feature to `data/processed/greyback_assessment_aoi.geojson`.

The raster stack is requested over the polygon's bounding box because STAC requires a search footprint. After stacking, `rasterio.features.geometry_mask` removes pixels outside the actual assessment watershed. The polygon is transformed from WGS84 (`EPSG:4326`) to UTM Zone 11N (`EPSG:32611`) before raster masking.

## Streamlit application

In the Streamlit app, you can:

- Single-year classified NDVI, continuous NDVI, or RGB composite views.
- Two-year classified comparison with a continuous NDVI change surface.
- Two-year continuous NDVI comparison with an NDVI change surface.
- Two-year RGB composite comparison for visual inspection.
- Controls for analysis mode, baseline and comparison years, seasonal start month and day, season length, and scene-level cloud filtering.
- A table of the eligible scenes used in each composite, including acquisition date, platform, cloud-cover metadata, and required-band availability.
- Class-pixel summaries for classified outputs and NDVI-change percentiles for comparison outputs.

The change map uses a red-to-green scale centered at zero. Negative values indicate lower NDVI in the comparison composite, while positive values indicate higher NDVI. RGB is provided for visual context and does not use the NDVI cloud mask by default.

## Interpretation and limitations

The six output categories come from NDVI thresholds. They are not validated land-cover classes and should not be read as definitive water, built-up, shrubland, or forest labels. NDVI alone cannot reliably separate all of those surfaces.

The year-to-year difference is an exploratory comparison. Phenology, acquisition timing, scene availability, residual contamination, sensor differences, and changes in valid-pixel coverage can all affect it. Using the same seasonal window improves comparability, but does not remove these sources of uncertainty.

The repository does not include the validation workflow from the earlier project draft. There is no cutblock/PostGIS validation routine, confusion matrix, accuracy estimate, or Random Forest classifier. These remain future extensions.

The current AOI is an FWA assessment watershed, a custom lake catchment. It is substantially larger than the lake itself, so conclusions describe the assessment watershed unless a more narrowly defined catchment is substituted.

## Future work

- Report valid-pixel coverage, scene counts, and acquisition-date distributions for each composite.
- Validate the NDVI categories or replace them with a supervised, multi-band land-cover classification.
- Add an independent reference dataset and report a confusion matrix with class-specific accuracy measures.
- Measure sensitivity to the seasonal window, cloud threshold, reducer, and NDVI thresholds.
- Examine reservoir or water-level relationships after validating the spatial and temporal analysis.
