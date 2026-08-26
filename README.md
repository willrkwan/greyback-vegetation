# Greyback Lake vegetation analysis

This project examines seasonal vegetation indices around Greyback Lake, British Columbia. It uses Landsat Collection 2 Level-2 imagery from the Microsoft Planetary Computer STAC API and presents the results in a Streamlit app.

## Current workflow

1. Read the committed AOI from `data/aoi/greyback_assessment_aoi.geojson` and use its bounds to search Landsat scenes for a selected year and seasonal date range. The app supports years 2013 through 2026.
2. Filter scenes with the STAC `eo:cloud_cover` field and keep scenes that contain the bands required by the selected profile.
3. Stack imagery on a 30 m grid in UTM Zone 11N (`EPSG:32611`). The stack uses the AOI bounds for scene discovery, then masks pixels outside the polygon.
4. Apply the Landsat `QA_PIXEL` mask where required, calculate continuous indices or band composites, and reduce the eligible scenes across the seasonal window.
5. Use named band profiles for NDVI, NBR, and RGB. NDVI uses `nir08` and `red`; NBR uses `nir08` and `swir22`; RGB uses `red`, `green`, and `blue`.
6. Compare two seasonal NDVI composites by subtracting baseline values from comparison-year values. Missing pixels remain `NaN`.

## Assessment watershed AOI

The AOI was identified from the BC Data Catalogue layer `WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY` through this WFS endpoint:

`https://openmaps.gov.bc.ca/geo/pub/WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY/ows`

The selected polygon is committed at `data/aoi/greyback_assessment_aoi.geojson`. The Streamlit app reads that local file and does not contact the WFS service.

To use another AOI, export a polygon to the same path or change `AOI_PATH` in `src/aoi.py`. The file must contain at least one feature with a valid CRS.

STAC searches use the polygon's bounding box as their footprint. After stacking, `rasterio.features.geometry_mask` removes pixels outside the polygon. The app transforms the AOI from WGS84 (`EPSG:4326`) to UTM Zone 11N (`EPSG:32611`) before raster masking.

## Streamlit application

The app provides:

- Single-year continuous NDVI and RGB composite views.
- Two-year NDVI comparisons with an NDVI change surface.
- Two-year RGB comparisons for visual inspection.
- Controls for analysis mode, baseline and comparison years, seasonal start month and day, season length, and scene cloud filtering.
- A table of eligible scenes with acquisition date, platform, cloud cover, and available bands.
- NDVI-change percentiles for comparison outputs.

NBR is available as a named profile in the raster service for notebook or future UI exploration. The app does not classify pixels into land-cover categories using arbitrary NDVI thresholds.

The change map uses a red-to-green scale centered at zero. Negative values indicate lower NDVI in the comparison composite, while positive values indicate higher NDVI. RGB is provided for visual context and does not use the NDVI cloud mask by default.

## Interpretation and limitations

NDVI and NBR are continuous normalized-difference indices. They should be interpreted as spectral measurements, not definitive vegetation or land-cover labels.

The year-to-year difference is exploratory. Phenology, acquisition timing, scene availability, residual contamination, sensor differences, and changes in valid-pixel coverage can all affect it. Using the same seasonal window improves comparability but does not remove these sources of uncertainty.

The current AOI is an FWA assessment watershed, not a custom lake catchment. It is substantially larger than the lake, so conclusions describe the assessment watershed unless a narrower catchment is substituted.

The repository does not include the validation workflow from the earlier project draft. There is no cutblock/PostGIS validation routine, confusion matrix, accuracy estimate, or Random Forest classifier.

## Future work

- Report valid-pixel coverage, scene counts, and acquisition-date distributions for each composite.
- Add more named band profiles or indices for spectral exploration.
- Add an independent reference dataset and report a confusion matrix with class-specific accuracy measures.
- Measure sensitivity to the seasonal window, cloud threshold, reducer, and index definitions.

