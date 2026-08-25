# Greyback Lake vegetation analysis

This project compares seasonal vegetation-index patterns around Greyback Lake, British Columbia. It uses Landsat Collection 2 Level-2 imagery from the Microsoft Planetary Computer STAC API and displays the results in a Streamlit application.

## Implemented workflow

1. Define a square study window centered near Greyback Lake. With `half_side_km=5`, the current window is approximately 10 km by 10 km.
2. Query Landsat scenes for a selected year and seasonal date range. The available year range is 1984 through 2025.
3. Use the STAC `eo:cloud_cover` field to filter scenes, then retain scenes with the required Landsat assets.
4. Stack the red, near-infrared, QA, and RGB assets on a 30 m grid in UTM Zone 11N (`EPSG:32611`).
5. Apply a Landsat `QA_PIXEL` bit mask to the red and near-infrared data, calculate NDVI, and take the temporal median across all eligible scenes in the seasonal window.
6. Group the composite NDVI values with fixed thresholds of `-0.05`, `0.10`, `0.20`, `0.35`, and `0.50`.
7. Compare two seasonal composites by subtracting the baseline NDVI from the comparison-year NDVI. Missing values remain `NaN` instead of being assigned to a vegetation class.

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

The study window is a geometric square, not a watershed boundary. Conclusions should stay within the configured study area unless a hydrologically derived catchment boundary is added.

## Future work

- Replace the square window with a documented watershed or other defensible area of interest.
- Report valid-pixel coverage, scene counts, and acquisition-date distributions for each composite.
- Validate the NDVI categories or replace them with a supervised, multi-band land-cover classification.
- Add an independent reference dataset and report a confusion matrix with class-specific accuracy measures.
- Measure sensitivity to the seasonal window, cloud threshold, reducer, and NDVI thresholds.
- Examine reservoir or water-level relationships after validating the spatial and temporal analysis.
