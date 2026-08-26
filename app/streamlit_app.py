import os
import sys
import calendar
from pathlib import Path
from datetime import date, timedelta

from dataclasses import dataclass
from typing import Optional
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import requests
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from planetary_computer import sign_inplace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import BandProfile, RasterConfig
from src.raster_service import NoClearPixelsError, NoScenesFoundError, RasterService


@st.cache_resource
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=sign_inplace,
    )


YEAR_MIN = 1984
YEAR_MAX = 2025
ASSESSMENT_WATERSHED_WFS_URL = (
    "https://openmaps.gov.bc.ca/geo/pub/"
    "WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY/ows"
)


def get_band_profiles(config: RasterConfig) -> dict[str, BandProfile]:
    return {
        "NDVI": BandProfile(
            name="NDVI",
            bands=(config.ndvi_num_band, config.ndvi_den_band),
            kind="normalized_difference",
            numerator_band=config.ndvi_num_band,
            denominator_band=config.ndvi_den_band,
        ),
        "NBR": BandProfile(
            name="NBR",
            bands=(config.nbr_num_band, config.nbr_den_band),
            kind="normalized_difference",
            numerator_band=config.nbr_num_band,
            denominator_band=config.nbr_den_band,
        ),
        "RGB": BandProfile(
            name="RGB",
            bands=(config.red_band, config.green_band, config.blue_band),
            kind="composite",
            apply_cloud_mask=False,
        ),
    }


@st.cache_data(ttl=86400)
def load_assessment_aoi() -> dict:
    lon = -119.42400064714786
    lat = 49.630227703275324
    response = requests.get(
        ASSESSMENT_WATERSHED_WFS_URL,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY",
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "bbox": f"{lon - 0.1},{lat - 0.1},{lon + 0.1},{lat + 0.1},EPSG:4326",
        },
        timeout=60,
    )
    response.raise_for_status()
    candidates = gpd.GeoDataFrame.from_features(response.json(), crs="EPSG:4326")
    point = gpd.GeoSeries.from_xy([lon], [lat], crs="EPSG:4326").iloc[0]
    matches = candidates[candidates.geometry.intersects(point)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one assessment watershed at the AOI point, found {len(matches)}."
        )
    return matches.geometry.iloc[0].__geo_interface__


@dataclass(frozen=True)
class ViewSelection:
    mode: str
    selected_year: int
    baseline_year: int
    comparison_year: Optional[int]


@dataclass(frozen=True)
class AnalysisSettings:
    season_start_month: int
    season_start_day: int
    duration_days: int
    max_cloud: float


@st.cache_resource
def load_service(
    season_start_month: int,
    season_start_day: int,
    duration_days: int,
    max_cloud: float,
) -> RasterService:
    catalog = get_catalog()
    aoi_geometry = load_assessment_aoi()
    config = RasterConfig(
        catalog=catalog,
        collection_id="landsat-c2-l2",
        center_lat=49.630227703275324,
        center_lon=-119.42400064714786,
        half_side_km=5.0,
        aoi_geometry=aoi_geometry,
        season_start_month=season_start_month,
        season_start_day=season_start_day,
        duration_days=duration_days,
        max_cloud=max_cloud,
        resolution=30,
        epsg=32611,
        ndvi_num_band="nir08",
        ndvi_den_band="red",
        qa_band="qa_pixel",
        cloud_bits=(1, 3, 4),
        reducer="median",
        thresholds=(-0.05, 0.10, 0.20, 0.35, 0.50),
        class_values=(0, 1, 2, 3, 4, 5),
    )
    return RasterService(config)


def render_array(array, title, mode="continuous", figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    ndvi_cmap = LinearSegmentedColormap.from_list(
        "ndvi_change",
        [(0.0, "red"), (0.5, "grey"), (1.0, "green")],
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    img = ax.imshow(array.values, cmap=ndvi_cmap, norm=norm)
    colorbar_label = "NDVI" if mode == "ndvi" else "NDVI change"
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    ax.set_title(title)
    ax.axis("off")
    return fig


def render_rgb_array(array, title, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    values = np.asarray(array.values, dtype=float)
    if values.ndim != 3:
        raise ValueError("RGB rendering expects a 3D array with 3 bands.")

    if values.shape[0] == 3:
        rgb = np.moveaxis(values, 0, -1)
    elif values.shape[-1] == 3:
        rgb = values
    else:
        raise ValueError("RGB rendering requires exactly 3 channels.")

    low = np.nanpercentile(rgb, 2)
    high = np.nanpercentile(rgb, 98)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        stretched = np.zeros_like(rgb)
    else:
        stretched = np.clip((rgb - low) / (high - low), 0.0, 1.0)

    ax.imshow(stretched)
    ax.set_title(title)
    ax.axis("off")
    return fig


def summarize_change_percentiles(change_array):
    values = np.asarray(change_array.values if hasattr(change_array, "values") else change_array, dtype=float)
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}

    return {
        "p10": float(np.percentile(valid, 10)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "p90": float(np.percentile(valid, 90)),
    }


def render_change_percentiles(percentiles):
    st.caption("NDVI change percentiles")
    cols = st.columns(5)
    labels = ["P10", "P25", "P50", "P75", "P90"]
    keys = ["p10", "p25", "p50", "p75", "p90"]
    for col, label, key in zip(cols, labels, keys):
        with col:
            st.metric(label, f"{percentiles[key]:.3f}")


def format_season_window(year, settings):
    start_dt = date(year, settings.season_start_month, settings.season_start_day)
    end_dt = start_dt + timedelta(days=settings.duration_days - 1)
    return f"{start_dt.isoformat()} to {end_dt.isoformat()}"


def reset_analysis_submission():
    st.session_state["analysis_submitted"] = False


@st.cache_data(ttl=600)
def get_scene_rows(year, season_start_month, season_start_day, duration_days, max_cloud):
    service = load_service(
        season_start_month=season_start_month,
        season_start_day=season_start_day,
        duration_days=duration_days,
        max_cloud=max_cloud,
    )
    summaries = service.list_scene_summaries(year)
    rows = []
    for summary in summaries:
        rows.append(
            {
                "scene_id": summary.scene_id,
                "acquired_at": summary.acquired_at,
                "cloud_cover": summary.cloud_cover,
                "platform": summary.platform,
                "available_bands": summary.available_bands,
            }
        )
    return rows


def get_profile_for_output(output_mode, profiles):
    profile_name = (
        "NDVI"
        if output_mode in {"NDVI", "NDVI + change"}
        else "RGB"
    )
    return profiles[profile_name]


def get_required_bands(profile, qa_band):
    required_bands = set(profile.bands)
    if profile.apply_cloud_mask:
        required_bands.add(qa_band)
    return required_bands


def build_profile_raster(service, year, profile):
    if profile.kind == "normalized_difference":
        return service.build_normalized_difference(
            year,
            numerator_band=profile.numerator_band,
            denominator_band=profile.denominator_band,
            apply_cloud_mask=profile.apply_cloud_mask,
        )
    return service.build_composite(
        year,
        bands=profile.bands,
        apply_cloud_mask=profile.apply_cloud_mask,
    )


def filter_scene_rows_for_output(scene_rows, output_mode, profiles, qa_band):
    required_bands = get_required_bands(
        get_profile_for_output(output_mode, profiles),
        qa_band,
    )
    return [
        row for row in scene_rows
        if required_bands.issubset(row["available_bands"])
    ]


def render_scene_availability_table(title, scene_rows, profiles, qa_band):
    st.caption(title)
    table_rows = []
    for row in scene_rows:
        table_rows.append(
            {
                "Scene ID": row["scene_id"],
                "Acquired": row["acquired_at"],
                "Cloud %": None if row["cloud_cover"] is None else round(row["cloud_cover"], 1),
                "Platform": row["platform"],
                "Available bands": ", ".join(sorted(row["available_bands"])),
                "NDVI ready": get_required_bands(profiles["NDVI"], qa_band).issubset(row["available_bands"]),
                "RGB ready": get_required_bands(profiles["RGB"], qa_band).issubset(row["available_bands"]),
            }
        )
    st.dataframe(table_rows, width="stretch", hide_index=True)


def get_user_inputs():
    with st.sidebar:
        st.subheader(":material/compare_arrows: Mode")
        st.caption("Choose whether to inspect one year or compare two years side by side.")
        mode = st.selectbox(
            "View mode",
            options=["Single year", "Compare years"],
            index=1,
            key="view_mode",
            on_change=reset_analysis_submission,
        )

        with st.form("analysis_controls"):
            year_options = list(range(YEAR_MIN, YEAR_MAX + 1))
            if mode == "Single year":
                selected_year = st.selectbox(
                    "Selected year",
                    options=year_options,
                    index=0,
                    help="Year used for a single-season vegetation classification.",
                )
                selection = ViewSelection(
                    mode=mode,
                    selected_year=selected_year,
                    baseline_year=selected_year,
                    comparison_year=None,
                )
            else:
                baseline_year = st.selectbox(
                    "Baseline year",
                    options=year_options,
                    index=0,
                    help="Earlier year used as the reference for change analysis.",
                )
                comparison_year = st.selectbox(
                    "Comparison year",
                    options=year_options,
                    index=year_options.index(2024),
                    help="Later year compared with the baseline.",
                )
                selection = ViewSelection(
                    mode=mode,
                    selected_year=baseline_year,
                    baseline_year=baseline_year,
                    comparison_year=comparison_year,
                )

            st.markdown("---")
            st.subheader(":material/tune: Analysis settings")
            st.caption("Adjust the seasonal window used to fetch and aggregate Landsat imagery for each year.")
            season_start_month = st.selectbox(
                "Season start month",
                options=list(range(1, 13)),
                index=6,
                format_func=lambda month: date(2000, month, 1).strftime("%B"),
                help="Month used to begin the seasonal window.",
            )
            max_day = 28 if season_start_month == 2 else calendar.monthrange(2024, season_start_month)[1]
            season_start_day = st.selectbox(
                "Season start day",
                options=list(range(1, max_day + 1)),
                index=0,
                help="Day within the starting month. February is limited to 28 so every year remains valid.",
            )
            settings = AnalysisSettings(
                season_start_month=season_start_month,
                season_start_day=season_start_day,
                duration_days=st.slider(
                    "Season length (days)",
                    min_value=7,
                    max_value=180,
                    value=31,
                    help="Number of days included in the window for each year.",
                ),
                max_cloud=st.slider(
                    "Cloud filter (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="The maximum acceptable cloud cover before a scene is excluded.",
                ),
            )
            submitted = st.form_submit_button(
                "Apply analysis",
                type="primary",
                icon=":material/play_arrow:",
            )

    if submitted:
        st.session_state["analysis_submitted"] = True

    return selection, settings, st.session_state.get("analysis_submitted", False)


def get_output_mode_config(view_selection):
    if view_selection.mode == "Single year":
        return ["NDVI", "RGB"], "NDVI"
    return ["NDVI + change", "RGB"], "NDVI + change"


def get_output_mode_from_state(view_selection):
    options, default = get_output_mode_config(view_selection)
    current = st.session_state.get("composite_output_mode", default)
    if current not in options:
        current = default
        st.session_state["composite_output_mode"] = default
    return current, options, default


def main():
    st.set_page_config(
        page_title="Greyback Lake Land Use",
        page_icon=":material/terrain:",
        layout="wide",
    )
    st.title("Greyback Lake vegetation analysis")
    st.caption("Explore seasonal vegetation patterns and compare NDVI-driven land cover change across years.")

    view_selection, analysis_settings, analysis_submitted = get_user_inputs()

    if not analysis_submitted:
        st.info("Choose the analysis settings, then select Apply analysis to run the workflow.")
        return

    if view_selection.mode == "Compare years" and view_selection.comparison_year == view_selection.baseline_year:
        st.warning("Baseline and comparison years must be different for change analysis.")
        return

    service = load_service(
        season_start_month=analysis_settings.season_start_month,
        season_start_day=analysis_settings.season_start_day,
        duration_days=analysis_settings.duration_days,
        max_cloud=analysis_settings.max_cloud,
    )
    band_profiles = get_band_profiles(service.config)

    baseline_scene_rows = get_scene_rows(
        year=view_selection.baseline_year,
        season_start_month=analysis_settings.season_start_month,
        season_start_day=analysis_settings.season_start_day,
        duration_days=analysis_settings.duration_days,
        max_cloud=analysis_settings.max_cloud,
    )

    if not baseline_scene_rows:
        st.warning(
            f"No scenes are available for {view_selection.baseline_year} with the current seasonal controls and cloud filter."
        )
        st.info("Try a wider season window or increase the cloud filter.")
        return

    comparison_scene_rows = []
    if view_selection.mode == "Compare years":
        comparison_scene_rows = get_scene_rows(
            year=view_selection.comparison_year,
            season_start_month=analysis_settings.season_start_month,
            season_start_day=analysis_settings.season_start_day,
            duration_days=analysis_settings.duration_days,
            max_cloud=analysis_settings.max_cloud,
        )
        if not comparison_scene_rows:
            st.warning(
                f"No scenes are available for {view_selection.comparison_year} with the current seasonal controls and cloud filter."
            )
            st.info("Try a wider season window or increase the cloud filter.")
            return

    output_mode, output_options, output_default = get_output_mode_from_state(view_selection)

    baseline_used_rows = filter_scene_rows_for_output(
        baseline_scene_rows,
        output_mode,
        band_profiles,
        service.config.qa_band,
    )
    if not baseline_used_rows:
        st.warning("No baseline scenes are eligible for the selected composite output.")
        st.info("Try another output mode or loosen the cloud and season filters.")
        return

    comparison_used_rows = []
    if view_selection.mode == "Compare years":
        comparison_used_rows = filter_scene_rows_for_output(
            comparison_scene_rows,
            output_mode,
            band_profiles,
            service.config.qa_band,
        )
        if not comparison_used_rows:
            st.warning("No comparison scenes are eligible for the selected composite output.")
            st.info("Try another output mode or loosen the cloud and season filters.")
            return

    st.subheader("Eligible scenes used in composite")
    render_scene_availability_table(
        f"{view_selection.baseline_year} scenes used",
        baseline_used_rows,
        band_profiles,
        service.config.qa_band,
    )
    if view_selection.mode == "Compare years":
        render_scene_availability_table(
            f"{view_selection.comparison_year} scenes used",
            comparison_used_rows,
            band_profiles,
            service.config.qa_band,
        )

    st.subheader("Output")
    selected_mode = st.segmented_control(
        "Composite output",
        options=output_options,
        default=output_default,
        key="composite_output_mode",
    )
    if selected_mode != output_mode:
        st.rerun()
    output_mode = selected_mode
    output_profile = get_profile_for_output(output_mode, band_profiles)

    try:
        with st.spinner("Loading composite output from eligible scenes..."):
            if view_selection.mode == "Single year" and output_mode == "NDVI":
                baseline_ndvi = build_profile_raster(
                    service, view_selection.selected_year, output_profile
                )
            elif view_selection.mode == "Single year" and output_mode == "RGB":
                baseline_rgb = build_profile_raster(
                    service, view_selection.selected_year, output_profile
                )
            elif view_selection.mode == "Compare years" and output_mode == "NDVI + change":
                baseline_ndvi = build_profile_raster(
                    service, view_selection.baseline_year, output_profile
                )
                comparison_ndvi = build_profile_raster(
                    service, view_selection.comparison_year, output_profile
                )
                ndvi_diff = comparison_ndvi - baseline_ndvi
            else:
                baseline_rgb = build_profile_raster(
                    service, view_selection.baseline_year, output_profile
                )
                comparison_rgb = build_profile_raster(
                    service, view_selection.comparison_year, output_profile
                )
    except NoScenesFoundError as exc:
        st.warning(str(exc))
        st.info("Try another year, widen the seasonal window, or loosen the cloud filter.")
        return
    except NoClearPixelsError as exc:
        st.warning(str(exc))
        st.info("Consider increasing the cloud threshold or selecting a different year.")
        return
    except Exception as exc:  # pragma: no cover - fallback for unexpected runtime issues
        st.error(f"Unexpected processing error: {exc}")
        return

    if view_selection.mode == "Single year" and output_mode == "NDVI":
        season_text = format_season_window(view_selection.selected_year, analysis_settings)
        st.subheader(f"{view_selection.selected_year} NDVI composite")
        st.caption(
            f"Seasonal window: {season_text}"
        )
        st.pyplot(
            render_array(
                baseline_ndvi,
                f"{view_selection.selected_year} NDVI seasonal composite",
                mode="ndvi",
            )
        )
        return

    if view_selection.mode == "Single year" and output_mode == "RGB":
        season_text = format_season_window(view_selection.selected_year, analysis_settings)
        st.subheader(f"{view_selection.selected_year} RGB composite")
        st.caption(
            f"Seasonal window: {season_text}"
        )
        st.pyplot(
            render_rgb_array(
                baseline_rgb,
                f"{view_selection.selected_year} RGB seasonal composite",
            )
        )
        return

    if view_selection.mode == "Compare years" and output_mode == "NDVI + change":
        baseline_season = format_season_window(view_selection.baseline_year, analysis_settings)
        comparison_season = format_season_window(view_selection.comparison_year, analysis_settings)
        st.subheader("NDVI seasonal composite comparison")
        st.caption(
            f"Baseline season: {baseline_season} | Comparison season: {comparison_season}"
        )
        render_change_percentiles(summarize_change_percentiles(ndvi_diff))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"Baseline NDVI: {view_selection.baseline_year}")
            st.pyplot(
                render_array(
                    baseline_ndvi,
                    f"{view_selection.baseline_year} NDVI seasonal composite",
                    mode="ndvi",
                    figsize=(5.5, 5.5),
                )
            )
        with col2:
            st.subheader("NDVI change")
            st.pyplot(
                render_array(
                    ndvi_diff,
                    f"{view_selection.baseline_year} to {view_selection.comparison_year} NDVI change",
                    mode="continuous",
                    figsize=(5.5, 5.5),
                )
            )
        with col3:
            st.subheader(f"Comparison NDVI: {view_selection.comparison_year}")
            st.pyplot(
                render_array(
                    comparison_ndvi,
                    f"{view_selection.comparison_year} NDVI seasonal composite",
                    mode="ndvi",
                    figsize=(5.5, 5.5),
                )
            )
        return

    if view_selection.mode == "Compare years" and output_mode == "RGB":
        baseline_season = format_season_window(view_selection.baseline_year, analysis_settings)
        comparison_season = format_season_window(view_selection.comparison_year, analysis_settings)
        st.subheader("RGB seasonal composite comparison")
        st.caption(
            f"Baseline season: {baseline_season} | Comparison season: {comparison_season}"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Baseline RGB: {view_selection.baseline_year}")
            st.pyplot(
                render_rgb_array(
                    baseline_rgb,
                    f"{view_selection.baseline_year} RGB seasonal composite",
                    figsize=(6, 5.5),
                )
            )
        with col2:
            st.subheader(f"Comparison RGB: {view_selection.comparison_year}")
            st.pyplot(
                render_rgb_array(
                    comparison_rgb,
                    f"{view_selection.comparison_year} RGB seasonal composite",
                    figsize=(6, 5.5),
                )
            )
        return

    st.error("Unsupported output mode.")


if __name__ == "__main__":
    main()
