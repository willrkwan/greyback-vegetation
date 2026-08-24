import os
import sys
from pathlib import Path
from datetime import date, timedelta

from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import streamlit as st
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from planetary_computer import sign_inplace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import RasterConfig
from src.raster_service import NoClearPixelsError, NoScenesFoundError, RasterService


@st.cache_resource
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=sign_inplace,
    )

YEAR_MIN = 2013
YEAR_MAX = 2025

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


@dataclass(frozen=True)
class SceneSelection:
    baseline_scene_id: str
    comparison_scene_id: Optional[str]
    output_mode: str


@st.cache_resource
def load_service(
    season_start_month: int,
    season_start_day: int,
    duration_days: int,
    max_cloud: float,
) -> RasterService:
    catalog = get_catalog()
    config = RasterConfig(
        catalog=catalog,
        collection_id="landsat-c2-l2",
        center_lat=49.630227703275324,
        center_lon=-119.42400064714786,
        half_side_km=5.0,
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


def render_array(array, title, mode="continuous"):
    fig, ax = plt.subplots(figsize=(8, 6))

    if mode == "classified":
        colors = [
            "#d73027",
            "#fc8d59",
            "#fee08b",
            "#d9ef8b",
            "#91cf60",
            "#1a9850",
        ]
        class_names = [
            "Water / very low vegetation",
            "Bare / sparse vegetation",
            "Low vegetation",
            "Moderate vegetation",
            "High vegetation",
            "Very high vegetation",
        ]

        discrete_cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), discrete_cmap.N)
        img = ax.imshow(array.values, cmap=discrete_cmap, norm=norm)

        legend_elements = [
            Patch(facecolor=color, label=f"{idx}: {name}")
            for idx, (color, name) in enumerate(zip(colors, class_names))
        ]
        ax.legend(
            handles=legend_elements,
            title="Vegetation class",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )

    else:
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


def render_rgb_array(array, title):
    fig, ax = plt.subplots(figsize=(8, 6))

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


CLASS_NAMES = [
    "Water / very low vegetation",
    "Bare / sparse vegetation",
    "Low vegetation",
    "Moderate vegetation",
    "High vegetation",
    "Very high vegetation",
]


def summarize_class_distribution(classified_array):
    values = np.asarray(classified_array.values if hasattr(classified_array, "values") else classified_array)
    flat = values.ravel()
    valid = flat[np.isfinite(flat)]

    if valid.size == 0:
        return {idx: 0.0 for idx in range(len(CLASS_NAMES))}

    counts = np.bincount(valid.astype(int), minlength=len(CLASS_NAMES))
    total = counts.sum()
    if total == 0:
        return {idx: 0.0 for idx in range(len(CLASS_NAMES))}

    return {idx: float(count / total * 100.0) for idx, count in enumerate(counts)}


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


def build_class_distribution_rows(summary):
    return [
        {
            "Class": idx,
            "Vegetation class": CLASS_NAMES[idx],
            "Share of pixels": round(summary[idx], 2),
        }
        for idx in range(len(CLASS_NAMES))
    ]


def build_comparison_rows(baseline_summary, comparison_summary):
    return [
        {
            "Class": idx,
            "Vegetation class": CLASS_NAMES[idx],
            "Baseline %": round(baseline_summary[idx], 2),
            "Comparison %": round(comparison_summary[idx], 2),
            "Change % points": round(comparison_summary[idx] - baseline_summary[idx], 2),
        }
        for idx in range(len(CLASS_NAMES))
    ]


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


def format_scene_stamp(scene_meta):
    acquired = scene_meta.get("acquired_at")
    if not acquired:
        return "unknown acquisition"
    return acquired.split("T")[0]


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
                "has_ndvi_bands": summary.has_ndvi_bands,
                "has_rgb_bands": summary.has_rgb_bands,
            }
        )
    return rows


def format_scene_options(scene_rows):
    options = []
    labels = {}
    for row in scene_rows:
        cloud_value = row["cloud_cover"]
        cloud_text = "n/a" if cloud_value is None else f"{cloud_value:.1f}%"
        label = f"{row['acquired_at']} | cloud {cloud_text} | {row['scene_id']}"
        options.append(row["scene_id"])
        labels[row["scene_id"]] = label
    return options, labels


def render_scene_availability_table(title, scene_rows):
    st.caption(title)
    table_rows = []
    for row in scene_rows:
        table_rows.append(
            {
                "Scene ID": row["scene_id"],
                "Acquired": row["acquired_at"],
                "Cloud %": None if row["cloud_cover"] is None else round(row["cloud_cover"], 1),
                "Platform": row["platform"],
                "NDVI ready": row["has_ndvi_bands"],
                "RGB ready": row["has_rgb_bands"],
            }
        )
    st.dataframe(table_rows, width="stretch", hide_index=True)


def get_user_inputs():
    with st.sidebar:
        st.subheader(":material/compare_arrows: Mode")
        st.caption("Choose whether to inspect one year or compare two years side by side.")
        mode = st.segmented_control("View mode", options=["single year", "compare years"], default="compare years")

        if mode == "single year":
            selected_year = st.slider(
                "Selected year",
                min_value=YEAR_MIN,
                max_value=YEAR_MAX,
                value=YEAR_MIN,
                step=1,
                help="Year used for a single-season vegetation classification.",
            )
            selection = ViewSelection(
                mode=mode,
                selected_year=selected_year,
                baseline_year=selected_year,
                comparison_year=None,
            )
        else:
            baseline_year = st.slider("Baseline year", min_value=YEAR_MIN, max_value=YEAR_MAX, value=YEAR_MIN, step=1)
            comparison_year = st.slider("Comparison year", min_value=YEAR_MIN, max_value=YEAR_MAX, value=2024, step=1)
            selection = ViewSelection(
                mode=mode,
                selected_year=baseline_year,
                baseline_year=baseline_year,
                comparison_year=comparison_year,
            )

        st.markdown("---")
        st.subheader(":material/tune: Analysis settings")
        st.caption("Adjust the seasonal window used to fetch and aggregate Landsat imagery for each year.")
        settings = AnalysisSettings(
            season_start_month=st.slider(
                "Season start month",
                min_value=1,
                max_value=12,
                value=7,
                help="Month used to begin the seasonal window.",
            ),
            season_start_day=st.slider(
                "Season start day",
                min_value=1,
                max_value=31,
                value=1,
                help="Day within the starting month.",
            ),
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

    return selection, settings


def get_scene_selection(view_selection, baseline_scene_rows, comparison_scene_rows):
    st.subheader("Scene selection")
    st.caption("Select specific scenes after reviewing availability metadata.")

    baseline_options, baseline_labels = format_scene_options(baseline_scene_rows)
    baseline_scene_id = st.selectbox(
        "Baseline scene",
        options=baseline_options,
        format_func=lambda scene_id: baseline_labels[scene_id],
    )

    if view_selection.mode == "single year":
        output_mode = st.segmented_control(
            "Output",
            options=["classified", "ndvi", "rgb"],
            default="classified",
        )
        return SceneSelection(
            baseline_scene_id=baseline_scene_id,
            comparison_scene_id=None,
            output_mode=output_mode,
        )

    comparison_options, comparison_labels = format_scene_options(comparison_scene_rows)
    comparison_scene_id = st.selectbox(
        "Comparison scene",
        options=comparison_options,
        format_func=lambda scene_id: comparison_labels[scene_id],
    )
    output_mode = st.segmented_control(
        "Output",
        options=["classified + change", "ndvi + change", "rgb"],
        default="classified + change",
    )
    return SceneSelection(
        baseline_scene_id=baseline_scene_id,
        comparison_scene_id=comparison_scene_id,
        output_mode=output_mode,
    )


def main():
    st.set_page_config(
        page_title="Greyback Lake Land Use",
        page_icon=":material/terrain:",
        layout="wide",
    )
    st.title("Greyback Lake vegetation analysis")
    st.caption("Explore seasonal vegetation patterns and compare NDVI-driven land cover change across years.")

    view_selection, analysis_settings = get_user_inputs()

    if view_selection.mode == "compare years" and view_selection.comparison_year == view_selection.baseline_year:
        st.warning("Baseline and comparison years must be different for change analysis.")
        return

    service = load_service(
        season_start_month=analysis_settings.season_start_month,
        season_start_day=analysis_settings.season_start_day,
        duration_days=analysis_settings.duration_days,
        max_cloud=analysis_settings.max_cloud,
    )

    baseline_scene_rows = get_scene_rows(
        year=view_selection.baseline_year,
        season_start_month=analysis_settings.season_start_month,
        season_start_day=analysis_settings.season_start_day,
        duration_days=analysis_settings.duration_days,
        max_cloud=analysis_settings.max_cloud,
    )

    if not baseline_scene_rows:
        st.warning(
            f"No scenes available for {view_selection.baseline_year} with current seasonal controls and cloud filter."
        )
        st.info("Try a wider season window or increase the cloud filter.")
        return

    comparison_scene_rows = []
    if view_selection.mode == "compare years":
        comparison_scene_rows = get_scene_rows(
            year=view_selection.comparison_year,
            season_start_month=analysis_settings.season_start_month,
            season_start_day=analysis_settings.season_start_day,
            duration_days=analysis_settings.duration_days,
            max_cloud=analysis_settings.max_cloud,
        )
        if not comparison_scene_rows:
            st.warning(
                f"No scenes available for {view_selection.comparison_year} with current seasonal controls and cloud filter."
            )
            st.info("Try a wider season window or increase the cloud filter.")
            return

    st.subheader("Scene availability")
    render_scene_availability_table(f"{view_selection.baseline_year} candidate scenes", baseline_scene_rows)
    if view_selection.mode == "compare years":
        render_scene_availability_table(f"{view_selection.comparison_year} candidate scenes", comparison_scene_rows)

    scene_selection = get_scene_selection(view_selection, baseline_scene_rows, comparison_scene_rows)

    baseline_scene_meta = {
        row["scene_id"]: row for row in baseline_scene_rows
    }[scene_selection.baseline_scene_id]
    comparison_scene_meta = None
    if view_selection.mode == "compare years":
        comparison_scene_meta = {
            row["scene_id"]: row for row in comparison_scene_rows
        }[scene_selection.comparison_scene_id]

    if scene_selection.output_mode in {"classified", "ndvi", "classified + change", "ndvi + change"}:
        if not baseline_scene_meta["has_ndvi_bands"]:
            st.warning("The selected baseline scene does not include all NDVI bands.")
            return
        if comparison_scene_meta is not None and not comparison_scene_meta["has_ndvi_bands"]:
            st.warning("The selected comparison scene does not include all NDVI bands.")
            return

    if scene_selection.output_mode == "rgb":
        if not baseline_scene_meta["has_rgb_bands"]:
            st.warning("The selected baseline scene does not include all RGB bands.")
            return
        if comparison_scene_meta is not None and not comparison_scene_meta["has_rgb_bands"]:
            st.warning("The selected comparison scene does not include all RGB bands.")
            return

    try:
        with st.spinner("Loading selected scene output..."):
            if view_selection.mode == "single year" and scene_selection.output_mode == "classified":
                baseline = service.build_classified_raster(
                    view_selection.selected_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
            elif view_selection.mode == "single year" and scene_selection.output_mode == "ndvi":
                baseline_ndvi = service.build_ndvi_raster(
                    view_selection.selected_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
            elif view_selection.mode == "single year" and scene_selection.output_mode == "rgb":
                baseline_rgb = service.build_rgb_raster(
                    view_selection.selected_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
            elif view_selection.mode == "compare years" and scene_selection.output_mode == "classified + change":
                baseline = service.build_classified_raster(
                    view_selection.baseline_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
                comparison = service.build_classified_raster(
                    view_selection.comparison_year,
                    scene_id=scene_selection.comparison_scene_id,
                )
                change = service.build_change_raster(
                    view_selection.baseline_year,
                    view_selection.comparison_year,
                    base_scene_id=scene_selection.baseline_scene_id,
                    target_scene_id=scene_selection.comparison_scene_id,
                )
            elif view_selection.mode == "compare years" and scene_selection.output_mode == "ndvi + change":
                baseline_ndvi = service.build_ndvi_raster(
                    view_selection.baseline_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
                comparison_ndvi = service.build_ndvi_raster(
                    view_selection.comparison_year,
                    scene_id=scene_selection.comparison_scene_id,
                )
                ndvi_diff = comparison_ndvi - baseline_ndvi
            else:
                baseline_rgb = service.build_rgb_raster(
                    view_selection.baseline_year,
                    scene_id=scene_selection.baseline_scene_id,
                )
                comparison_rgb = service.build_rgb_raster(
                    view_selection.comparison_year,
                    scene_id=scene_selection.comparison_scene_id,
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

    if view_selection.mode == "single year" and scene_selection.output_mode == "classified":
        season_text = format_season_window(view_selection.selected_year, analysis_settings)
        scene_stamp = format_scene_stamp(baseline_scene_meta)
        st.subheader(f"{view_selection.selected_year} classified NDVI ({scene_stamp})")
        st.caption(f"Season window: {season_text}")

        class_summary = summarize_class_distribution(baseline.raster)
        st.dataframe(build_class_distribution_rows(class_summary), width="stretch", hide_index=True)

        st.pyplot(
            render_array(
                baseline.raster,
                f"{view_selection.selected_year} classification | {scene_stamp}",
                mode="classified",
            )
        )
        return

    if view_selection.mode == "single year" and scene_selection.output_mode == "ndvi":
        season_text = format_season_window(view_selection.selected_year, analysis_settings)
        scene_stamp = format_scene_stamp(baseline_scene_meta)
        st.subheader(f"{view_selection.selected_year} NDVI ({scene_stamp})")
        st.caption(f"Season window: {season_text}")
        st.pyplot(
            render_array(
                baseline_ndvi,
                f"{view_selection.selected_year} NDVI | {scene_stamp}",
                mode="ndvi",
            )
        )
        return

    if view_selection.mode == "single year" and scene_selection.output_mode == "rgb":
        season_text = format_season_window(view_selection.selected_year, analysis_settings)
        scene_stamp = format_scene_stamp(baseline_scene_meta)
        st.subheader(f"{view_selection.selected_year} RGB ({scene_stamp})")
        st.caption(f"Season window: {season_text}")
        st.pyplot(
            render_rgb_array(
                baseline_rgb,
                f"{view_selection.selected_year} RGB | {scene_stamp}",
            )
        )
        return

    if view_selection.mode == "compare years" and scene_selection.output_mode == "ndvi + change":
        baseline_season = format_season_window(view_selection.baseline_year, analysis_settings)
        comparison_season = format_season_window(view_selection.comparison_year, analysis_settings)
        baseline_stamp = format_scene_stamp(baseline_scene_meta)
        comparison_stamp = format_scene_stamp(comparison_scene_meta)
        st.subheader("NDVI comparison")
        st.caption(
            f"Baseline season: {baseline_season} ({baseline_stamp}) | "
            f"Comparison season: {comparison_season} ({comparison_stamp})"
        )
        render_change_percentiles(summarize_change_percentiles(ndvi_diff))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"{view_selection.baseline_year} NDVI ({baseline_stamp})")
            st.pyplot(render_array(baseline_ndvi, f"{view_selection.baseline_year} NDVI | {baseline_stamp}", mode="ndvi"))
        with col2:
            st.subheader("NDVI change")
            st.pyplot(
                render_array(
                    ndvi_diff,
                    f"{view_selection.baseline_year} to {view_selection.comparison_year} NDVI change",
                    mode="continuous",
                )
            )
        with col3:
            st.subheader(f"{view_selection.comparison_year} NDVI ({comparison_stamp})")
            st.pyplot(render_array(comparison_ndvi, f"{view_selection.comparison_year} NDVI | {comparison_stamp}", mode="ndvi"))
        return

    if view_selection.mode == "compare years" and scene_selection.output_mode == "rgb":
        baseline_season = format_season_window(view_selection.baseline_year, analysis_settings)
        comparison_season = format_season_window(view_selection.comparison_year, analysis_settings)
        baseline_stamp = format_scene_stamp(baseline_scene_meta)
        comparison_stamp = format_scene_stamp(comparison_scene_meta)
        st.subheader("RGB comparison")
        st.caption(
            f"Baseline season: {baseline_season} ({baseline_stamp}) | "
            f"Comparison season: {comparison_season} ({comparison_stamp})"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{view_selection.baseline_year} RGB ({baseline_stamp})")
            st.pyplot(render_rgb_array(baseline_rgb, f"{view_selection.baseline_year} RGB | {baseline_stamp}"))
        with col2:
            st.subheader(f"{view_selection.comparison_year} RGB ({comparison_stamp})")
            st.pyplot(render_rgb_array(comparison_rgb, f"{view_selection.comparison_year} RGB | {comparison_stamp}"))
        return

    baseline_summary = summarize_class_distribution(baseline.raster)
    comparison_summary = summarize_class_distribution(comparison.raster)
    change_percentiles = summarize_change_percentiles(change.ndvi_diff)
    baseline_season = format_season_window(view_selection.baseline_year, analysis_settings)
    comparison_season = format_season_window(view_selection.comparison_year, analysis_settings)
    baseline_stamp = format_scene_stamp(baseline_scene_meta)
    comparison_stamp = format_scene_stamp(comparison_scene_meta)

    st.subheader("Land-cover summary")
    st.caption(
        f"Baseline season: {baseline_season} ({baseline_stamp}) | "
        f"Comparison season: {comparison_season} ({comparison_stamp})"
    )

    st.dataframe(build_comparison_rows(baseline_summary, comparison_summary), width="stretch", hide_index=True)
    render_change_percentiles(change_percentiles)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(f"{view_selection.baseline_year} classified raster ({baseline_stamp})")
        st.pyplot(
            render_array(
                baseline.raster,
                f"{view_selection.baseline_year} classification | {baseline_stamp}",
                mode="classified",
            )
        )

    with col2:
        st.subheader("NDVI change")
        st.pyplot(
            render_array(
                change.ndvi_diff,
                f"{view_selection.baseline_year} to {view_selection.comparison_year} change",
                mode="continuous",
            )
        )

    with col3:
        st.subheader(f"{view_selection.comparison_year} classified raster ({comparison_stamp})")
        st.pyplot(
            render_array(
                comparison.raster,
                f"{view_selection.comparison_year} classification | {comparison_stamp}",
                mode="classified",
            )
        )

    st.caption("Values are derived from the raster service pipeline for the selected years and seasonal settings.")


if __name__ == "__main__":
    main()

