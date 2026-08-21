import os
import sys
from pathlib import Path

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
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="NDVI change")

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


def main():
    st.set_page_config(
        page_title="Greyback Lake Land Use",
        page_icon=":material/terrain:",
        layout="wide",
    )
    st.title("Greyback Lake vegetation analysis")
    st.caption("Explore seasonal vegetation patterns and compare NDVI-driven land cover change across years.")

    with st.sidebar:
        st.subheader(":material/compare_arrows: Mode")
        st.caption("Choose whether to inspect one year or compare two years side by side.")
        mode = st.segmented_control("View mode", options=["single year", "compare years"], default="compare years")

        if mode == "single year":
            selected_year = st.slider("Selected year", min_value=2013, max_value=2025, value=2013, step=1, help="Year used for a single-season vegetation classification.")
            baseline_year = selected_year
            comparison_year = None
        else:
            baseline_year = st.slider("Baseline year", min_value=2013, max_value=2025, value=2013, step=1)
            comparison_year = st.slider("Comparison year", min_value=2013, max_value=2025, value=2024, step=1)
            selected_year = baseline_year

        st.markdown("---")
        st.subheader(":material/tune: Analysis settings")
        st.caption("Adjust the seasonal window used to fetch and aggregate Landsat imagery for each year.")
        season_start_month = st.slider("Season start month", min_value=1, max_value=12, value=7, help="Month used to begin the seasonal window.")
        season_start_day = st.slider("Season start day", min_value=1, max_value=31, value=1, help="Day within the starting month.")
        duration_days = st.slider("Season length (days)", min_value=7, max_value=180, value=31, help="Number of days included in the window for each year.")
        max_cloud = st.slider("Cloud filter (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, help="The maximum acceptable cloud cover before a scene is excluded.")

    service = load_service(
        season_start_month=season_start_month,
        season_start_day=season_start_day,
        duration_days=duration_days,
        max_cloud=max_cloud,
    )

    try:
        with st.spinner("Loading raster data..."):
            if mode == "compare years":
                baseline = service.build_classified_raster(baseline_year)
                comparison = service.build_classified_raster(comparison_year)
                change = service.build_change_raster(baseline_year, comparison_year)
            else:
                baseline = service.build_classified_raster(selected_year)
                comparison = None
                change = None
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

    if mode == "single year":
        st.subheader(f"{selected_year} classified NDVI")
        st.caption("Classified vegetation cover for the selected season and location.")

        class_summary = summarize_class_distribution(baseline.raster)
        summary_rows = [
            {"Class": idx, "Vegetation class": CLASS_NAMES[idx], "Share of pixels": round(class_summary[idx], 2)}
            for idx in range(len(CLASS_NAMES))
        ]

        st.dataframe(summary_rows, width="stretch", hide_index=True)

        st.pyplot(
            render_array(
                baseline.raster,
                f"{selected_year} classification",
                mode="classified",
            )
        )
        return

    baseline_summary = summarize_class_distribution(baseline.raster)
    comparison_summary = summarize_class_distribution(comparison.raster)
    change_percentiles = summarize_change_percentiles(change.ndvi_diff)

    st.subheader("Land-cover summary")
    st.caption("Share of pixels by vegetation class in each selected year.")

    baseline_df = [
        {
            "Class": idx,
            "Vegetation class": CLASS_NAMES[idx],
            "Baseline %": round(baseline_summary[idx], 2),
            "Comparison %": round(comparison_summary[idx], 2),
            "Change % points": round(comparison_summary[idx] - baseline_summary[idx], 2),
        }
        for idx in range(len(CLASS_NAMES))
    ]
    st.dataframe(baseline_df, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(f"{baseline_year} classified raster")
        st.pyplot(
            render_array(
                baseline.raster,
                f"{baseline_year} classification",
                mode="classified",
            )
        )

    with col2:
        st.subheader("NDVI change")
        st.pyplot(
            render_array(
                change.ndvi_diff,
                f"{baseline_year} to {comparison_year} change",
                mode="continuous",
            )
        )

    with col3:
        st.subheader(f"{comparison_year} classified raster")
        st.pyplot(
            render_array(
                comparison.raster,
                f"{comparison_year} classification",
                mode="classified",
            )
        )

    st.caption("Values are derived from the raster service pipeline for the selected years and seasonal settings.")


if __name__ == "__main__":
    main()

