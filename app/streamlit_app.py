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


def load_service() -> RasterService:
    catalog = get_catalog()
    config = RasterConfig(
        catalog=catalog,
        collection_id="landsat-c2-l2",
        center_lat=49.630227703275324,
        center_lon=-119.42400064714786,
        half_side_km=5.0,
        season_start_month=7,
        season_start_day=1,
        duration_days=31,
        max_cloud=10.0,
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


def main():
    st.set_page_config(page_title="Greyback Lake Land Use", layout="wide")
    st.title("Greyback Lake vegetation change")

    service = load_service()
    baseline_year = st.slider("Baseline year", min_value=2013, max_value=2025, value=2013, step=1)
    comparison_year = st.slider("Comparison year", min_value=2013, max_value=2025, value=2024, step=1)

    try:
        with st.spinner("Loading raster data..."):
            baseline = service.build_classified_raster(baseline_year)
            comparison = service.build_classified_raster(comparison_year)
            change = service.build_change_raster(baseline_year, comparison_year)
    except NoScenesFoundError as exc:
        st.warning(str(exc))
        st.info("Try another year or widen the seasonal window for this location.")
        return
    except NoClearPixelsError as exc:
        st.warning(str(exc))
        st.info("Consider increasing the cloud threshold or selecting a different year.")
        return
    except Exception as exc:  # pragma: no cover - fallback for unexpected runtime issues
        st.error(f"Unexpected processing error: {exc}")
        return

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

    st.caption("Values are derived from the raster service pipeline for the selected years.")


if __name__ == "__main__":
    main()

