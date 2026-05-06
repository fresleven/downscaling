"""
Visualize Train / Val / Test spatial split over the original-resolution LULC.

Everything is displayed in UTM 13N (EPSG:32613) at 30 m so north is always up.
The LULC is reprojected from its native AEA CRS; split labels come from the
block-grid logic in model/dataset.py (reproduced exactly here).
"""

import csv
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, rowcol
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "data")
LULC_PATH   = os.path.join(DATA_ROOT, "LULC_final.tiff")
DEM_1KM     = os.path.join(DATA_ROOT, "DEM", "dem_aoi_1km.tif")
STATION_CSV = os.path.join(DATA_ROOT, "stations", "final_stations.csv")

# ---------------------------------------------------------------------------
# Constants (mirrors dataset.py)
# ---------------------------------------------------------------------------
BLOCK_GRID       = (5, 6)
URBAN_LULC_RANGE = (21, 30)
LULC_NODATA      = 250
REF_CRS          = "EPSG:32613"
DISPLAY_RES      = 30   # meters — keeps LULC at original 30 m resolution

NLCD_COLORS = {
    0:  (0,   0,   0),
    11: (70,  107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235,  62,  38),
    24: (171,   0,   0),
    31: (179, 172, 159),
    41: (104, 171,  95),
    42: ( 28,  95,  44),
    43: (181, 197, 143),
    52: (204, 184, 121),
    71: (223, 223, 194),
    81: (220, 217,  57),
    82: (171, 108,  40),
    90: (184, 217, 235),
    95: (108, 159, 184),
}
NLCD_NAMES = {
    0:  "No Data",     11: "Water",         12: "Snow/Ice",
    21: "Dev. Open",   22: "Dev. Low",       23: "Dev. Med.",    24: "Dev. High",
    31: "Barren",      41: "Decid. Forest",  42: "Evgr. Forest", 43: "Mixed Forest",
    52: "Shrub/Scrub", 71: "Grassland",      81: "Pasture",      82: "Crops",
    90: "Woody Wetland", 95: "Herb. Wetland",
}

SPLIT_COLORS = {"train": "#2166ac", "val": "#f4a11b", "test": "#d6604d"}
SPLIT_ALPHA  = 0.45


# ---------------------------------------------------------------------------
# Spatial-split helpers (exact copy of dataset.py logic)
# ---------------------------------------------------------------------------

def build_block_grid(hr_shape, grid=(5, 6)):
    H, W = hr_shape
    rows, cols = grid
    block_id = np.zeros((H, W), dtype=np.int32)
    row_edges = np.linspace(0, H, rows + 1, dtype=int)
    col_edges = np.linspace(0, W, cols + 1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            block_id[row_edges[r]:row_edges[r+1],
                     col_edges[c]:col_edges[c+1]] = r * cols + c
    return block_id


def assign_blocks_to_splits(block_id, lulc_hr, dem_hr, seed=42):
    rng = np.random.default_rng(seed)
    n_blocks = int(block_id.max()) + 1
    urban_frac = np.zeros(n_blocks)
    elev_mean  = np.zeros(n_blocks)
    for b in range(n_blocks):
        m = block_id == b
        if not m.any():
            continue
        urban_frac[b] = ((lulc_hr[m] >= URBAN_LULC_RANGE[0]) &
                         (lulc_hr[m] <  URBAN_LULC_RANGE[1])).mean()
        elev_mean[b]  = np.nanmean(dem_hr[m])

    urban_hi = urban_frac > np.median(urban_frac)
    elev_hi  = elev_mean  > np.median(elev_mean)
    strata   = urban_hi.astype(int) * 2 + elev_hi.astype(int)

    assignment: dict = {}
    for s in range(4):
        members = np.where(strata == s)[0]
        if len(members) == 0:
            continue
        rng.shuffle(members)
        n       = len(members)
        n_val   = max(1, int(round(n * 0.20)))
        n_test  = max(1, int(round(n * 0.20)))
        n_train = n - n_val - n_test
        for b in members[:n_train]:              assignment[int(b)] = "train"
        for b in members[n_train:n_train+n_val]: assignment[int(b)] = "val"
        for b in members[n_train+n_val:]:        assignment[int(b)] = "test"
    for b in range(n_blocks):
        assignment.setdefault(b, "train")
    return assignment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- 1 km DEM: defines the HR grid used for splitting ---
    print("Loading 1 km DEM …")
    with rasterio.open(DEM_1KM) as ds:
        hr_shape     = ds.shape
        hr_transform = ds.transform
        hr_crs       = str(ds.crs)          # EPSG:32613
        dem_raw      = ds.read(1).astype(np.float32)
        nodata       = ds.nodata
        dem_bounds   = ds.bounds
    if nodata is not None:
        dem_raw[dem_raw == nodata] = np.nan

    # --- Build 30 m UTM display grid that exactly covers the DEM AOI ---
    disp_left, disp_bottom = dem_bounds.left,  dem_bounds.bottom
    disp_right, disp_top   = dem_bounds.right, dem_bounds.top
    disp_w = int(round((disp_right  - disp_left)   / DISPLAY_RES))
    disp_h = int(round((disp_top    - disp_bottom)  / DISPLAY_RES))
    disp_transform = from_bounds(disp_left, disp_bottom, disp_right, disp_top,
                                 disp_w, disp_h)
    disp_shape = (disp_h, disp_w)
    disp_crs   = REF_CRS
    print(f"Display grid: {disp_h} × {disp_w} px  ({DISPLAY_RES} m UTM 13N)")

    # --- Reproject LULC (AEA 30 m) → display grid ---
    print("Reprojecting LULC to display grid …")
    lulc_disp = np.full(disp_shape, LULC_NODATA, dtype=np.uint8)
    with rasterio.open(LULC_PATH) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=lulc_disp,
            src_nodata=LULC_NODATA,
            dst_nodata=LULC_NODATA,
            dst_transform=disp_transform,
            dst_crs=disp_crs,
            resampling=Resampling.mode,
        )
    lulc_disp[lulc_disp == LULC_NODATA] = 0

    # --- Spatial split (uses 1 km grid, same as dataset.py) ---
    print("Computing spatial split …")
    # LULC on 1 km grid (for stratification)
    lulc_1km = np.full(hr_shape, LULC_NODATA, dtype=np.uint8)
    with rasterio.open(LULC_PATH) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=lulc_1km,
            src_nodata=LULC_NODATA,
            dst_nodata=LULC_NODATA,
            dst_transform=hr_transform,
            dst_crs=hr_crs,
            resampling=Resampling.mode,
        )
    lulc_1km[lulc_1km == LULC_NODATA] = 0

    block_id   = build_block_grid(hr_shape, BLOCK_GRID)
    assignment = assign_blocks_to_splits(block_id, lulc_1km, dem_raw)

    # Station blocks — mirrors dataset.py: excluded from train, kept in val/test
    station_block_set = set()
    if os.path.exists(STATION_CSV):
        proj_s = Transformer.from_crs("EPSG:4326", hr_crs, always_xy=True)
        with open(STATION_CSV, newline="") as f:
            for row in csv.DictReader(f):
                lat, lon = float(row["latitude"]), float(row["longitude"])
                x, y     = proj_s.transform(lon, lat)
                r, c     = rowcol(hr_transform, x, y)
                r, c     = int(r), int(c)
                if 0 <= r < hr_shape[0] and 0 <= c < hr_shape[1]:
                    station_block_set.add(int(block_id[r, c]))

    # Split-label raster at 1 km:
    #   1=train  2=val  3=test  4=excluded (train-assigned but contains a station)
    SPLIT_INT = {"train": 1, "val": 2, "test": 3}
    split_1km = np.zeros(hr_shape, dtype=np.float32)
    for b, s in assignment.items():
        if s == "train" and b in station_block_set:
            split_1km[block_id == b] = 4   # excluded
        else:
            split_1km[block_id == b] = SPLIT_INT[s]

    # Reproject split labels to display grid
    split_disp = np.zeros(disp_shape, dtype=np.float32)
    reproject(
        source=split_1km,
        destination=split_disp,
        src_transform=hr_transform,
        src_crs=hr_crs,
        dst_transform=disp_transform,
        dst_crs=disp_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    split_disp = np.round(split_disp).astype(np.uint8)
    print(f"  Split label coverage: train={( split_disp==1).sum():,}  "
          f"val={(split_disp==2).sum():,}  test={(split_disp==3).sum():,}  "
          f"nodata={(split_disp==0).sum():,}")

    # --- Station pixel coords in display grid ---
    stations_rc = []
    if os.path.exists(STATION_CSV):
        proj = Transformer.from_crs("EPSG:4326", disp_crs, always_xy=True)
        with open(STATION_CSV, newline="") as f:
            for row in csv.DictReader(f):
                lat, lon = float(row["latitude"]), float(row["longitude"])
                x, y     = proj.transform(lon, lat)
                r_f = (disp_top - y)  / DISPLAY_RES
                c_f = (x - disp_left) / DISPLAY_RES
                stations_rc.append((r_f, c_f))

    # --- Block boundary lines (edges between different split labels) ---
    ey = np.abs(np.diff(split_disp.astype(np.int16), axis=0)) > 0
    ex = np.abs(np.diff(split_disp.astype(np.int16), axis=1)) > 0
    # Convert to row/col arrays for plotting
    ry, cy = np.where(ey)
    rx, cx = np.where(ex)

    # --- Build LULC RGB image ---
    present = sorted(int(v) for v in np.unique(lulc_disp) if v != 0)
    lulc_rgb = np.zeros((*disp_shape, 3), dtype=np.uint8)
    for cls in present:
        c = NLCD_COLORS.get(cls, (128, 128, 128))
        lulc_rgb[lulc_disp == cls] = c

    # --- Build split RGBA overlay ---
    from matplotlib.colors import to_rgb as mpl_to_rgb
    SPLIT_RGBA = {
        1: (*[int(v*255) for v in mpl_to_rgb(SPLIT_COLORS["train"])],    int(SPLIT_ALPHA*255)),
        2: (*[int(v*255) for v in mpl_to_rgb(SPLIT_COLORS["val"])],      int(SPLIT_ALPHA*255)),
        3: (*[int(v*255) for v in mpl_to_rgb(SPLIT_COLORS["test"])],     int(SPLIT_ALPHA*255)),
        4: (*[int(v*255) for v in mpl_to_rgb("#888888")],                int(SPLIT_ALPHA*255)),  # excluded
    }
    overlay = np.zeros((*disp_shape, 4), dtype=np.uint8)
    for val, rgba in SPLIT_RGBA.items():
        m = split_disp == val
        overlay[m] = rgba

    # --- Plot ---
    print("Plotting …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    # --- Panel 1: LULC ---
    ax = axes[0]
    ax.imshow(lulc_rgb, interpolation="nearest", origin="upper")
    ax.set_title("Land Use / Land Cover  (30 m NLCD)", fontsize=13,
                 fontweight="bold", color="black")
    ax.axis("off")

    handles = [
        mpatches.Patch(color=[v/255 for v in NLCD_COLORS.get(c, (128,128,128))],
                       label=NLCD_NAMES.get(c, f"Class {c}"))
        for c in present
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              framealpha=0.85, ncol=2, facecolor="white",
              title="NLCD class", title_fontsize=8)

    # --- Panel 2: split overlay on greyscale LULC ---
    lulc_grey = (0.299 * lulc_rgb[:,:,0] +
                 0.587 * lulc_rgb[:,:,1] +
                 0.114 * lulc_rgb[:,:,2]).astype(np.uint8)
    ax = axes[1]
    ax.imshow(lulc_grey, cmap="gray", vmin=0, vmax=255,
              interpolation="nearest", origin="upper")
    ax.imshow(overlay, interpolation="nearest", origin="upper")

    # Block boundaries (thin white lines)
    if ry.size:
        ax.scatter(cy, ry, c="white", s=0.01, linewidths=0, rasterized=True)
    if rx.size:
        ax.scatter(cx, rx, c="white", s=0.01, linewidths=0, rasterized=True)

    # Stations
    if stations_rc:
        rs, cs = zip(*stations_rc)
        ax.scatter(cs, rs, marker="*", s=100, c="white",
                   edgecolors="black", linewidths=0.6, zorder=5, label="Station")

    ax.set_title("Train / Val / Test  Spatial Split  (30 m)", fontsize=13,
                 fontweight="bold", color="black")
    ax.axis("off")

    split_handles = [
        mpatches.Patch(color=SPLIT_COLORS["train"], alpha=0.7, label="Train  (Jan–Sep 2022)"),
        mpatches.Patch(color=SPLIT_COLORS["val"],   alpha=0.7, label="Val    (Oct–Dec 2022)"),
        mpatches.Patch(color=SPLIT_COLORS["test"],  alpha=0.7, label="Test   (2023)"),
        mpatches.Patch(color="#888888",             alpha=0.7, label="Excluded (station in train block)"),
    ]
    if stations_rc:
        split_handles.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                       markeredgecolor="black", markersize=10, label="Weather Station",
                       linewidth=0)
        )
    ax.legend(handles=split_handles, loc="lower left", fontsize=9,
              framealpha=0.85, facecolor="white",
              title="Temporal split", title_fontsize=9)

    out_path = os.path.join(os.path.dirname(__file__), "..", "split_viz.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
