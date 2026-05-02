"""
Dataloader for MODIS LST downscaling — Colorado Front Range, 2022–2023.

Reads the post-2026-04 layout of `akhot2/downscaling`:

    data/
        MODIS/        MOD11A1.A*.h09v05.061.*_cropped.hdf   (HDF5; 1.39 km, daily)
        NDVI/         MOD13Q1.A*.h09v05.061.*_cropped.hdf   (HDF5; 347 m, 16-day)
        DEM/          dem_aoi_{4km,1km,250m}.tif + dem_aoi.tif (native ~30 m)
        LULC_final.tiff      30 m, single-layer, Albers AEA
        stations/     final_stations.csv  (lat/lon of validation stations)
        aoi/colorado_bbox.shp

Supported resolution pairs (lr_res → hr_res):
    "4km" → "1km"   standard task; 4x upscaling
    "1km" → "250m"  finer task; 4x upscaling
    (any combination of "4km", "1km", "250m" is accepted)

NOTE: MODIS LST is natively 1 km. For hr_res="250m" the target is bilinearly
interpolated from 1 km — not a true 250 m measurement. Swap in Landsat/ASTER
thermal data when you have it.

Each sample is one (date, spatial-block) pair:
    lr_lst       (1, LR_H, LR_W)   full LR scene at lr_res
    hr_lst       (1, bH, bW)        HR target cropped to this block at hr_res
    ndvi         (1, bH, bW)        nearest NDVI composite, reprojected to hr_res
    dem          (1, bH, bW)        static DEM at hr_res
    lulc         (1, bH, bW)        NLCD class at hr_res
    lulc_onehot  (N, bH, bW)        one-hot over AOI-present classes
    loc          (2, bH, bW)        normalized UTM (x, y) ∈ [0, 1] for each pixel
    data_mask    (1, bH, bW)        finite LST
    valid_mask   (1, bH, bW)        data_mask AND not a station holdout pixel
    date         str                ISO yyyy-mm-dd
    block_id     int

Splits:
  Temporal — train: Jan–Sep 2022, val: Oct–Dec 2022, test: 2023.
  Spatial  — the HR AOI is divided into BLOCK_GRID coarse blocks, stratified by
             (urban-frac × elevation), ~60/20/20 to train/val/test.
  Stations — blocks containing any station from `stations/final_stations.csv`
             are dropped from the training split entirely. Val/test keep them
             so they can be used for point-based evaluation.
"""

from __future__ import annotations

import csv
import os
import re
import glob
import bisect
import warnings
from datetime import datetime, timedelta
from functools import lru_cache

import h5py
import numpy as np
import rasterio
import torch
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine, rowcol
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_CRS = "EPSG:32613"  # UTM 13N

SPLIT_RANGES = {
    "train": (datetime(2022, 1, 1), datetime(2022, 9, 30)),
    "val":   (datetime(2022, 10, 1), datetime(2022, 12, 31)),
    "test":  (datetime(2023, 1, 1), datetime(2023, 12, 31)),
}

# DEM filename for each supported resolution label
RESOLUTION_DEMS = {
    "4km":  "dem_aoi_4km.tif",
    "1km":  "dem_aoi_1km.tif",
    "250m": "dem_aoi_250m.tif",
}

# 5×6 coarse block grid — 30 blocks, each ~4× the LST autocorrelation length.
BLOCK_GRID = (5, 6)

URBAN_LULC_RANGE = (21, 30)
LULC_NODATA = 250


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _modis_date(filename: str) -> datetime:
    m = re.search(r"\.A(\d{4})(\d{3})\.", filename)
    return datetime(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)


def _read_h5_layer(path: str, layer: str) -> tuple[np.ndarray, Affine, str]:
    with h5py.File(path, "r") as h:
        ds = h[layer]
        arr = ds[:]
        tf_arr = np.asarray(ds.attrs["transform"], dtype=float)
        crs_attr = ds.attrs["crs"]
        if isinstance(crs_attr, bytes):
            crs_attr = crs_attr.decode()
    transform = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    return arr, transform, str(crs_attr)


def read_modis_lst(path: str, layer: str = "LST_Day_1km") -> tuple[np.ndarray, Affine]:
    raw, tf, _ = _read_h5_layer(path, layer)
    valid = raw != 0  # fill value is 0; all non-zero DN are valid temperatures
    lst = np.where(valid, raw.astype(np.float32) * 0.02 - 273.15, np.nan)
    return lst.astype(np.float32), tf


def read_ndvi(path: str, layer: str = "250m 16 days NDVI") -> tuple[np.ndarray, Affine]:
    raw, tf, _ = _read_h5_layer(path, layer)
    arr = raw.astype(np.float32) * 1e-4
    arr[(raw == -3000) | (raw < -2000) | (raw > 10000)] = np.nan
    return arr, tf


def reproject_to(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: str = REF_CRS,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = np.nan,
) -> np.ndarray:
    dst = np.full(dst_shape, nodata, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=resampling,
    )
    return dst


# ---------------------------------------------------------------------------
# Spatial split helpers
# ---------------------------------------------------------------------------

def build_block_grid(hr_shape: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    H, W = hr_shape
    rows, cols = grid
    block_id = np.zeros((H, W), dtype=np.int32)
    row_edges = np.linspace(0, H, rows + 1, dtype=int)
    col_edges = np.linspace(0, W, cols + 1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            block_id[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]] = r * cols + c
    return block_id


def assign_blocks_to_splits(
    block_id: np.ndarray,
    lulc_hr: np.ndarray,
    dem_hr: np.ndarray,
    seed: int = 42,
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    n_blocks = int(block_id.max()) + 1

    urban_frac = np.zeros(n_blocks)
    elev_mean = np.zeros(n_blocks)
    for b in range(n_blocks):
        m = block_id == b
        if not m.any():
            continue
        urban_frac[b] = ((lulc_hr[m] >= URBAN_LULC_RANGE[0]) &
                         (lulc_hr[m] < URBAN_LULC_RANGE[1])).mean()
        elev_mean[b] = np.nanmean(dem_hr[m])

    urban_hi = urban_frac > np.median(urban_frac)
    elev_hi   = elev_mean  > np.median(elev_mean)
    strata = urban_hi.astype(int) * 2 + elev_hi.astype(int)

    assignment: dict[int, str] = {}
    for s in range(4):
        members = np.where(strata == s)[0]
        if len(members) == 0:
            continue
        rng.shuffle(members)
        n = len(members)
        n_val   = max(1, int(round(n * 0.20)))
        n_test  = max(1, int(round(n * 0.20)))
        n_train = n - n_val - n_test
        for b in members[:n_train]:
            assignment[int(b)] = "train"
        for b in members[n_train:n_train + n_val]:
            assignment[int(b)] = "val"
        for b in members[n_train + n_val:]:
            assignment[int(b)] = "test"
    for b in range(n_blocks):
        assignment.setdefault(b, "train")
    return assignment


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DownscalingDataset(Dataset):
    """Each sample = one (date, spatial-block) pair.

    lr_lst is the full LR scene so the model retains global thermal context.
    All HR tensors are cropped to the block.
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        lr_res: str = "4km",
        hr_res: str = "1km",
        lst_layer: str = "LST_Day_1km",
        download: bool = True,
        block_seed: int = 42,
        min_valid_frac: float = 0.05,
    ):
        assert split in SPLIT_RANGES, f"split must be in {list(SPLIT_RANGES)}"
        assert lr_res in RESOLUTION_DEMS, f"lr_res must be one of {list(RESOLUTION_DEMS)}"
        assert hr_res in RESOLUTION_DEMS, f"hr_res must be one of {list(RESOLUTION_DEMS)}"
        super().__init__()
        self.root = root
        self.split = split
        self.lr_res = lr_res
        self.hr_res = hr_res
        self.lst_layer = lst_layer
        self.min_valid_frac = min_valid_frac

        self.modis_dir = os.path.join(root, "MODIS")
        self.ndvi_dir  = os.path.join(root, "NDVI")
        self.dem_dir   = os.path.join(root, "DEM")
        self.lulc_path = os.path.join(root, "LULC_final.tiff")
        self.station_csv = os.path.join(root, "stations", "final_stations.csv")

        if download:
            self._download_if_missing()

        # --- HR grid ---
        with rasterio.open(os.path.join(self.dem_dir, RESOLUTION_DEMS[hr_res])) as ds:
            self.hr_shape     = ds.shape
            self.hr_transform = ds.transform
            self.hr_crs       = str(ds.crs)
            dem_raw           = ds.read(1).astype(np.float32)
            dem_nodata        = ds.nodata
        if dem_nodata is not None:
            dem_raw[dem_raw == dem_nodata] = np.nan
        self.dem_hr = _fillna(dem_raw)

        # --- LR grid ---
        with rasterio.open(os.path.join(self.dem_dir, RESOLUTION_DEMS[lr_res])) as ds:
            self.lr_shape     = ds.shape
            self.lr_transform = ds.transform
            self.lr_crs       = str(ds.crs)

        # --- Static covariates on HR grid ---
        self.lulc_hr      = self._load_lulc_hr()
        self.lulc_classes = np.array(sorted(np.unique(self.lulc_hr).tolist()), dtype=np.int32)

        # --- Spatial split ---
        self.block_id         = build_block_grid(self.hr_shape, BLOCK_GRID)
        self.block_assignment = assign_blocks_to_splits(
            self.block_id, self.lulc_hr, self.dem_hr, seed=block_seed,
        )
        # Precompute bounding box of each block in HR pixel coords
        self.block_bboxes = _compute_block_bboxes(self.block_id)
        # Max block size — all crops are padded to this so the collator can stack them
        self.block_h = max(r1 - r0 for r0, r1, c0, c1 in self.block_bboxes.values())
        self.block_w = max(c1 - c0 for r0, r1, c0, c1 in self.block_bboxes.values())

        # Blocks containing a station are dropped from training only
        self.station_blocks = self._station_block_ids()

        # Blocks belonging to this split — these are the sample atoms
        self.split_blocks = sorted(
            b for b, s in self.block_assignment.items()
            if s == split and (split != "train" or b not in self.station_blocks)
        )

        # spatial_mask is still needed for _filter_by_valid_frac
        self._spatial_mask = np.isin(self.block_id, self.split_blocks)

        # --- Location encoding: normalized UTM coords for every HR pixel ---
        self.loc_x, self.loc_y = _build_location_encoding(self.hr_shape, self.hr_transform)

        # --- File indices ---
        self.ndvi_index = self._index_ndvi()
        self.dates      = self._index_modis_dates()
        self.dates      = self._filter_by_valid_frac(self.dates)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _download_if_missing(self) -> None:
        modis_files = glob.glob(os.path.join(self.modis_dir, "*_cropped.hdf"))
        dem_files   = glob.glob(os.path.join(self.dem_dir, "*.tif"))
        ndvi_files  = glob.glob(os.path.join(self.ndvi_dir, "*_cropped.hdf"))
        if len(modis_files) > 100 and len(dem_files) >= 4 and len(ndvi_files) > 0:
            return
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="akhot2/downscaling",
            repo_type="dataset",
            local_dir=self.root,
        )

    def _load_lulc_hr(self) -> np.ndarray:
        with rasterio.open(self.lulc_path) as src:
            lulc = np.full(self.hr_shape, LULC_NODATA, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=lulc,
                src_nodata=LULC_NODATA,
                dst_nodata=LULC_NODATA,
                dst_transform=self.hr_transform,
                dst_crs=self.hr_crs,
                resampling=Resampling.mode,
            )
        lulc[lulc == LULC_NODATA] = 0
        return lulc

    def _station_block_ids(self) -> set[int]:
        if not os.path.exists(self.station_csv):
            return set()
        proj = Transformer.from_crs("EPSG:4326", self.hr_crs, always_xy=True)
        blocks: set[int] = set()
        with open(self.station_csv, newline="") as f:
            for row in csv.DictReader(f):
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                x, y = proj.transform(lon, lat)
                r, c = rowcol(self.hr_transform, x, y)
                r, c = int(r), int(c)
                if 0 <= r < self.hr_shape[0] and 0 <= c < self.hr_shape[1]:
                    blocks.add(int(self.block_id[r, c]))
        return blocks

    def _index_modis_dates(self) -> list[tuple[datetime, str]]:
        start, end = SPLIT_RANGES[self.split]
        out = []
        for f in sorted(glob.glob(os.path.join(self.modis_dir, "*_cropped.hdf"))):
            d = _modis_date(f)
            if start <= d <= end:
                out.append((d, f))
        return out

    def _index_ndvi(self) -> list[tuple[datetime, str]]:
        out = [(_modis_date(f), f)
               for f in sorted(glob.glob(os.path.join(self.ndvi_dir, "*_cropped.hdf")))]
        return sorted(out)

    def _filter_by_valid_frac(
        self, dates: list[tuple[datetime, str]],
    ) -> list[tuple[datetime, str]]:
        if self.min_valid_frac <= 0:
            return dates
        kept = []
        for d, path in dates:
            lst, tf = read_modis_lst(path, self.lst_layer)
            hr = reproject_to(lst, tf, REF_CRS, self.hr_shape, self.hr_transform,
                               resampling=Resampling.bilinear)
            valid = np.isfinite(hr) & self._spatial_mask
            frac  = valid.sum() / max(self._spatial_mask.sum(), 1)
            if frac >= self.min_valid_frac:
                kept.append((d, path))
        return kept

    def _nearest_ndvi(self, target: datetime) -> str:
        if not self.ndvi_index:
            raise RuntimeError("No NDVI files found")
        dates = [d for d, _ in self.ndvi_index]
        i = bisect.bisect_left(dates, target)
        cand = [j for j in (i, i - 1) if 0 <= j < len(dates)]
        best = min(cand, key=lambda j: abs((dates[j] - target).days))
        return self.ndvi_index[best][1]

    @lru_cache(maxsize=64)
    def _ndvi_hr(self, ndvi_path: str) -> np.ndarray:
        ndvi, tf = read_ndvi(ndvi_path)
        return reproject_to(ndvi, tf, REF_CRS, self.hr_shape, self.hr_transform,
                             resampling=Resampling.bilinear)

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dates) * len(self.split_blocks)

    def __getitem__(self, idx: int) -> dict:
        date_idx  = idx // len(self.split_blocks)
        block_idx = idx %  len(self.split_blocks)
        block_id  = self.split_blocks[block_idx]
        r0, r1, c0, c1 = self.block_bboxes[block_id]

        date, modis_path = self.dates[date_idx]
        lst_native, tf = read_modis_lst(modis_path, self.lst_layer)

        # Full LR scene — gives the model global thermal context
        lr_lst = reproject_to(lst_native, tf, REF_CRS, self.lr_shape, self.lr_transform,
                               resampling=Resampling.average)

        # HR scene cropped to this block
        hr_lst_full = reproject_to(lst_native, tf, REF_CRS, self.hr_shape, self.hr_transform,
                                    resampling=Resampling.bilinear)
        hr_block = hr_lst_full[r0:r1, c0:c1]

        ndvi_block = self._ndvi_hr(self._nearest_ndvi(date))[r0:r1, c0:c1]

        data_mask = np.isfinite(hr_block)
        valid     = data_mask

        lulc_block = self.lulc_hr[r0:r1, c0:c1]
        lulc_oh    = (lulc_block[None] == self.lulc_classes[:, None, None]).astype(np.float32)

        loc = np.stack([self.loc_x[r0:r1, c0:c1], self.loc_y[r0:r1, c0:c1]])  # (2, bH, bW)

        # Pad to max block size so all samples collate to the same shape.
        # valid_mask is False for padded pixels so they never enter the loss.
        H, W = self.block_h, self.block_w
        return {
            "lr_lst":      torch.from_numpy(_fillna(lr_lst)).unsqueeze(0),
            "hr_lst":      torch.from_numpy(_pad2d(_fillna(hr_block), H, W)).unsqueeze(0),
            "ndvi":        torch.from_numpy(_pad2d(_fillna(ndvi_block, fill=0.0), H, W)).unsqueeze(0),
            "dem":         torch.from_numpy(_pad2d(self.dem_hr[r0:r1, c0:c1], H, W)).unsqueeze(0),
            "lulc":        torch.from_numpy(_pad2d(lulc_block.astype(np.int64), H, W)).unsqueeze(0),
            "lulc_onehot": torch.from_numpy(_pad2d(lulc_oh, H, W)),
            "loc":         torch.from_numpy(_pad2d(loc, H, W)),
            "data_mask":   torch.from_numpy(_pad2d(data_mask, H, W)).unsqueeze(0),
            "valid_mask":  torch.from_numpy(_pad2d(valid, H, W)).unsqueeze(0),
            "date":        date.strftime("%Y-%m-%d"),
            "block_id":    block_id,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _pad2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad the last two dimensions of `arr` to (target_h, target_w) with zeros."""
    ph = target_h - arr.shape[-2]
    pw = target_w - arr.shape[-1]
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(arr, pad)


def _compute_block_bboxes(block_id: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    n_blocks = int(block_id.max()) + 1
    bboxes = {}
    for b in range(n_blocks):
        m    = block_id == b
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        bboxes[b] = (int(rows[0]), int(rows[-1]) + 1,
                     int(cols[0]), int(cols[-1]) + 1)
    return bboxes


def _build_location_encoding(
    hr_shape: tuple[int, int], hr_transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized UTM pixel-center coordinates in [0, 1] over the AOI."""
    H, W = hr_shape
    tf   = hr_transform
    xs   = tf.c + (np.arange(W, dtype=np.float64) + 0.5) * tf.a   # easting
    ys   = tf.f + (np.arange(H, dtype=np.float64) + 0.5) * tf.e   # northing (e < 0)
    xx, yy = np.meshgrid(xs, ys)
    x_norm = ((xx - xx.min()) / (xx.max() - xx.min())).astype(np.float32)
    y_norm = ((yy - yy.min()) / (yy.max() - yy.min())).astype(np.float32)
    return x_norm, y_norm


def _fillna(arr: np.ndarray, fill: float | None = None) -> np.ndarray:
    arr  = arr.astype(np.float32, copy=True)
    mask = ~np.isfinite(arr)
    if mask.any():
        v = (float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0) if fill is None else fill
        arr[mask] = v
    return arr


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str = "data",
    batch_size: int = 8,
    num_workers: int = 0,
    download: bool = True,
    lr_res: str = "4km",
    hr_res: str = "1km",
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = DownscalingDataset(
            root=root, split=split, lr_res=lr_res, hr_res=hr_res,
            download=download, **dataset_kwargs,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
