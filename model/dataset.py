"""
Dataloader for MODIS LST downscaling — Colorado Front Range, 2022–2023.

Reads the post-2026-04 layout of `akhot2/downscaling`:

    data/
        MODIS/        MOD11A1.A*.h09v05.061.*_cropped.hdf   (HDF5; 1.39 km, daily)
        NDVI/         MOD13Q1.A*.h09v05.061.*_cropped.hdf   (HDF5; 347 m, 16-day)
        DEM/          dem_aoi_{4km,1km,250m}.tif + dem_aoi.tif (native ~30 m)
        LULC_final.tiff      30 m, single-layer, Albers AEA
        aoi/colorado_bbox.shp

Reference grids:
    HR  = DEM 1 km grid (112, 87) in EPSG:32613.
    LR  = DEM 4 km grid (28, 22)  in EPSG:32613.

Each sample (one date):
    lr_lst       (1, 28, 22)   °C   coarsened MODIS LST_Day_1km @ 4 km
    hr_lst       (1, 112, 87)  °C   MODIS LST_Day_1km @ 1 km          ← target
    ndvi         (1, 112, 87)  -    MOD13Q1 NDVI nearest composite, 1 km
    dem          (1, 112, 87)  m    static
    lulc         (1, 112, 87)  cls  static, NLCD 2022 land-cover class
    lulc_onehot  (N, 112, 87)  float one-hot of `lulc` over classes present in the AOI
    data_mask    (1, 112, 87)  bool finite LST (cloud/QC mask only)
    valid_mask   (1, 112, 87)  bool data_mask AND in this split's spatial blocks
    date         str           ISO yyyy-mm-dd

`N` (LULC class count) is detected from the AOI at init and exposed as
`dataset.lulc_classes` (sorted list of class codes).

Splits combine a temporal block (train=Jan–Sep'22, val=Oct–Dec'22, test=2023)
with a spatial holdout: the AOI is divided into BLOCK_GRID coarse blocks,
stratified by (urban-frac × elevation), and ~60/20/20 of blocks go to
train/val/test. `valid_mask` restricts the loss to each split's own blocks
so val/test pixels are spatially disjoint from train pixels — this defeats
spatial autocorrelation while keeping urban + rural coverage in every split.
"""

from __future__ import annotations

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
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_CRS = "EPSG:32613"  # UTM 13N — every gridded layer lives here
SPLIT_RANGES = {
    "train": (datetime(2022, 1, 1), datetime(2022, 9, 30)),
    "val":   (datetime(2022, 10, 1), datetime(2022, 12, 31)),
    "test":  (datetime(2023, 1, 1), datetime(2023, 12, 31)),
}
SPLIT_IDX = {"train": 0, "val": 1, "test": 2}

# 5×6 coarse block grid over the HR (112, 87) AOI: 30 blocks of ~22×15 km.
# Bigger than LST spatial autocorrelation (~5–15 km) so adjacent val/train
# blocks aren't redundant; small enough that 60/20/20 = 18/6/6 leaves enough
# blocks per split for stratified urban+rural sampling to balance out.
BLOCK_GRID = (5, 6)

# NLCD developed classes — anything starting with "Developed" (legend codes 21–24
# in legacy NLCD; 22–25 in NLCD Annual). We treat ≥21 and <30 as "urban".
URBAN_LULC_RANGE = (21, 30)

# DEM/LULC nodata sentinels in the source rasters
LULC_NODATA = 250


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _modis_date(filename: str) -> datetime:
    m = re.search(r"\.A(\d{4})(\d{3})\.", filename)
    return datetime(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)


def _read_h5_layer(path: str, layer: str) -> tuple[np.ndarray, Affine, str]:
    """Read an HDF5 layer + its rasterio transform/CRS attrs."""
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
    """Return (LST in Celsius with NaNs, src transform). Applies QC filter."""
    raw, tf, _ = _read_h5_layer(path, layer)
    qc_layer = "QC_Day" if "Day" in layer else "QC_Night"
    qc, _, _ = _read_h5_layer(path, qc_layer)
    valid = (raw >= 7500) & (raw <= 65535)
    mandatory = qc & 0b11
    quality = (mandatory == 0) | (mandatory == 1)
    lst = np.where(valid & quality, raw.astype(np.float32) * 0.02 - 273.15, np.nan)
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
    """Reproject a 2-D array onto a target grid."""
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
# Spatial split
# ---------------------------------------------------------------------------

def build_block_grid(hr_shape: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    """Return (H, W) int array labeling each HR pixel with its block id."""
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
    """Assign each block to train/val/test, stratified by urban-frac × elevation.

    Within each stratum we shuffle deterministically and take 60/20/20 of the
    blocks. This guarantees urban and rural blocks are present in every split,
    while keeping val/test spatially disjoint from train.
    """
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
    elev_hi = elev_mean > np.median(elev_mean)
    strata = urban_hi.astype(int) * 2 + elev_hi.astype(int)  # 0..3

    assignment: dict[int, str] = {}
    for s in range(4):
        members = np.where(strata == s)[0]
        if len(members) == 0:
            continue
        rng.shuffle(members)
        n = len(members)
        n_val = max(1, int(round(n * 0.20)))
        n_test = max(1, int(round(n * 0.20)))
        n_train = n - n_val - n_test
        for b in members[:n_train]:
            assignment[int(b)] = "train"
        for b in members[n_train:n_train + n_val]:
            assignment[int(b)] = "val"
        for b in members[n_train + n_val:]:
            assignment[int(b)] = "test"
    # Any block we missed (none expected): default to train
    for b in range(n_blocks):
        assignment.setdefault(b, "train")
    return assignment


def build_spatial_mask(
    block_id: np.ndarray, assignment: dict[int, str], split: str,
) -> np.ndarray:
    """Bool mask: True where the pixel's block is assigned to `split`."""
    target_blocks = {b for b, s in assignment.items() if s == split}
    return np.isin(block_id, list(target_blocks))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DownscalingDataset(Dataset):
    """One sample per MODIS date in the split's temporal range, full scene."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        lst_layer: str = "LST_Day_1km",
        download: bool = True,
        block_seed: int = 42,
        min_valid_frac: float = 0.05,
    ):
        assert split in SPLIT_RANGES, f"split must be in {list(SPLIT_RANGES)}"
        super().__init__()
        self.root = root
        self.split = split
        self.lst_layer = lst_layer
        self.min_valid_frac = min_valid_frac

        self.modis_dir = os.path.join(root, "MODIS")
        self.ndvi_dir = os.path.join(root, "NDVI")
        self.dem_dir = os.path.join(root, "DEM")
        self.lulc_path = os.path.join(root, "LULC_final.tiff")

        if download:
            self._download_if_missing()

        # --- HR/LR reference grids from DEM tiffs ---
        with rasterio.open(os.path.join(self.dem_dir, "dem_aoi_1km.tif")) as ds:
            self.hr_shape = ds.shape
            self.hr_transform = ds.transform
            self.hr_crs = str(ds.crs)
            dem_raw = ds.read(1).astype(np.float32)
            dem_nodata = ds.nodata
        if dem_nodata is not None:
            dem_raw[dem_raw == dem_nodata] = np.nan
        # Fill DEM NaNs with the spatial mean so models don't see sentinels.
        self.dem_hr = _fillna(dem_raw)
        with rasterio.open(os.path.join(self.dem_dir, "dem_aoi_4km.tif")) as ds:
            self.lr_shape = ds.shape
            self.lr_transform = ds.transform
            self.lr_crs = str(ds.crs)

        # --- Static covariates on HR grid ---
        self.lulc_hr = self._load_lulc_hr()
        # Sorted list of NLCD classes present in this AOI; defines one-hot order.
        self.lulc_classes = np.array(sorted(np.unique(self.lulc_hr).tolist()), dtype=np.int32)

        # --- Spatial split mask ---
        self.block_id = build_block_grid(self.hr_shape, BLOCK_GRID)
        self.block_assignment = assign_blocks_to_splits(
            self.block_id, self.lulc_hr, self.dem_hr, seed=block_seed,
        )
        self.spatial_mask = build_spatial_mask(
            self.block_id, self.block_assignment, split,
        )

        # --- File indices ---
        self.ndvi_index = self._index_ndvi()
        self.dates = self._index_modis_dates()
        self.dates = self._filter_by_valid_frac(self.dates)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _download_if_missing(self) -> None:
        modis_files = glob.glob(os.path.join(self.modis_dir, "*_cropped.hdf"))
        dem_files = glob.glob(os.path.join(self.dem_dir, "*.tif"))
        if len(modis_files) > 100 and len(dem_files) >= 4:
            return
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="akhot2/downscaling",
            repo_type="dataset",
            local_dir=self.root,
        )

    def _load_lulc_hr(self) -> np.ndarray:
        """Reproject LULC from native AEA 30 m → HR (1 km) UTM grid (mode)."""
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
        # Replace nodata with 0 (NLCD "Unclassified") so embeddings see a real class.
        lulc[lulc == LULC_NODATA] = 0
        return lulc

    def _index_modis_dates(self) -> list[tuple[datetime, str]]:
        """List (date, path) within this split's temporal range."""
        start, end = SPLIT_RANGES[self.split]
        out = []
        for f in sorted(glob.glob(os.path.join(self.modis_dir, "*_cropped.hdf"))):
            d = _modis_date(f)
            if start <= d <= end:
                out.append((d, f))
        return out

    def _index_ndvi(self) -> list[tuple[datetime, str]]:
        out = []
        for f in sorted(glob.glob(os.path.join(self.ndvi_dir, "*_cropped.hdf"))):
            out.append((_modis_date(f), f))
        return sorted(out)

    def _filter_by_valid_frac(
        self, dates: list[tuple[datetime, str]],
    ) -> list[tuple[datetime, str]]:
        """Drop scenes where < min_valid_frac of this split's pixels have valid LST.

        Reads only the small native (77×59) raw LST + QC arrays, so this is cheap.
        """
        if self.min_valid_frac <= 0:
            return dates
        kept = []
        for d, path in dates:
            lst, tf = read_modis_lst(path, self.lst_layer)
            hr = reproject_to(
                lst, tf, REF_CRS, self.hr_shape, self.hr_transform,
                resampling=Resampling.bilinear,
            )
            valid = np.isfinite(hr) & self.spatial_mask
            frac = valid.sum() / max(self.spatial_mask.sum(), 1)
            if frac >= self.min_valid_frac:
                kept.append((d, path))
        return kept

    def _nearest_ndvi(self, target: datetime) -> str:
        if not self.ndvi_index:
            raise RuntimeError("No NDVI files found")
        dates = [d for d, _ in self.ndvi_index]
        i = bisect.bisect_left(dates, target)
        cand = []
        if i < len(dates):
            cand.append(i)
        if i > 0:
            cand.append(i - 1)
        best = min(cand, key=lambda j: abs((dates[j] - target).days))
        return self.ndvi_index[best][1]

    @lru_cache(maxsize=64)
    def _ndvi_hr(self, ndvi_path: str) -> np.ndarray:
        ndvi, tf = read_ndvi(ndvi_path)
        return reproject_to(
            ndvi, tf, REF_CRS, self.hr_shape, self.hr_transform,
            resampling=Resampling.bilinear,
        )

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        date, modis_path = self.dates[idx]
        lst_native, tf = read_modis_lst(modis_path, self.lst_layer)

        hr_lst = reproject_to(
            lst_native, tf, REF_CRS, self.hr_shape, self.hr_transform,
            resampling=Resampling.bilinear,
        )
        lr_lst = reproject_to(
            lst_native, tf, REF_CRS, self.lr_shape, self.lr_transform,
            resampling=Resampling.average,
        )
        ndvi_hr = self._ndvi_hr(self._nearest_ndvi(date))

        data_mask = np.isfinite(hr_lst)
        valid = data_mask & self.spatial_mask

        # One-hot encode LULC against AOI-present classes (broadcasted equality).
        lulc_oh = (self.lulc_hr[None, ...] == self.lulc_classes[:, None, None]).astype(np.float32)

        return {
            "lr_lst":      torch.from_numpy(_fillna(lr_lst)).unsqueeze(0),
            "hr_lst":      torch.from_numpy(_fillna(hr_lst)).unsqueeze(0),
            "ndvi":        torch.from_numpy(_fillna(ndvi_hr, fill=0.0)).unsqueeze(0),
            "dem":         torch.from_numpy(self.dem_hr).unsqueeze(0),
            "lulc":        torch.from_numpy(self.lulc_hr.astype(np.int64)).unsqueeze(0),
            "lulc_onehot": torch.from_numpy(lulc_oh),
            "data_mask":   torch.from_numpy(data_mask).unsqueeze(0),
            "valid_mask":  torch.from_numpy(valid).unsqueeze(0),
            "date":        date.strftime("%Y-%m-%d"),
        }


def _fillna(arr: np.ndarray, fill: float | None = None) -> np.ndarray:
    """Replace NaNs with `fill` (default = nanmean of array, 0 if all NaN)."""
    arr = arr.astype(np.float32, copy=True)
    mask = ~np.isfinite(arr)
    if mask.any():
        if fill is None:
            v = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
        else:
            v = fill
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
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = DownscalingDataset(root=root, split=split, download=download,
                                **dataset_kwargs)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
