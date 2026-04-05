# Temperature Downscaling

Machine learning models for downscaling MODIS Land Surface Temperature over the Colorado Front Range.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/main.ipynb) | `main.ipynb` — Main pipeline |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/explore_data.ipynb) | `explore_data.ipynb` — Data exploration & visualization |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/baselines.ipynb) | `baselines.ipynb` — BCSD & Lasso downscaling baselines (5km → 1km) |

## Dataset

All data is hosted on Hugging Face: [`akhot2/downscaling`](https://huggingface.co/datasets/akhot2/downscaling)

**Study area:** Colorado Front Range, 2022–2023

### Structure

```
akhot2/downscaling/
├── ASTER/            # ASTER GDEM v3 — 30m elevation (static)
│   ├── *_dem.tif     # Elevation tiles (4 tiles: N39–N40, W105–W106)
│   └── *_num.tif     # Observation count quality files
├── bounding box/     # Study area shapefile
│   ├── *.shp/shx/dbf/prj/cpg
├── LULC/             # NLCD Land Cover — 30m, annual
│   ├── Annual_NLCD_LndCov_*.tiff   # Land cover rasters (2021–2024)
│   ├── nlcd_legend.csv
│   └── metadata/auxiliary files
├── MODIS/            # MOD11A1 — 1km daily LST
│   └── MOD11A1.A{YYYYDDD}.h{HH}v{VV}.061.*.hdf
└── Sentinel2/        # Sentinel-2 L2A — 10m, ~bimonthly
    └── S2{A,B}_MSIL2A_{date}_{tile}_{band}.tif
```

### Sources

| Source | Resolution | Temporal | Format | Role |
|--------|-----------|----------|--------|------|
| MODIS MOD11A1 | 1 km | Daily | HDF-EOS | Land Surface Temperature target |
| Sentinel-2 L2A | 10 m | ~2/month | GeoTIFF | NDVI, NDWI predictors (B03, B04, B08) |
| NLCD LULC | 30 m | Annual | GeoTIFF | Land cover predictor |
| ASTER GDEM v3 | 30 m | Static | GeoTIFF | Elevation predictor |

### Coverage

- **MODIS tiles:** h09v04, h09v05, h10v04
- **Sentinel-2 tiles:** T13SDD, T13SED, T13TEE
- **ASTER tiles:** N39–N40, W105–W106 (2×2 grid)
- **Temporal range:** 2022–2023

## Setup

```bash
pip install -r requirements.txt
```

Data is downloaded automatically when running the notebooks via `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='akhot2/downscaling', repo_type='dataset', local_dir='data')
```
