# Temperature Downscaling

Machine learning models for downscaling MODIS Land Surface Temperature over the Colorado Front Range.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/main.ipynb) | `main.ipynb` — Main pipeline |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/explore_data.ipynb) | `explore_data.ipynb` — Data exploration & visualization |

## Datasets

| Source | Resolution | Temporal | Role |
|--------|-----------|----------|------|
| MODIS MOD11A1 | 1 km | Daily | Land Surface Temperature target |
| Sentinel-2 | 10 m | ~2/month | NDVI, NDWI predictors (B03, B04, B08) |
| NLCD LULC | 30 m | Annual | Land cover predictor |
| ASTER GDEM v3 | 30 m | Static | Elevation predictor |

**Study area:** Colorado Front Range, 2022–2023

## Setup

```bash
pip install -r requirements.txt
```
