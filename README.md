# Temperature Downscaling

Machine-learning models for downscaling MODIS Land Surface Temperature
(LST) from **4 km → 1 km** over the Colorado Front Range (2022–2023).

The pipeline trains guided super-resolution networks that take a coarse
(4 km) MODIS LST scene plus 1 km HR covariates (NDVI, DEM, NLCD land
cover) and predict the 1 km LST field. Three classical baselines
(bicubic, BCSD, Lasso) provide reference points.

## Repository layout

```
.
├── model/
│   ├── dataset.py        DownscalingDataset + spatial/temporal split
│   ├── cnn.py            GuidedCNN — plain dual-branch baseline
│   └── attention_cnn.py  AttentionAugmentedCNN — adds CA + SA
├── notebooks/
│   ├── dataloader_viz.py     Inspect splits, covariates, sample patches
│   └── colabs/
│       ├── baselines.ipynb        Bicubic, BCSD, Lasso
│       ├── cnn_baseline.ipynb     Train GuidedCNN
│       └── attention_cnn.ipynb    Train AttentionAugmentedCNN
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

Hosted on Hugging Face: [`akhot2/downscaling`](https://huggingface.co/datasets/akhot2/downscaling).
Everything is pre-cropped to a 1°×1° AOI over the Colorado Front Range
(38.98–39.98 N, 105.82–104.82 W) and projected to UTM 13N (EPSG:32613).

| Layer | Native resolution | Format | Temporal |
|-------|-------------------|--------|----------|
| MODIS LST (`MOD11A1`) | 1.39 km | HDF5 | Daily |
| NDVI (`MOD13Q1`) | 347 m | HDF5 | 16-day composite |
| DEM | pre-coarsened to 4 km / 1 km / 250 m / native ~30 m | GeoTIFF | Static |
| NLCD land cover | 30 m (reprojected from Albers AEA) | GeoTIFF | Static, single layer |
| AOI bbox | — | Shapefile | Static |

Despite the `.hdf` extension the cropped MODIS / NDVI files are HDF5 — read
them with `h5py`, not `pyhdf`.

`model/dataset.py` builds an HR (1 km, 112×87) and an LR (4 km, 28×22)
reference grid from the pre-coarsened DEM tiffs, then reprojects every
other layer onto them.

## Splits

The split mixes a **temporal block** with a **spatial holdout** to
defeat both seasonal and spatial autocorrelation:

| Split | Dates | Spatial blocks | Scenes (after cloud filter) |
|-------|-------|----------------|----------------------------|
| train | Jan–Sep 2022 | 18 / 30 (60 %) | 234 |
| val   | Oct–Dec 2022 | 6 / 30 (20 %) | 67 |
| test  | 2023 | 6 / 30 (20 %) | 308 |

Blocks are a 5 × 6 grid over the AOI (~22 × 15 km each), assigned to
splits stratified by `(urban-fraction × elevation)` so every split spans
both Denver-area developed land and Front-Range high terrain. Each
sample carries a `valid_mask` = `(finite LST) ∧ (this split's blocks)`;
losses and metrics must be computed under that mask so val/test pixels
stay disjoint from train pixels.

## Method

Both deep models share the same dual-branch guided-SR design:

1. **LR branch** — residual blocks on the 4 km LST input.
2. **HR branch** — residual blocks on the HR covariate stack (NDVI + DEM
   + LULC one-hot).
3. **Fuse** the upsampled LR features with the HR features, refine with
   more residual blocks at HR.
4. Add the predicted residual on top of a **bicubic-upsampled LR
   baseline** so the network only learns the high-frequency correction.

`AttentionAugmentedCNN` adds Squeeze-and-Excitation channel attention
and a spatial-attention mask inside every residual block;
`GuidedCNN` is the same architecture without those modules and serves
as the ablation control.

## Quick start

```bash
pip install -r requirements.txt
```

Data downloads automatically the first time you instantiate
`DownscalingDataset` (via `huggingface_hub.snapshot_download`).

```python
from model.dataset import DownscalingDataset, get_dataloaders

loaders = get_dataloaders(root="data", batch_size=8)
batch = next(iter(loaders["train"]))
print(batch["lr_lst"].shape, batch["hr_lst"].shape)   # (8,1,28,22) (8,1,112,87)
```

```python
from model.attention_cnn import AttentionAugmentedCNN, make_hr_cov

n_classes = loaders["train"].dataset.lulc_classes.shape[0]
model = AttentionAugmentedCNN(cov_channels=2 + n_classes)
pred = model(batch["lr_lst"], make_hr_cov(batch))     # (8,1,112,87)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/notebooks/colabs/baselines.ipynb) | `colabs/baselines.ipynb` — bicubic, BCSD, Lasso |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/notebooks/colabs/cnn_baseline.ipynb) | `colabs/cnn_baseline.ipynb` — train `GuidedCNN` |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fresleven/downscaling/blob/main/notebooks/colabs/attention_cnn.ipynb) | `colabs/attention_cnn.ipynb` — train `AttentionAugmentedCNN` |

`notebooks/dataloader_viz.py` is a script (open as a Jupyter notebook
via the `# %%` cell markers) for inspecting the splits, LULC one-hot
channels, and sample LR/HR pairs.

## License

MIT.
