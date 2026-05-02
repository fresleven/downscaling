# %% [markdown]
# # Temperature Downscaling — DataLoader Visualization
#
# Visualize the `DownscalingDataset` (block-as-sample, multi-resolution):
# temporal × spatial splits, LULC one-hot, sample LR / HR block pairs,
# and LST at 4 km / 1 km / 250 m in a shared spatial domain.

# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, '..')

from model.dataset import DownscalingDataset, get_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--no-download', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)

args = parser.parse_args(args=[]) if ipython is not None else parser.parse_args()

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## 1. Load Datasets

# %%
download = not args.no_download

train_ds = DownscalingDataset(root=args.root, split='train', download=download)
val_ds   = DownscalingDataset(root=args.root, split='val',   download=download)
test_ds  = DownscalingDataset(root=args.root, split='test',  download=download)

for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
    print(f"{name}: {len(ds.dates)} dates × {len(ds.split_blocks)} blocks = {len(ds)} samples")
print(f"\nHR shape: {train_ds.hr_shape}  LR shape: {train_ds.lr_shape}")
print(f"LULC classes ({len(train_ds.lulc_classes)}): {train_ds.lulc_classes.tolist()}")
print(f"Station blocks held out from training: {sorted(train_ds.station_blocks)}")

sample = train_ds[0]
print(f"\nSample 0 keys + shapes:")
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k:12s} {tuple(v.shape)}  {v.dtype}")
    else:
        print(f"  {k:12s} {v}")

# %% [markdown]
# ## 2. Temporal split — dates per month

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, ds) in zip(axes, [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]):
    if len(ds.dates) == 0:
        ax.set_title(f'{name} — 0 dates'); continue
    months = [d.strftime('%Y-%m') for d, _ in ds.dates]
    counts = Counter(months); keys = sorted(counts)
    ax.bar(range(len(keys)), [counts[k] for k in keys],
           color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{name} — {len(ds.dates)} dates / {len(ds)} samples')
    ax.set_ylabel('Dates'); ax.grid(axis='y', alpha=0.3)

plt.suptitle('Dates per month by split (after valid-fraction filter)', fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Spatial split — block assignment over the AOI

# %%
import matplotlib.colors as mcolors

label_arr = np.full(train_ds.hr_shape, -1, dtype=np.int8)
for b, s in train_ds.block_assignment.items():
    label_arr[train_ds.block_id == b] = {'train': 0, 'val': 1, 'test': 2}[s]

# Mark station-holdout blocks
station_overlay = np.zeros(train_ds.hr_shape, dtype=bool)
for b in train_ds.station_blocks:
    station_overlay[train_ds.block_id == b] = True

cmap = mcolors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im = axes[0].imshow(label_arr, cmap=cmap, vmin=0, vmax=2)
axes[0].imshow(station_overlay, cmap='Reds', alpha=0.4 * station_overlay)
axes[0].set_title('Block assignment (blue=train, orange=val, green=test)\nred overlay = station holdout')

axes[1].imshow(train_ds.dem_hr, cmap='terrain')
axes[1].set_title('DEM (m)')

axes[2].imshow(train_ds.lulc_hr, cmap='tab20')
axes[2].set_title('LULC (NLCD class)')

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Spatial split with terrain & land cover', fontsize=14)
plt.tight_layout(); plt.show()

print("Per-split spatial composition:")
for ds, name in [(train_ds, 'train'), (val_ds, 'val'), (test_ds, 'test')]:
    m = ds._spatial_mask
    urban = ((ds.lulc_hr[m] >= 21) & (ds.lulc_hr[m] < 30)).mean()
    elev  = ds.dem_hr[m].mean()
    print(f"  {name:5s}: pixels={m.sum():5d} ({m.mean():.1%}), urban={urban:.1%}, mean_elev={elev:.0f} m")

# %% [markdown]
# ## 4. HR vs LR LST distributions

# %%
def collect_lst(ds, n_max=80):
    hr_vals, lr_vals = [], []
    for i in range(min(len(ds), n_max)):
        s = ds[i]
        m = s['valid_mask'][0].numpy()
        hr_vals.append(s['hr_lst'][0].numpy()[m])
        lr_vals.append(s['lr_lst'][0].numpy().ravel())
    return np.concatenate(hr_vals), np.concatenate(lr_vals)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, ds) in zip(axes, [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]):
    if len(ds) == 0: continue
    hr, lr = collect_lst(ds)
    lo = min(np.percentile(hr, 1), np.percentile(lr, 1))
    hi = max(np.percentile(hr, 99), np.percentile(lr, 99))
    bins = np.linspace(lo, hi, 70)
    ax.hist(hr, bins=bins, alpha=0.6, label=f'HR ({train_ds.hr_res})', color='steelblue', density=True)
    ax.hist(lr, bins=bins, alpha=0.6, label=f'LR ({train_ds.lr_res})', color='coral',     density=True)
    ax.set_title(name); ax.set_xlabel('LST (°C)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('LST distributions: HR vs LR', fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Sample blocks — all channels

# %%
n_show = min(3, len(train_ds))
indices = np.linspace(0, len(train_ds) - 1, n_show, dtype=int) if n_show > 0 else []

panels = [
    ('LR LST',   'lr_lst',     'RdYlBu_r'),
    ('HR LST',   'hr_lst',     'RdYlBu_r'),
    ('NDVI',     'ndvi',       'RdYlGn'),
    ('DEM (m)',  'dem',        'terrain'),
    ('LULC',     'lulc',       'tab20'),
    ('Loc X',    'loc_x',      'viridis'),
    ('Loc Y',    'loc_y',      'viridis'),
    ('valid',    'valid_mask', 'gray'),
]

if n_show > 0:
    fig, axes = plt.subplots(n_show, len(panels), figsize=(2.3 * len(panels), 2.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for row, idx in enumerate(indices):
        s = train_ds[int(idx)]
        for col, (title, key, cmap) in enumerate(panels):
            ax = axes[row, col]
            if key == 'loc_x':
                img = s['loc'][0].float().numpy()
            elif key == 'loc_y':
                img = s['loc'][1].float().numpy()
            else:
                img = s[key].squeeze().float().numpy()
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(title, fontsize=8)
            if col == 0: ax.set_ylabel(f"{s['date']}\nblock {s['block_id']}", fontsize=7)
    plt.suptitle('Train samples — all channels (each row = one date×block)', fontsize=13, y=1.01)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. LULC one-hot encoding

# %%
if len(train_ds) > 0:
    s = train_ds[0]
    oh = s['lulc_onehot'].numpy()
    n_classes = oh.shape[0]
    cols = min(n_classes, 8)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
    axes = np.atleast_2d(axes)
    for c in range(rows * cols):
        ax = axes[c // cols, c % cols]
        if c < n_classes:
            ax.imshow(oh[c], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'cls {int(train_ds.lulc_classes[c])}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'LULC one-hot ({n_classes} channels)', fontsize=14)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. DataLoader batching

# %%
import torch

loaders = get_dataloaders(root=args.root, batch_size=args.batch_size,
                          num_workers=0, download=False)
for name, ld in loaders.items():
    if len(ld.dataset) == 0:
        print(f"{name:5s} — empty"); continue
    b = next(iter(ld))
    print(f"{name:5s}  lr_lst={tuple(b['lr_lst'].shape)}  hr_lst={tuple(b['hr_lst'].shape)}"
          f"  loc={tuple(b['loc'].shape)}  lulc_oh={tuple(b['lulc_onehot'].shape)}")

# %% [markdown]
# ## 8. Multi-resolution LST — 4 km / 1 km / 250 m (same AOI, same projection)

# %%
import rasterio
from model.dataset import read_modis_lst, reproject_to, REF_CRS, RESOLUTION_DEMS
from rasterio.enums import Resampling

def _load_grid(root, res):
    with rasterio.open(os.path.join(root, 'DEM', RESOLUTION_DEMS[res])) as ds:
        return ds.shape, ds.transform

def _extent(shape, tf):
    H, W = shape
    return [tf.c, tf.c + W * tf.a, tf.f + H * tf.e, tf.f]

# Prefer July–August: clearest skies over Colorado, fewest NaNs
summer = [(d, p) for d, p in train_ds.dates if d.month in (7, 8)]
date_pick, modis_path = summer[len(summer) // 2] if summer else train_ds.dates[len(train_ds.dates) // 2]

resolutions = ['4km', '1km', '250m']
resamplings = [Resampling.bilinear, Resampling.bilinear, Resampling.bilinear]

lst_native, lst_tf = read_modis_lst(modis_path)

lst_layers, extents = [], []
for res, rsmp in zip(resolutions, resamplings):
    shape, tf = _load_grid(args.root, res)
    lst = reproject_to(lst_native, lst_tf, REF_CRS, shape, tf, resampling=rsmp)
    lst_layers.append(lst)
    extents.append(_extent(shape, tf))

all_finite = np.concatenate([l[np.isfinite(l)] for l in lst_layers])
vmin, vmax = np.percentile(all_finite, 2), np.percentile(all_finite, 98)
aoi_ext = extents[1]  # 1 km grid defines the AOI extent

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (lst, ext) in zip(axes, zip(lst_layers, extents)):
    im = ax.imshow(lst, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                   extent=ext, origin='upper', aspect='auto')
    ax.set_xlim(aoi_ext[0], aoi_ext[1])
    ax.set_ylim(aoi_ext[2], aoi_ext[3])
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(im, ax=axes[-1], label='LST (°C)', shrink=0.9)
plt.tight_layout()
plt.savefig('lst_multiresolution.pdf', bbox_inches='tight')
plt.show()
print(f'Saved lst_multiresolution.pdf  ({date_pick.strftime("%Y-%m-%d")})')
