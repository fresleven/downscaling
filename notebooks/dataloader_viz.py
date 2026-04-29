# %% [markdown]
# # Temperature Downscaling — DataLoader Visualization
#
# Visualize the new `DownscalingDataset` (post-2026-04 HF layout):
# inspect the temporal × spatial split, the LULC one-hot scheme, and
# sample LR (4 km) / HR (1 km) pairs with their HR covariates.

# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, '..')

from model.dataset import DownscalingDataset, get_dataloaders

parser = argparse.ArgumentParser(description="Visualize the new downscaling dataloader.")
parser.add_argument('--root', type=str, default='../data',
                    help='Path to data directory (contains MODIS/, NDVI/, DEM/, LULC_final.tiff, aoi/)')
parser.add_argument('--no-download', action='store_true',
                    help='Do not download data from HuggingFace if missing')
parser.add_argument('--batch-size', type=int, default=4)

if ipython is None:
    args = parser.parse_args()
else:
    args = parser.parse_args(args=[])

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## 1. Load Datasets

# %%
download = not args.no_download

train_ds = DownscalingDataset(root=args.root, split='train', download=download)
val_ds   = DownscalingDataset(root=args.root, split='val',   download=download)
test_ds  = DownscalingDataset(root=args.root, split='test',  download=download)

print(f"Train: {len(train_ds)} scenes")
print(f"Val:   {len(val_ds)} scenes")
print(f"Test:  {len(test_ds)} scenes")
print(f"\nHR shape: {train_ds.hr_shape}, LR shape: {train_ds.lr_shape}")
print(f"LULC classes (one-hot dims): {train_ds.lulc_classes.tolist()}")
print(f"  → lulc_onehot has {len(train_ds.lulc_classes)} channels")

sample = train_ds[0]
print(f"\nSample 0 keys + shapes:")
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k:12s} {tuple(v.shape)}  {v.dtype}")
    else:
        print(f"  {k:12s} {v}")

# %% [markdown]
# ## 2. Temporal split — scenes per month

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, ds) in zip(axes, [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]):
    if len(ds) == 0:
        ax.set_title(f'{name} — 0 scenes')
        continue
    months = [d.strftime('%Y-%m') for d, _ in ds.dates]
    counts = Counter(months)
    keys = sorted(counts.keys())
    ax.bar(range(len(keys)), [counts[k] for k in keys],
           color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{name} — {len(ds)} scenes')
    ax.set_ylabel('Scenes')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Scenes per month by split (after valid-fraction filter)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Spatial split — block assignment over the AOI
#
# Each HR pixel is assigned to train/val/test by the block it lies in.
# Stratified by (urban-frac × elevation) so every split spans both.

# %%
import matplotlib.colors as mcolors

label_arr = np.full(train_ds.hr_shape, -1, dtype=np.int8)
for b, s in train_ds.block_assignment.items():
    label_arr[train_ds.block_id == b] = {'train': 0, 'val': 1, 'test': 2}[s]

cmap = mcolors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(label_arr, cmap=cmap, vmin=0, vmax=2)
axes[0].set_title('Block assignment (blue=train, orange=val, green=test)')
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(train_ds.dem_hr, cmap='terrain')
axes[1].set_title('DEM (m)')
axes[1].set_xticks([]); axes[1].set_yticks([])

axes[2].imshow(train_ds.lulc_hr, cmap='tab20')
axes[2].set_title('LULC (NLCD class)')
axes[2].set_xticks([]); axes[2].set_yticks([])

plt.suptitle('Spatial split with terrain & land cover', fontsize=14)
plt.tight_layout()
plt.show()

# Per-split summary
print("Per-split spatial composition:")
for ds, name in [(train_ds, 'train'), (val_ds, 'val'), (test_ds, 'test')]:
    m = ds.spatial_mask
    urban = ((ds.lulc_hr[m] >= 21) & (ds.lulc_hr[m] < 30)).mean()
    elev = ds.dem_hr[m].mean()
    print(f"  {name:5s}: pixels={m.sum():5d} ({m.mean():.1%}), urban={urban:.1%}, mean_elev={elev:.0f} m")

# %% [markdown]
# ## 4. HR vs LR LST distributions

# %%
def collect_lst(ds, n_max=80):
    """Gather LST values from the first n_max samples, mask-restricted."""
    hr_vals, lr_vals = [], []
    for i in range(min(len(ds), n_max)):
        s = ds[i]
        m = s['valid_mask'][0].numpy()
        hr_vals.append(s['hr_lst'][0].numpy()[m])
        lr_vals.append(s['lr_lst'][0].numpy().ravel())
    return np.concatenate(hr_vals), np.concatenate(lr_vals)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, ds) in zip(axes, [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]):
    if len(ds) == 0:
        continue
    hr, lr = collect_lst(ds)
    lo = min(np.percentile(hr, 1), np.percentile(lr, 1))
    hi = max(np.percentile(hr, 99), np.percentile(lr, 99))
    bins = np.linspace(lo, hi, 70)
    ax.hist(hr, bins=bins, alpha=0.6, label='HR (1 km)', color='steelblue', density=True)
    ax.hist(lr, bins=bins, alpha=0.6, label='LR (4 km)', color='coral', density=True)
    ax.set_title(name)
    ax.set_xlabel('LST (°C)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('LST distributions: HR (1 km) vs LR (4 km)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Sample scenes — all channels

# %%
n_show = min(3, len(train_ds))
indices = np.linspace(0, len(train_ds) - 1, n_show, dtype=int) if n_show > 0 else []

panels = [
    ('LR LST (4 km)', 'lr_lst', 'RdYlBu_r'),
    ('HR LST (1 km)', 'hr_lst', 'RdYlBu_r'),
    ('NDVI',          'ndvi',   'RdYlGn'),
    ('DEM (m)',       'dem',    'terrain'),
    ('LULC',          'lulc',   'tab20'),
    ('valid mask',    'valid_mask', 'gray'),
]

if n_show > 0:
    fig, axes = plt.subplots(n_show, len(panels), figsize=(2.5 * len(panels), 2.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for row, idx in enumerate(indices):
        s = train_ds[int(idx)]
        for col, (title, key, cmap) in enumerate(panels):
            ax = axes[row, col]
            img = s[key].squeeze().float().numpy()
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(s['date'], fontsize=8)
    plt.suptitle('Train scenes — all channels', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. LULC one-hot encoding

# %%
if len(train_ds) > 0:
    s = train_ds[0]
    oh = s['lulc_onehot'].numpy()  # (N, H, W)
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
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. DataLoader batching

# %%
import torch

loaders = get_dataloaders(
    root=args.root,
    batch_size=args.batch_size,
    num_workers=0,
    download=False,
)

for name, ld in loaders.items():
    if len(ld.dataset) == 0:
        print(f"{name:5s} — empty")
        continue
    b = next(iter(ld))
    print(f"{name:5s} — lr_lst={tuple(b['lr_lst'].shape)} hr_lst={tuple(b['hr_lst'].shape)}, "
          f"ndvi={tuple(b['ndvi'].shape)}, lulc_onehot={tuple(b['lulc_onehot'].shape)}, "
          f"valid={tuple(b['valid_mask'].shape)}")
