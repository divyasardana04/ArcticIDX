"""
Figure 3 for the ArcticIDX 

Standardised mean-bias heatmap: 30 CMIP6 models (rows) x 6 indices (columns).
Each cell is the model's 1981-2010 mean minus ERA5's 1981-2010 mean, divided by
ERA5's interannual standard deviation for that index over the same period, so the
six columns (different physical units) are directly comparable in "ERA5 sigma".

Column order: GBI, UBI, BHI, SHI, PVI (the paper's usual order), then AAI on the far
right as the special case (see below). Temporal aggregation (stated in the labels):
  * DJF means for PVI and SHI (winter phenomena),
  * annual means for GBI, UBI, BHI, AAI.
The colour scale is clipped at +/-3 ERA5-sigma so the AAI column (whose ERA5
interannual SD is very small) no longer compresses the other five; the annotated
numbers are always the true standardised bias, so extreme cells stay readable.

AAI column: the AAI index is an anomaly relative to each dataset's own 1981-2010
climatology, so its 1981-2010 MEAN bias is zero by construction for every model.
The column therefore uses the physical quantity AAI is built on -- the area-weighted
Arctic (>70N) minus GLOBAL-mean 2 m temperature contrast, T_arctic - T_global --
whose mean bias is meaningful. (Noted on the figure.)

Rows are sorted by mean absolute standardised bias (best agreement at the top); a
separated bottom row shows the multi-model mean bias per index. Positive (red) =
model higher than ERA5. Uses the HISTORICAL experiment only, so the CanESM5 ssp126
PVI gap does not affect this figure.

"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARCTICIDX = r"C:\Users\karti\Desktop\arcticidx"
Y0, Y1 = 1981, 2010

# (index, aggregation) in column order -- paper's usual order, AAI last (far right)
COLS = [("GBI", "annual"), ("UBI", "annual"), ("BHI", "annual"),
        ("SHI", "DJF"), ("PVI", "DJF"), ("AAI", "annual")]
# display names (data key stays "AAI"); AAI shown as its physical quantity, matching fig2
DISPLAY = {"AAI": r"$T_{\mathrm{Arctic}}-T_{\mathrm{global}}$"}
COLLABELS = [f"{DISPLAY.get(n, n)}\n({agg})" for n, agg in COLS]


def _var(ds: xr.Dataset, name: str) -> np.ndarray:
    # AAI mean bias is 0 by construction -> use the Arctic-minus-global T contrast instead
    if name == "AAI":
        return ds["tas_arctic"].values - ds["tas_global"].values
    return ds[name].values


def series(annual_ds, seasonal_ds, name, agg) -> np.ndarray:
    """1981-2010 yearly values: annual means, or DJF means (Dec -> following year)."""
    if agg == "annual":
        yrs = np.array([int(str(x)) for x in annual_ds["time_label"].values])
        val = _var(annual_ds, name)
    else:
        lbl = [str(x) for x in seasonal_ds["time_label"].values]
        yrs = np.array([int(s[:4]) for s in lbl])
        sea = np.array([s[5:] for s in lbl])
        val = _var(seasonal_ds, name)
        keep = sea == "DJF"
        yrs, val = yrs[keep], val[keep]
    m = (yrs >= Y0) & (yrs <= Y1)
    return val[m]


def open_pair(folder, stem):
    a = xr.open_dataset(os.path.join(folder, f"{stem}_annual.nc"))
    s = xr.open_dataset(os.path.join(folder, f"{stem}_seasonal.nc"))
    return a, s


# ------------------------------------------------------------- ERA5 reference
era5_a, era5_s = open_pair(os.path.join(ARCTICIDX, "ERA5"), "ArcticIDX_ERA5_reanalysis")
ref_mean, ref_std = {}, {}
for name, agg in COLS:
    v = series(era5_a, era5_s, name, agg)
    ref_mean[(name, agg)] = v.mean()
    ref_std[(name, agg)] = v.std(ddof=1)          # interannual std
era5_a.close(); era5_s.close()

# --------------------------------------------------------- per-model biases
models = (pd.read_csv(os.path.join(ARCTICIDX, "models_included.csv"))
          .query("n_ssps == 4")["model"].tolist())
matrix = np.full((len(models), len(COLS)), np.nan)
for i, m in enumerate(models):
    a, s = open_pair(os.path.join(ARCTICIDX, m), f"ArcticIDX_{m}_historical")
    for j, (name, agg) in enumerate(COLS):
        v = series(a, s, name, agg)
        matrix[i, j] = (v.mean() - ref_mean[(name, agg)]) / ref_std[(name, agg)]
    a.close(); s.close()

# sort rows by mean |standardised bias| (best agreement on top); MMM row at bottom
mean_abs = np.abs(matrix).mean(axis=1)
order = np.argsort(mean_abs)
matrix_s = matrix[order]
rows = [models[k] for k in order]
mmm = matrix.mean(axis=0)
full = np.vstack([matrix_s, mmm])
rows = rows + ["MME"]
n = len(rows)

print(f"{len(models)} models. matrix |value|: max={np.nanmax(np.abs(matrix)):.2f}  "
      f"95pct={np.nanpercentile(np.abs(matrix),95):.2f}")
print("per-column RMS standardised bias: "
      + ", ".join(f"{c[0]}({c[1]})={np.sqrt(np.nanmean(matrix[:,j]**2)):.2f}"
                  for j, c in enumerate(COLS)))

# --------------------------------------------------------------------- figure
# Colour scale clipped at +/-3 ERA5-sigma so the AAI contrast column (very small ERA5
# interannual SD -> large standardised bias) no longer compresses the other five.
# Extreme cells saturate (colorbar 'extend' arrows); annotations keep the true values.
vlim = 3.0
fig, ax = plt.subplots(figsize=(8.4, 12.4))
im = ax.imshow(full, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)

ax.set_xticks(np.arange(len(COLS)))
ax.set_xticklabels(COLLABELS, fontsize=10, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(rows, fontsize=8)
ax.get_yticklabels()[-1].set_fontweight("bold")     # highlight the MMM row label

# white cell separators
ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)

# separator line above the multi-model-mean row
ax.axhline(n - 1.5, color="black", linewidth=1.8)

# annotate every cell to one decimal
for i in range(n):
    for j in range(len(COLS)):
        val = full[i, j]
        txt = "black" if abs(val) < 0.62 * vlim else "white"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=7.2, color=txt)

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="both")
cbar.set_label("Standardised bias", fontsize=9.5)

fig.subplots_adjust(left=0.20, right=0.99, top=0.94, bottom=0.04)
out = os.path.join(HERE, "fig3_bias_heatmap")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}.png (300 dpi) and {out}.pdf")
