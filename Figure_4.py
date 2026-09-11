"""
Figure 4 for the ArcticIDX Scientific Data paper. Two panels.

(a) Interannual variability-AMPLITUDE check. For each index, the detrended interannual
    standard deviation over 1979-2014 is computed for each of the 30 CMIP6 models
    individually and for ERA5, and shown as one box-and-whisker per index (models),
    each model's SD expressed as a RATIO to the ERA5 SD so the six indices share one
    axis. A dashed line at 1.0 marks perfect agreement; the ERA5 reference (= 1.0 by
    definition) is drawn as a black star. (Correlating the multi-model mean against
    ERA5 is NOT meaningful: each model runs its own, unsynchronised internal
    variability, so averaging 30 members cancels it -- hence this amplitude ratio,
    which is a fair per-model comparison, is used instead.)

(b) Independent code verification. NOAA PSL's published daily Greenland Blocking
    Index (https://psl.noaa.gov/data/correlation/gbi.ncep.day) is aggregated to
    monthly means and scattered against our ERA5-derived GBI over the common period,
    with Pearson r and the OLS regression slope. NOAA's GBI is computed from
    NCEP/NCAR reanalysis 500 hPa heights by an INDEPENDENT implementation, so a high
    correlation validates the ArcticIDX computational framework, not the input data.

"""
import os
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ARCTICIDX = r"C:\Users\karti\Desktop\arcticidx"
NOAA_URL = "https://psl.noaa.gov/data/correlation/gbi.ncep.day"
NOAA_LOCAL = os.path.join(HERE, "gbi.ncep.day")

INDICES = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]     # paper's usual order
AGG = {"GBI": "annual", "UBI": "annual", "BHI": "annual",
       "SHI": "DJF", "PVI": "DJF", "AAI": "annual"}       # DJF for SHI/PVI (match fig3/fig5)
Y0, Y1 = 1979, 2014
YEARS = np.arange(Y0, Y1 + 1)

# box-plot style (matches the supplementary AAI-trend figure)
BOX_FACE, MEDIAN_COL, MODEL_PT, ERA5_COL = "#e3ddf0", "#5b3a8e", "0.55", "black"


def detrend(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y))
    return y - np.polyval(np.polyfit(x, y, 1), x)


def pick_series(a_ds, s_ds, name):
    """Values aligned to YEARS (1979-2014): annual mean, or DJF mean (Dec -> following
    year) for SHI/PVI, matching the aggregation used in fig3 and fig5."""
    if AGG[name] == "annual":
        yrs = np.array([int(str(x)) for x in a_ds["time_label"].values])
        vals = a_ds[name].values
    else:
        lbl = [str(x) for x in s_ds["time_label"].values]
        yrs = np.array([int(x[:4]) for x in lbl]); sea = np.array([x[5:] for x in lbl])
        keep = sea == "DJF"; yrs, vals = yrs[keep], s_ds[name].values[keep]
    d = dict(zip(yrs, vals))
    return np.array([d[y] for y in YEARS])


# ============================================================ panel (a) data
era5_a = xr.open_dataset(os.path.join(ARCTICIDX, "ERA5",
                                      "ArcticIDX_ERA5_reanalysis_annual.nc"))
era5_s = xr.open_dataset(os.path.join(ARCTICIDX, "ERA5",
                                      "ArcticIDX_ERA5_reanalysis_seasonal.nc"))
models = (pd.read_csv(os.path.join(ARCTICIDX, "models_included.csv"))
          .query("n_ssps == 4")["model"].tolist())
model_a = {m: xr.open_dataset(os.path.join(ARCTICIDX, m,
                              f"ArcticIDX_{m}_historical_annual.nc")) for m in models}
model_s = {m: xr.open_dataset(os.path.join(ARCTICIDX, m,
                              f"ArcticIDX_{m}_historical_seasonal.nc")) for m in models}

# detrended interannual SD per index (DJF for SHI/PVI, else annual): model / ERA5 ratio
era5_sd = {ix: detrend(pick_series(era5_a, era5_s, ix)).std(ddof=1) for ix in INDICES}
sd_ratio = {ix: np.array([detrend(pick_series(model_a[m], model_s[m], ix)).std(ddof=1)
                          for m in models]) / era5_sd[ix] for ix in INDICES}
print("(a) median SD ratio (model/ERA5): "
      + ", ".join(f"{ix}={np.median(sd_ratio[ix]):.2f}" for ix in INDICES))

# ============================================================ panel (b) data
if not os.path.exists(NOAA_LOCAL):
    print("downloading NOAA GBI ...")
    urllib.request.urlretrieve(NOAA_URL, NOAA_LOCAL)

noaa = pd.DataFrame(np.loadtxt(NOAA_LOCAL), columns=["year", "month", "day", "gbi"])
noaa = noaa[(noaa.gbi > 4000) & (noaa.gbi < 6000)]                 # drop any flagged
noaa_m = noaa.groupby(["year", "month"])["gbi"].mean()            # daily -> monthly

era5_mon = xr.open_dataset(os.path.join(ARCTICIDX, "ERA5", "ArcticIDX_ERA5_reanalysis.nc"))
ey = era5_mon["time"].dt.year.values
em = era5_mon["time"].dt.month.values
eg = era5_mon["GBI"].values
xs, ys = [], []                                                   # x=ours, y=NOAA
for y, m, g in zip(ey, em, eg):
    if (y, m) in noaa_m.index:
        xs.append(float(g)); ys.append(float(noaa_m.loc[(y, m)]))
xs, ys = np.array(xs), np.array(ys)
rb = float(np.corrcoef(xs, ys)[0, 1])
slope, intercept = np.polyfit(xs, ys, 1)
yr_lo, yr_hi = int(min(ey)), int(max(ey))
print(f"(b) n={len(xs)} months  r={rb:.3f}  slope={slope:.3f}  ({yr_lo}-{yr_hi})")

# ==================================================================== figure
fig = plt.figure(figsize=(10.6, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5],
                      left=0.075, right=0.985, top=0.955, bottom=0.07,
                      hspace=0.32)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[1, 0])            # square + centred (adjustable=box) -> white space on either side

# ---- panel (a): SD-ratio box plot ----
pos = np.arange(1, len(INDICES) + 1)
axa.axhline(1.0, color="0.5", lw=1.1, ls="--", zorder=1)          # perfect agreement
bp = axa.boxplot([sd_ratio[ix] for ix in INDICES], positions=pos, widths=0.55,
                 patch_artist=True, showfliers=False, zorder=2,
                 medianprops=dict(color=MEDIAN_COL, lw=2.0),
                 boxprops=dict(facecolor=BOX_FACE, edgecolor="0.4", lw=1.0),
                 whiskerprops=dict(color="0.4", lw=1.0),
                 capprops=dict(color="0.4", lw=1.0))
rng = np.random.default_rng(0)
for x, ix in zip(pos, INDICES):
    axa.scatter(x + rng.uniform(-0.14, 0.14, size=len(sd_ratio[ix])), sd_ratio[ix],
                s=14, color=MODEL_PT, alpha=0.55, edgecolors="none", zorder=3)
axa.scatter(pos, np.ones_like(pos, dtype=float), marker="*", s=250, color=ERA5_COL,
            edgecolors="white", linewidths=1.0, zorder=5)
axa.set_xticks(pos)
axa.set_xticklabels([ix if AGG[ix] == "annual" else f"{ix}\n(DJF)" for ix in INDICES])
axa.set_ylabel("Detrended interannual SD ratio\n(Model / ERA5)", fontsize=11.5)
axa.set_xlabel("Index", fontsize=11.5, labelpad=2)
axa.xaxis.set_label_coords(0.5, -0.17)   # pin directly under panel (a)'s tick labels
axa.set_ylim(bottom=0)
axa.text(0.008, 0.955, "(a)", transform=axa.transAxes, fontsize=13,
         fontweight="bold", va="top", ha="left")
axa.grid(True, axis="y", color="0.92", lw=0.6, zorder=0)
axa.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PT, markersize=6, alpha=0.7, label="individual models"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=ERA5_COL, markeredgecolor="white", markersize=15, label="ERA5 (= 1.0 by definition)"),
], loc="lower right", frameon=True, fontsize=8.5, handlelength=1.6, borderpad=0.6)

# ---- panel (b): NOAA verification scatter (UNCHANGED) ----
axb.scatter(xs, ys, s=9, color="#4472A4", alpha=0.45, edgecolors="none", zorder=2)
lo = min(xs.min(), ys.min()); hi = max(xs.max(), ys.max())
axb.plot([lo, hi], [lo, hi], color="0.6", lw=1.0, ls="--", zorder=1, label="1:1")
xx = np.array([xs.min(), xs.max()])
axb.plot(xx, slope * xx + intercept, color="#C1272D", lw=1.8, zorder=3,
         label=f"OLS fit (slope = {slope:.3f})")
axb.set_xlabel("ERA5-derived GBI (m)", fontsize=11.5)
axb.set_ylabel("NOAA PSL GBI (m)", fontsize=11.5)
axb.text(0.03, 0.975, "(b)", transform=axb.transAxes, fontsize=13,
         fontweight="bold", va="top", ha="left")
axb.text(0.03, 0.88,
         f"Pearson r = {rb:.3f}\nslope = {slope:.3f}\nn = {len(xs)} months",
         transform=axb.transAxes, va="top", ha="left", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.7))
axb.legend(loc="lower right", frameon=True, fontsize=9)
axb.grid(True, color="0.92", lw=0.6, zorder=0)
axb.set_aspect("equal", adjustable="box")   # keep square; centre it -> white space on either side

out = os.path.join(HERE, "fig4_variability_verification")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}.png (300 dpi) and {out}.pdf")
