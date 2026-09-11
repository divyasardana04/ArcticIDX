"""
Figure 2 for the ArcticIDX Scientific Data paper.

Seasonal-cycle climatology (1981-2010, historical experiment) of the six Arctic
atmospheric indices, one panel each: 30 CMIP6 models as thin grey lines, the
multi-model mean as a thick coloured line, and ERA5 reanalysis as a thick black
line. x-axis January -> December.

IMPORTANT -- the AAI panel:
  The AAI index is defined as an anomaly relative to the 1981-2010 monthly
  climatology, so its OWN 1981-2010 seasonal-cycle climatology is identically zero
  (verified: 0.0 for every month, all models and ERA5). A flat-zero panel carries
  no information, so this panel instead shows the physical quantity the index is
  built on: the area-weighted Arctic (poleward of 70N) minus GLOBAL-mean 2 m air
  temperature contrast, T_arctic - T_global (K), whose seasonal cycle is meaningful
  (Arctic coldest relative to the globe in winter). The y-axis is labelled
  accordingly. All other panels show the index itself.

"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ARCTICIDX = r"C:\Users\karti\Desktop\arcticidx"
BASE = (1981, 2010)

PANELS = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]          
UNITS = {"AAI": "K", "BHI": "hPa", "SHI": "hPa",
         "GBI": "m", "UBI": "m", "PVI": "m s$^{-1}$"}
TITLE = {
    "AAI": "Arctic minus global temperature",
    "BHI": "Beaufort High Index",
    "SHI": "Siberian High Index",
    "GBI": "Greenland Blocking Index",
    "UBI": "Ural Blocking Index",
    "PVI": "Polar Vortex Index",
}
YLABEL = {k: f"{k} ({UNITS[k]})" for k in PANELS}
YLABEL["AAI"] = r"$T_{\mathrm{Arctic}}-T_{\mathrm{global}}$ (K)"  

MMM_COL = "#D55E00"     # muti-model mean
MODEL_COL = "0.72"      # individual models -- thin grey
ERA5_COL = "black"      # ERA5 -- thick black

MONTHS = np.arange(1, 13)
MLAB = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def monthly_clim(ds: xr.Dataset, panel: str) -> np.ndarray:
    """1981-2010 per-calendar-month climatology (12 values, Jan..Dec). For AAI the
    quantity is the Arctic-minus-rest 2 m temperature contrast (see module header)."""
    da = (ds["tas_arctic"] - ds["tas_global"]) if panel == "AAI" else ds[panel]
    yr = ds["time"].dt.year
    c = (da.where((yr >= BASE[0]) & (yr <= BASE[1]), drop=True)
           .groupby("time.month").mean("time").sortby("month"))
    return c.values


# ---------------------------------------------------------------- load the data
models = (pd.read_csv(os.path.join(ARCTICIDX, "models_included.csv"))
          .query("n_ssps == 4")["model"].tolist())

model_clim = {p: [] for p in PANELS}
for m in models:
    with xr.open_dataset(os.path.join(ARCTICIDX, m, f"ArcticIDX_{m}_historical.nc")) as ds:
        for p in PANELS:
            model_clim[p].append(monthly_clim(ds, p))
model_clim = {p: np.array(v) for p, v in model_clim.items()}          # (n_models, 12)

with xr.open_dataset(os.path.join(ARCTICIDX, "ERA5", "ArcticIDX_ERA5_reanalysis.nc")) as eds:
    era5_clim = {p: monthly_clim(eds, p) for p in PANELS}
mmm = {p: model_clim[p].mean(axis=0) for p in PANELS}
print(f"loaded {len(models)} models; per-panel array {model_clim['GBI'].shape}")

# ----------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 11.5, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.8), sharex=True)
axes = axes.ravel()

for i, (ax, p) in enumerate(zip(axes, PANELS)):
    for row in model_clim[p]:
        ax.plot(MONTHS, row, color=MODEL_COL, lw=0.6, alpha=0.75, zorder=1)
    ax.plot(MONTHS, mmm[p], color=MMM_COL, lw=2.4, zorder=3)
    ax.plot(MONTHS, era5_clim[p], color=ERA5_COL, lw=2.4, zorder=4)
    ax.set_title(f"({'abcdef'[i]}) {p}", fontweight="bold")   # e.g. "(c) BHI"
    ax.set_ylabel(YLABEL[p])
    ax.set_xlim(1, 12)
    ax.set_xticks(MONTHS)
    ax.grid(True, color="0.9", lw=0.6, zorder=0)
    if i >= 3:                                   # bottom row (SHI, PVI, AAI) -> month labels
        ax.set_xticklabels(MLAB)
        ax.set_xlabel("Month")
    else:
        ax.tick_params(labelbottom=False)

handles = [
    Line2D([0], [0], color=MODEL_COL, lw=1.1, label=f"CMIP6 models (n = {len(models)})"),
    Line2D([0], [0], color=MMM_COL, lw=2.6, label="Multi-model mean"),
    Line2D([0], [0], color=ERA5_COL, lw=2.6, label="ERA5 reanalysis"),
]
fig.subplots_adjust(left=0.075, right=0.98, top=0.94, bottom=0.135,
                    hspace=0.28, wspace=0.30)
leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
                 bbox_to_anchor=(0.5, 0.02), fontsize=9.5, handlelength=2.6,
                 columnspacing=2.2, borderpad=0.7)
leg.get_frame().set_edgecolor("0.6")
leg.get_frame().set_linewidth(0.7)

out = os.path.join(HERE, "fig2_seasonal_cycle")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}.png (300 dpi) and {out}.pdf")
