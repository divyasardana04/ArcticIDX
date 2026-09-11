"""
Figure 6 for the ArcticIDX : future projections, 1950-2100.

Six panels (GBI, UBI, BHI, SHI, PVI, AAI) from the spliced historical+scenario
series. Each panel shows the multi-model mean for the historical experiment (black)
and for each of the four scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5) in its
own colour, with a shaded 10th-90th percentile inter-model band around every line.
An 11-year centred running mean is applied (per model, before the ensemble
statistics) to suppress interannual noise while keeping the forced signal. A dotted
vertical line at 2014 marks the historical/scenario split.

Aggregation (stated in the panel titles): DJF means for SHI and PVI, annual means for
GBI, UBI, BHI and (anomaly-based) AAI.

CanESM5 has no ssp126 ua for 2015-2100, so it is excluded from the PVI ssp126 curve
only; the reduced sample size is stated in the PVI panel.

"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ARCTICIDX = os.environ.get("ARCTICIDX_ROOT", r"C:\Users\karti\Desktop\arcticidx")

INDICES = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]
AGG = {"GBI": "annual", "UBI": "annual", "BHI": "annual",
       "SHI": "DJF", "PVI": "DJF", "AAI": "annual"}
UNIT = {"GBI": "m", "UBI": "m", "BHI": "hPa", "SHI": "hPa", "PVI": "m s$^{-1}$", "AAI": "K"}
SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SCEN_COL = {"ssp126": "#2E8B57", "ssp245": "#E8C547", "ssp370": "#E8873A", "ssp585": "#C1272D"}
SCEN_LAB = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}

HIST_YEARS = np.arange(1950, 2015)      # historical experiment
FULL_YEARS = np.arange(1950, 2101)      # spliced historical+scenario


def vdict(ds, name, agg):
    """year -> index value (annual mean, or DJF season with Dec -> following year)."""
    lbl = [str(x) for x in ds["time_label"].values]
    val = ds[name].values
    if agg == "annual":
        yrs = [int(s) for s in lbl]
    else:
        sea = np.array([s[5:] for s in lbl])
        yrs = [int(s[:4]) for s in lbl]
        keep = sea == "DJF"
        yrs = [y for y, k in zip(yrs, keep) if k]
        val = val[keep]
    return {int(y): float(v) for y, v in zip(yrs, val)}


def smooth(vals, years):
    return pd.Series(vals, index=years).rolling(11, center=True, min_periods=6).mean().values


def ensemble(dicts, years, drop=None):
    """MMM line + 10th/90th percentile band of the per-model 11-yr-smoothed series."""
    stack = []
    for i, d in enumerate(dicts):
        if drop is not None and i == drop:
            continue
        raw = np.array([d.get(int(y), np.nan) for y in years], float)
        stack.append(smooth(raw, years))
    arr = np.array(stack)
    return (np.nanmean(arr, 0), np.nanpercentile(arr, 10, 0),
            np.nanpercentile(arr, 90, 0), len(stack))


# ------------------------------------------------------------- collect series
models = (pd.read_csv(os.path.join(ARCTICIDX, "models_included.csv"))
          .query("n_ssps == 4")["model"].tolist())
data = {e: {ix: [] for ix in INDICES} for e in ["historical"] + SCENARIOS}

for m in models:
    base = os.path.join(ARCTICIDX, m)
    with xr.open_dataset(os.path.join(base, f"ArcticIDX_{m}_historical_annual.nc")) as ha, \
         xr.open_dataset(os.path.join(base, f"ArcticIDX_{m}_historical_seasonal.nc")) as hs:
        for ix in INDICES:
            data["historical"][ix].append(vdict(ha if AGG[ix] == "annual" else hs, ix, AGG[ix]))
    for ssp in SCENARIOS:
        with xr.open_dataset(os.path.join(base, f"ArcticIDX_{m}_historical-{ssp}_annual.nc")) as a, \
             xr.open_dataset(os.path.join(base, f"ArcticIDX_{m}_historical-{ssp}_seasonal.nc")) as s:
            for ix in INDICES:
                data[ssp][ix].append(vdict(a if AGG[ix] == "annual" else s, ix, AGG[ix]))

CAN = models.index("CanESM5")

# diagnostic: end-of-century (2081-2100) multi-model-mean under ssp585, for the
# old-vs-new comparison after the Arctic-minus-global AAI + areacella fix.
for ix in ("AAI", "GBI", "SHI"):
    per_model = [np.nanmean([d.get(y, np.nan) for y in range(2081, 2101)])
                 for d in data["ssp585"][ix]]
    print(f"[diag] ssp585 2081-2100 MMM {ix} = {np.nanmean(per_model):.3f} {UNIT[ix]}")

# ----------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(3, 2, figsize=(10.4, 12.6))
axes = axes.ravel()
pvi_n126 = None

for i, (ax, ix) in enumerate(zip(axes, INDICES)):
    # historical (black), 1950-2014
    mmm, lo, hi, _ = ensemble(data["historical"][ix], HIST_YEARS)
    ax.fill_between(HIST_YEARS, lo, hi, color="black", alpha=0.12, lw=0, zorder=1)
    ax.plot(HIST_YEARS, mmm, color="black", lw=1.9, zorder=4)
    # scenarios (colour), 2014-2100
    fut = FULL_YEARS >= 2014
    for ssp in SCENARIOS:
        drop = CAN if (ix == "PVI" and ssp == "ssp126") else None
        mmm, lo, hi, n = ensemble(data[ssp][ix], FULL_YEARS, drop=drop)
        ax.fill_between(FULL_YEARS[fut], lo[fut], hi[fut], color=SCEN_COL[ssp], alpha=0.13, lw=0, zorder=2)
        ax.plot(FULL_YEARS[fut], mmm[fut], color=SCEN_COL[ssp], lw=1.7, zorder=3)
        if ix == "PVI" and ssp == "ssp126":
            pvi_n126 = n
    ax.axvline(2014, color="0.5", lw=0.9, ls=":", zorder=0)
    agg_lbl = "Annual" if AGG[ix] == "annual" else AGG[ix]   # keep DJF as-is
    ax.set_title(f"({'abcdef'[i]}) {ix}  ({agg_lbl})", fontweight="bold")
    ax.set_ylabel(f"{ix} ({UNIT[ix]})")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.grid(True, color="0.93", lw=0.5, zorder=0)

for ax in axes[-2:]:
    ax.set_xlabel("Year")

handles = [Line2D([0], [0], color="black", lw=2.0, label="historical")]
handles += [Line2D([0], [0], color=SCEN_COL[s], lw=2.0, label=SCEN_LAB[s]) for s in SCENARIOS]
handles += [Patch(facecolor="0.5", alpha=0.25, label="10th–90th percentile (inter-model)")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, fontsize=9,
           handlelength=1.8, columnspacing=1.5, bbox_to_anchor=(0.5, 0.012))
fig.subplots_adjust(left=0.075, right=0.975, top=0.97, bottom=0.075, hspace=0.24, wspace=0.2)

out = os.path.join(HERE, "fig6_projections")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
print(f"models={len(models)}  PVI ssp126 sample size = {pvi_n126}")
print(f"Saved {out}.png (300 dpi) and {out}.pdf")
