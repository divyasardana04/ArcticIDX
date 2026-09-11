"""
Figure 5 for the ArcticIDX , historical trends, models vs ERA5.

For each of the six indices the OLS linear trend over 1979-2014 (historical) is
computed for each of the 30 CMIP6 models and for ERA5. Each index has its own panel
(2x3 grid: GBI, UBI, BHI, SHI, PVI, AAI) with its own y-axis in native units per
decade: a box-and-whisker of the 30 model trends, individual models as jittered grey
dots, and the ERA5 trend as a star with a 95% confidence-interval error bar. The
ERA5 star is FILLED when its trend is significant (p < 0.05) and OPEN (black edge)
when it is not -- so a non-significant ERA5 trend (e.g. the very noisy DJF PVI) is
not mistaken for a robust observed value. A dashed line at zero is drawn in every
panel. Aggregation: DJF for SHI and PVI, annual for GBI, UBI, BHI and (anomaly-based)
AAI.

Prints two tables: (i) MMM trend, +/-1 sigma spread, ERA5 trend and fraction of
models below ERA5; (ii) the ERA5 trend with its two-sided p-value and 95% CI, so the
robust observed trends are visible. Regression statistics use an exact Student-t
p-value (incomplete beta function) -- no scipy dependency.

"""
import os
import math

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
Y0, Y1 = 1979, 2014

INDICES = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]
AGG = {"GBI": "annual", "UBI": "annual", "BHI": "annual",
       "SHI": "DJF", "PVI": "DJF", "AAI": "annual"}
UNIT = {"GBI": "m", "UBI": "m", "BHI": "hPa", "SHI": "hPa",
        "PVI": "m s$^{-1}$", "AAI": "K"}
UNIT_TXT = {"GBI": "m", "UBI": "m", "BHI": "hPa", "SHI": "hPa", "PVI": "m/s", "AAI": "K"}
BOX_FACE, MEDIAN_COL, MODEL_PT, ERA5_COL = "#dfe7f0", "#1f4e79", "0.55", "black"


# -------- exact OLS trend statistics without scipy (Student-t via incomplete beta) --
def _betacf(a, b, x):
    FPMIN, EPS = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (FPMIN if abs(d) < FPMIN else d); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / (FPMIN if abs(1 + aa * d) < FPMIN else 1 + aa * d)
        c = FPMIN if abs(1 + aa / c) < FPMIN else 1 + aa / c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / (FPMIN if abs(1 + aa * d) < FPMIN else 1 + aa * d)
        c = FPMIN if abs(1 + aa / c) < FPMIN else 1 + aa / c
        de = d * c; h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def _t_crit(df, p=0.975):
    lo, hi, tgt = 0.0, 100.0, 2 * (1 - p)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _betai(df / 2.0, 0.5, df / (df + mid * mid)) > tgt else (lo, mid)
    return 0.5 * (lo + hi)


def ols_trend(years, vals):
    """Return (slope/decade, ci_lo/decade, ci_hi/decade, two-sided p)."""
    x, y = np.asarray(years, float), np.asarray(vals, float)
    n = len(x); xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - y.mean())).sum() / sxx
    resid = y - (slope * x + (y.mean() - slope * xm))
    df = n - 2
    se = math.sqrt((resid ** 2).sum() / df / sxx)
    t = slope / se
    tc = _t_crit(df)
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return slope * 10, (slope - tc * se) * 10, (slope + tc * se) * 10, p


def series(a_ds, s_ds, name, agg):
    if agg == "annual":
        yrs = np.array([int(str(x)) for x in a_ds["time_label"].values]); v = a_ds[name].values
    else:
        lbl = [str(x) for x in s_ds["time_label"].values]
        yrs = np.array([int(x[:4]) for x in lbl]); sea = np.array([x[5:] for x in lbl])
        v = s_ds[name].values; keep = sea == "DJF"; yrs, v = yrs[keep], v[keep]
    m = (yrs >= Y0) & (yrs <= Y1)
    return yrs[m], v[m]


def open_pair(folder, stem):
    return (xr.open_dataset(os.path.join(folder, f"{stem}_annual.nc")),
            xr.open_dataset(os.path.join(folder, f"{stem}_seasonal.nc")))


# --------------------------------------------------------------- gather trends
era5_a, era5_s = open_pair(os.path.join(ARCTICIDX, "ERA5"), "ArcticIDX_ERA5_reanalysis")
era5 = {ix: ols_trend(*series(era5_a, era5_s, ix, AGG[ix])) for ix in INDICES}   # (slope, lo, hi, p)

models = (pd.read_csv(os.path.join(ARCTICIDX, "models_included.csv"))
          .query("n_ssps == 4")["model"].tolist())
model_tr = {ix: [] for ix in INDICES}
for m in models:
    a, s = open_pair(os.path.join(ARCTICIDX, m), f"ArcticIDX_{m}_historical")
    for ix in INDICES:
        model_tr[ix].append(ols_trend(*series(a, s, ix, AGG[ix]))[0])
    a.close(); s.close()
model_tr = {ix: np.array(v) for ix, v in model_tr.items()}

# --------------------------------------------------------------- tables (print)
print(f"\nHistorical trends 1979-2014 (per decade), CMIP6 (n={len(models)}) vs ERA5\n" + "=" * 84)
print(f"{'Index':<6}{'Agg':<8}{'MMM trend':>12}{'model spread':>15}{'ERA5 trend':>12}{'frac<ERA5':>11}  unit")
print("-" * 84)
for ix in INDICES:
    mt, e = model_tr[ix], era5[ix][0]
    print(f"{ix:<6}{AGG[ix]:<8}{mt.mean():>12.3f}{('+/-' + format(mt.std(ddof=1), '.3f')):>15}"
          f"{e:>12.3f}{np.mean(mt < e):>11.2f}  {UNIT_TXT[ix]}/dec")
print("=" * 84)

print("\nERA5 trend significance (1979-2014):\n" + "=" * 72)
print(f"{'Index':<6}{'Agg':<8}{'ERA5 trend':>12}{'95% CI':>22}{'p':>9}   robust?")
print("-" * 72)
for ix in INDICES:
    e, lo, hi, p = era5[ix]
    print(f"{ix:<6}{AGG[ix]:<8}{e:>12.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:>9.3f}   "
          f"{'YES (p<0.05)' if p < 0.05 else 'no'}   {UNIT_TXT[ix]}/dec")
print("=" * 72)

# ----------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 3, figsize=(11.0, 8.0))
axes = axes.ravel()
rng = np.random.default_rng(0)

for i, (ax, ix) in enumerate(zip(axes, INDICES)):
    mt = model_tr[ix]
    e_slope, e_lo, e_hi, e_p = era5[ix]
    ax.axhline(0.0, color="0.6", lw=0.9, ls="--", zorder=1)
    ax.boxplot([mt], positions=[1], widths=0.5, patch_artist=True, showfliers=False,
               zorder=2, medianprops=dict(color=MEDIAN_COL, lw=2.0),
               boxprops=dict(facecolor=BOX_FACE, edgecolor="0.4", lw=1.0),
               whiskerprops=dict(color="0.4", lw=1.0), capprops=dict(color="0.4", lw=1.0))
    ax.scatter(1 + rng.uniform(-0.15, 0.15, size=len(mt)), mt, s=14, color=MODEL_PT,
               alpha=0.55, edgecolors="none", zorder=3)
    # ERA5: 95% CI error bar + filled (significant) / open (not) star
    ax.errorbar(1, e_slope, yerr=[[e_slope - e_lo], [e_hi - e_slope]], fmt="none",
                ecolor="black", elinewidth=1.3, capsize=4, zorder=4)
    if e_p < 0.05:
        ax.scatter(1, e_slope, marker="*", s=270, color=ERA5_COL, edgecolors="white",
                   linewidths=1.0, zorder=5)
    else:
        ax.scatter(1, e_slope, marker="*", s=270, facecolors="none", edgecolors=ERA5_COL,
                   linewidths=1.5, zorder=5)
    agg_lbl = "Annual" if AGG[ix] == "annual" else AGG[ix]   # keep DJF as-is
    ax.set_title(f"({'abcdef'[i]}) {ix}  ({agg_lbl})", fontweight="bold")
    ax.set_ylabel(f"Trend ({UNIT[ix]} decade$^{{-1}}$)")
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", color="0.92", lw=0.6, zorder=0)

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PT, markersize=6, alpha=0.7, label="individual models"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=ERA5_COL, markeredgecolor="white", markersize=15, label="ERA5 trend, significant (p < 0.05)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="none", markeredgecolor=ERA5_COL, markersize=15, markeredgewidth=1.4, label="ERA5 trend, not significant"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fontsize=8.6,
           handlelength=1.6, columnspacing=1.5, bbox_to_anchor=(0.5, 0.02))
fig.subplots_adjust(left=0.07, right=0.975, top=0.965, bottom=0.10,
                    hspace=0.20, wspace=0.30)

out = os.path.join(HERE, "fig5_historical_trends")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}.png (300 dpi) and {out}.pdf")
