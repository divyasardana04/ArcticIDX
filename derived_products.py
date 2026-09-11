"""ArcticIDX derived products — seasonal and annual aggregates.

Reads ONLY the monthly NetCDF files already produced by pilot_one_model.py
(local files, no cloud access) and aggregates them. It NEVER recomputes an index
from raw CMIP6 data — every value here is a mean of already-computed monthly
index values.

Products, per series:
  * seasonal means for DJF / MAM / JJA / SON, where December belongs to the
    FOLLOWING year's DJF; only COMPLETE 3-month seasons are kept (incomplete
    ones are dropped and reported);
  * annual means = calendar-year mean, requiring all 12 months present.

Series:
  * each experiment on its own (historical, ssp126, ssp245, ssp370, ssp585);
  * each historical+SSP spliced into one 1950-2100 monthly timeline first,
    named "historical-ssp585" etc.

Each product carries a REAL CF datetime 'time' coordinate (annual -> YYYY-07-01,
season -> midpoint of its 3-month span), a matching 'time_bnds' spanning the
aggregation period, a companion string coordinate 'time_label' ("1951-DJF",
"1950") for readability, and cell_methods="time: mean" on every index variable.
NetCDF + CSV are written per (series, frequency) with enrich_metadata applied;
the CSVs are indexed by the readable time_label.

    python derived_products.py
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr

import pilot_one_model as pilot

MODEL = pilot.MODEL
MEMBER = pilot.MEMBER
SSPS = pilot.SSPS
# the aggregatable data variables (six indices + the two tas components)
DATA_VARS = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI",
             "tas_arctic", "tas_rest", "tas_global"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_RANK = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}  # chronological within a year
# first month (and span in months) of each season, with Dec assigned to next-year DJF
SEASON_START = {"DJF": (-1, 12), "MAM": (0, 3), "JJA": (0, 6), "SON": (0, 9)}


# ------------------------------------------------------------------- local I/O
def load_monthly(experiment: str, outdir: str = ".") -> xr.Dataset:
    """Open one monthly product file and strip the 2-D time_bnds so it doesn't
    interfere with the time-dimension aggregation."""
    path = os.path.join(outdir, f"ArcticIDX_{MODEL}_{experiment}.nc")
    ds = xr.open_dataset(path)
    return ds.drop_vars("time_bnds", errors="ignore")


# ------------------------------------------------------- datetime axis helpers
def _month_start(year: int, month: int) -> np.datetime64:
    """First instant of (year, month), rolling months outside 1..12 into years."""
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return np.datetime64(f"{year:04d}-{month:02d}-01", "ns")


def _season_span(label: str):
    """(lower, upper) datetime bounds of a "YYYY-SEA" season, upper exclusive."""
    year, sea = int(label[:4]), label[5:]
    yr_off, start_month = SEASON_START[sea]
    lo = _month_start(year + yr_off, start_month)
    up = _month_start(year + yr_off, start_month + 3)
    return lo, up


def _annual_span(year: int):
    """(lower, upper) datetime bounds of a calendar year, upper exclusive."""
    return _month_start(year, 1), _month_start(year + 1, 1)


def _attach_time_axis(means: xr.Dataset, grp_dim: str, labels: list,
                      mids: np.ndarray, los: np.ndarray, ups: np.ndarray,
                      time_long_name: str) -> xr.Dataset:
    """Turn a grouped-mean dataset into one with a CF datetime 'time' axis, a
    'time_bnds' coordinate, and a readable 'time_label' string coordinate."""
    out = means.rename({grp_dim: "time"})
    out = out.assign_coords(
        time=("time", mids),
        time_bnds=(("time", "bnds"), np.stack([los, ups], axis=1)),
        time_label=("time", np.array(labels)),
    )
    out["time"].attrs.update(standard_name="time", long_name=time_long_name, axis="T")
    out["time_label"].attrs.update(long_name="human-readable time label")
    return out


# ----------------------------------------------------------------- aggregation
def seasonal_means(ds: xr.Dataset):
    """Return (seasonal_ds, dropped). December -> following year's DJF; only
    complete 3-month seasons survive. 'time' = midpoint of each season's span."""
    month = ds["time"].dt.month.values
    year = ds["time"].dt.year.values
    season = ds["time"].dt.season.values                 # 'DJF','MAM','JJA','SON'
    season_year = np.where(month == 12, year + 1, year)  # Dec -> next year's DJF
    labels = np.array([f"{int(y):04d}-{s}" for y, s in zip(season_year, season)])

    counts = Counter(labels.tolist())
    complete = sorted(
        [lab for lab, c in counts.items() if c == 3],
        key=lambda lab: (int(lab[:4]), SEASON_RANK[lab[5:]]),
    )
    dropped = {lab: counts[lab] for lab in counts if counts[lab] != 3}

    means = ds.assign_coords(_grp=("time", labels)).groupby("_grp").mean().sel(_grp=complete)
    spans = [_season_span(lab) for lab in complete]
    los = np.array([lo for lo, _ in spans])
    ups = np.array([up for _, up in spans])
    mids = los + (ups - los) / 2                         # exact midpoint of the span
    out = _attach_time_axis(means, "_grp", complete, mids, los, ups,
                            "season midpoint (Dec assigned to following-year DJF)")
    return out, dropped


def annual_means(ds: xr.Dataset):
    """Return (annual_ds, dropped). A calendar year is kept only if all 12 months
    are present. 'time' is placed at YYYY-07-01; bounds span the calendar year."""
    year = ds["time"].dt.year.values
    counts = Counter(year.tolist())
    complete = sorted([int(y) for y, c in counts.items() if c == 12])
    dropped = {int(y): counts[y] for y in counts if counts[y] != 12}

    means = ds.assign_coords(_grp=("time", year)).groupby("_grp").mean().sel(_grp=complete)
    spans = [_annual_span(y) for y in complete]
    los = np.array([lo for lo, _ in spans])
    ups = np.array([up for _, up in spans])
    mids = np.array([np.datetime64(f"{y:04d}-07-01", "ns") for y in complete])
    labels = [f"{y:04d}" for y in complete]
    out = _attach_time_axis(means, "_grp", labels, mids, los, ups, "calendar year")
    return out, dropped


# ----------------------------------------------------------------------- write
def save_product(prod: xr.Dataset, experiment: str, frequency: str,
                 suffix: str, weighting: str, outdir: str = ".") -> tuple[str, str]:
    """Apply enrich_metadata (with cell_methods) and write NetCDF + CSV. The CSV
    is indexed by the readable time_label; the NetCDF keeps the CF datetime axis."""
    prod.attrs["area_weighting"] = weighting
    prod = pilot.enrich_metadata(prod, MODEL, MEMBER, experiment,
                                 frequency=frequency, cell_methods="time: mean")
    nc = os.path.join(outdir, f"ArcticIDX_{MODEL}_{experiment}_{suffix}.nc")
    csv = os.path.join(outdir, f"ArcticIDX_{MODEL}_{experiment}_{suffix}.csv")
    prod.to_netcdf(nc)

    # CSV: readable labels as the index (unchanged from before), indices only
    df = pd.DataFrame({v: prod[v].values for v in DATA_VARS},
                      index=pd.Index(prod["time_label"].values, name="time"))
    df.to_csv(csv)
    return nc, csv


def _fmt_dropped(dropped: dict) -> str:
    if not dropped:
        return "none"
    return ", ".join(f"{k} ({v} mo)" for k, v in dropped.items())


# ------------------------------------------------------------------------ main
def process_series(name: str, monthly: xr.Dataset, weighting: str) -> dict:
    """Build + save seasonal and annual products for one monthly series."""
    seas, seas_dropped = seasonal_means(monthly)
    ann, ann_dropped = annual_means(monthly)

    print(f"\n----- {name} -----")
    print(f"  seasonal: {seas['time'].size} rows | dropped: {_fmt_dropped(seas_dropped)}")
    print(f"  annual  : {ann['time'].size} rows | dropped: {_fmt_dropped(ann_dropped)}")
    for suffix, prod, freq in [("seasonal", seas, "seasonal"), ("annual", ann, "annual")]:
        nc, csv = save_product(prod, name, freq, suffix, weighting)
        print(f"    wrote {nc} and {csv}")
    return {"seasonal": seas, "annual": ann}


def sanity_historical(products: dict) -> None:
    """Row-count sanity for the historical per-experiment products."""
    seas, ann = products["seasonal"], products["annual"]
    by_season = Counter(str(lab)[5:] for lab in seas["time_label"].values)
    expected = {"DJF": 64, "MAM": 65, "JJA": 65, "SON": 65, "annual": 65}
    got = {s: by_season.get(s, 0) for s in SEASONS}
    got["annual"] = int(ann["time"].size)

    print("\n" + "=" * 60)
    print("Sanity — historical rows per index")
    print("=" * 60)
    ok = True
    for k in ["DJF", "MAM", "JJA", "SON", "annual"]:
        flag = "OK" if got[k] == expected[k] else "MISMATCH"
        ok = ok and got[k] == expected[k]
        print(f"  {k:<7}: {got[k]:>3}  (expect {expected[k]:>3})  {flag}")
    print(f"  RESULT : {'ALL MATCH' if ok else 'CHECK FAILED'}")
    print("=" * 60)


def main() -> None:
    experiments = ["historical"] + SSPS
    monthly = {exp: load_monthly(exp) for exp in experiments}
    weighting = monthly["historical"].attrs.get("area_weighting", "cos-lat")

    print(f"ArcticIDX derived products — {MODEL} (local files only, no recompute)")

    historical_products = None
    for exp in experiments:
        prods = process_series(exp, monthly[exp], weighting)
        if exp == "historical":
            historical_products = prods

    for ssp in SSPS:
        spliced = (
            xr.concat([monthly["historical"], monthly[ssp]], dim="time")
            .sortby("time")
            .drop_duplicates("time")
        )
        process_series(f"historical-{ssp}", spliced, weighting)

    if historical_products is not None:
        sanity_historical(historical_products)


if __name__ == "__main__":
    main()
