"""era5_indices.py -- the six ArcticIDX indices from ERA5 monthly-mean reanalysis.

Mirrors the CMIP6 pipeline (pilot_one_model.py + derived_products.py) but reads the
four local ERA5 NetCDF files instead of Pangeo zarr. ALL index math is imported
from arcticidx_indices.py -- nothing here reimplements an index function.

Source -> index mapping (each index from its OWN file, on that file's native grid;
the files are never combined spatially, so their differing grids never interact):

    z   geopotential height (m) @ 500 hPa -> GBI, UBI  (already height; see note)
    msl sea-level pressure (Pa)          -> BHI, SHI   (to_hpa -> hPa)
    u   zonal wind (m/s) @ 10 hPa        -> PVI
    t2m 2 m temperature (K)              -> AAI

NOTE on z: this file's `units` attribute reads "m2 s-2" (geopotential), but its
values are ~5000-5900, i.e. already geopotential HEIGHT in metres. So it is used
directly, WITHOUT dividing by g -- dividing would wrongly yield ~560 m.

ERA5 has no areacella, so cosine-latitude weighting is used throughout
(area_weighting = "cos-lat"). ERA5 timestamps are inconsistent (mid-month in some
ranges, first-of-month in others), so every time axis is normalised to a canonical
first-of-month before any alignment -- the indices are joined by year-month, never
by exact datetime. Output is restricted to the common overlapping period across all
four files. Products (monthly / seasonal / annual, NetCDF + CSV) match the CMIP6
structure, tagged source_id "ERA5", with the same DJF convention as derived_products.

    python era5_indices.py
"""
from __future__ import annotations

import functools
import os

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

import arcticidx_indices as aidx
import pilot_one_model as pilot
import derived_products as dp

FILES = {
    "t2m": "ERA5_2m_temp_monmean_5024.nc",
    "z":   "ERA5_monmean_GPH_500hpa_50-24.nc",   # replaced 2026-08-11: now 1950-2024
    "msl": "ERA5_SLP_monmean_5024.nc",
    "u":   "ERA5_ucomp_10hpa_monmean__5024.nc",
}
MODEL = "ERA5"
MEMBER = "reanalysis"          # variant_label
EXPERIMENT = "reanalysis"      # experiment_id / filename token
OUTDIR = "ERA5"
WEIGHTING = "cos-lat"          # ERA5 has no areacella
BASELINE = ("1981", "2010")    # AAI climatology
# Recorded into every ERA5 output NetCDF (GBI/UBI provenance).
GPH_NOTE = ("GBI/UBI were computed from 500 hPa geopotential HEIGHT used directly in "
            "metres; the source file's units attribute reads 'm2 s-2' but its values "
            "are metres, so NO division by g (9.80665) was applied.")


# ------------------------------------------------------------------- loading
def _month_start(da: xr.DataArray) -> xr.DataArray:
    """Replace the time axis with a canonical first-of-month datetime64, so fields
    whose source timestamps disagree on day-of-month (mid-month vs 1st) still align
    by year-month. ERA5 must never be joined on exact datetimes."""
    t = pd.to_datetime(da["time"].values)
    ms = np.array([np.datetime64(f"{y:04d}-{m:02d}-01", "ns")
                   for y, m in zip(t.year, t.month)])
    return da.assign_coords(time=ms).sortby("time")


def load_field(key: str) -> xr.DataArray:
    """Open one ERA5 file lazily; return its DataArray with the time dim renamed to
    'time', normalised to month-start, and stray scalar coords dropped."""
    path = FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERA5 file not found: {path}")
    da = xr.open_dataset(path)[key]                 # in-range dates -> datetime64
    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename(valid_time="time")
    da = da.drop_vars([c for c in ("number", "expver") if c in da.coords],
                      errors="ignore")
    return _month_start(da)


# ------------------------------------------------------------- index compute
def compute_indices() -> tuple[xr.Dataset, np.ndarray]:
    """Compute all six indices (+ tas_arctic/tas_rest) on their native ERA5 grids
    and restrict to the common overlapping period. Returns (dataset, common_times)."""
    # GBI, UBI: 500 hPa geopotential HEIGHT (m). NB this specific file is MISLABELLED
    # -- its units attribute reads "m2 s-2" (geopotential) but its values are metres
    # (~4900-5900) -- so it is used directly, with NO division by g. The hard guard
    # below confirms the magnitude: if the file were ever true geopotential it fails
    # loudly telling you to divide by g, rather than silently producing ~560 m GBI.
    z500 = aidx.standardize(aidx.get_level(load_field("z"), 500))
    z_mean = float(z500.mean())
    if 45000.0 <= z_mean <= 60000.0:
        raise ValueError(
            f"z 500 hPa field mean is {z_mean:.0f} m2 s-2 -- this is GEOPOTENTIAL, "
            f"not geopotential height. Divide by 9.80665 (g) to convert to metres "
            f"before computing GBI/UBI (expected height mean ~4500-6000 m)."
        )
    if not (4500.0 <= z_mean <= 6000.0):
        raise ValueError(
            f"z 500 hPa field mean is {z_mean:.1f}, outside the expected geopotential-"
            f"height range (~4500-6000 m). Inspect the z file before computing GBI/UBI."
        )
    gbi = aidx.box_index(z500, "GBI")               # cell_area=None -> cos-lat weights
    ubi = aidx.box_index(z500, "UBI")

    # BHI, SHI: sea-level pressure Pa -> hPa.
    psl = aidx.to_hpa(aidx.standardize(load_field("msl")))
    bhi = aidx.box_index(psl, "BHI")
    shi = aidx.box_index(psl, "SHI")

    # PVI: zonal-mean u @ 10 hPa along 60N. Restrict to a latitude band around 60N
    # before the zonal mean -- an identical result (pvi zonal-means per latitude then
    # picks the nearest to 60N), but it avoids materialising the full 0.25 deg cube.
    ua10 = aidx.standardize(aidx.get_level(load_field("u"), 10))
    pvi = aidx.pvi(ua10.sel(lat=slice(55.0, 65.0)))

    # AAI: 2 m temperature, anomalies vs 1981-2010, cos-lat weighting.
    tas = aidx.standardize(load_field("t2m"))
    aai = aidx.aai(tas, base=BASELINE)

    # Common overlapping period (year-month) across the four source fields, taken on
    # the canonical month-start axes so the join is purely by year-month.
    common = functools.reduce(np.intersect1d, [
        gbi["time"].values, bhi["time"].values, pvi["time"].values, aai["time"].values])
    sel = lambda x: x.sel(time=common)

    out = xr.Dataset({
        "GBI": sel(gbi), "UBI": sel(ubi), "BHI": sel(bhi), "SHI": sel(shi),
        "PVI": sel(pvi), "AAI": sel(aai["aai"]),
        "tas_arctic": sel(aai["tas_arctic"]), "tas_rest": sel(aai["tas_rest"]),
        "tas_global": sel(aai["tas_global"]),
    })
    out.attrs.update(model=MODEL, member=MEMBER, experiment=EXPERIMENT)
    print("  streaming ERA5 and computing indices...")
    return out.compute(), common


# ------------------------------------------------------------------- sanity
def print_sanity(out: xr.Dataset, seas: xr.Dataset, ann: xr.Dataset) -> None:
    djf = out.sel(time=out["time.season"] == "DJF")
    yr = out["time"].dt.year
    aai_base = float(out["AAI"].where((yr >= 1981) & (yr <= 2010), drop=True).mean())
    index_vars = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI",
                  "tas_arctic", "tas_rest", "tas_global"]
    nan_counts = {v: int(np.isnan(out[v].values).sum()) for v in index_vars}
    total_nans = sum(nan_counts.values())

    print("\n============ Sanity table -- ERA5 ============")
    print(f"  mean GBI             : {float(out['GBI'].mean()):8.2f} m      (expect ~5200-5600)")
    print(f"  DJF mean SHI         : {float(djf['SHI'].mean()):8.2f} hPa    (expect ~1025-1040)")
    print(f"  DJF mean PVI         : {float(djf['PVI'].mean()):8.2f} m/s    (expect positive ~15-35)")
    print(f"  AAI 1981-2010 mean   : {aai_base:8.4f} K      (expect ~0)")
    print("  ---- full-series means ----")
    for v in ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]:
        print(f"    {v:11s}: {float(out[v].mean()):10.3f}")
    print(f"  row counts           : monthly={out['time'].size}  "
          f"seasonal={seas['time'].size}  annual={ann['time'].size}")
    if total_nans == 0:
        print("  NaN check            : PASS (0 NaNs across all indices)")
    else:
        print(f"  NaN check            : FAIL ({total_nans}: "
              f"{ {k: c for k, c in nan_counts.items() if c} })")
    print("=" * 46)


# ------------------------------------------------------------------------ main
def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    dp.MODEL, dp.MEMBER = MODEL, MEMBER

    print(f"ArcticIDX from ERA5 -- reading {len(FILES)} files, cos-lat weighting")
    out, common = compute_indices()
    cmin, cmax = pd.to_datetime(common.min()), pd.to_datetime(common.max())
    print(f"Common overlapping period: {cmin:%Y-%m} .. {cmax:%Y-%m} ({len(common)} months)")

    # Derived products from the pristine monthly indices (same code + DJF convention
    # as the CMIP6 models: December belongs to the following year's DJF).
    seas, seas_dropped = dp.seasonal_means(out)
    ann, ann_dropped = dp.annual_means(out)

    # Monthly product (NetCDF + CSV), CF metadata via the shared enricher. The GBI/UBI
    # provenance note is added AFTER enrich_metadata (which rewrites global attrs).
    out.attrs["area_weighting"] = WEIGHTING
    monthly = pilot.enrich_metadata(out, MODEL, MEMBER, EXPERIMENT)
    monthly.attrs["geopotential_height_note"] = GPH_NOTE
    nc = os.path.join(OUTDIR, f"ArcticIDX_{MODEL}_{EXPERIMENT}.nc")
    csv = os.path.join(OUTDIR, f"ArcticIDX_{MODEL}_{EXPERIMENT}.csv")
    monthly.to_netcdf(nc)
    pilot.write_csv(monthly, csv)
    print(f"wrote {nc} and {csv}")

    # Seasonal + annual (NetCDF + CSV) via the shared aggregator/saver. save_product
    # rewrites global attrs, so the note is appended in place afterwards.
    for prod, suffix, freq in [(seas, "seasonal", "seasonal"), (ann, "annual", "annual")]:
        p_nc, p_csv = dp.save_product(prod, EXPERIMENT, freq, suffix, WEIGHTING, outdir=OUTDIR)
        with netCDF4.Dataset(p_nc, "a") as d:
            d.setncattr("geopotential_height_note", GPH_NOTE)
        print(f"wrote {p_nc} and {p_csv}")
    print(f"  seasonal dropped (incomplete): {seas_dropped or 'none'}")
    print(f"  annual dropped (incomplete)  : {ann_dropped or 'none'}")

    print_sanity(out, seas, ann)


if __name__ == "__main__":
    main()
