"""
ArcticIDX pilot pipeline — one model, historical + 4 SSPs, end to end.

Run one or more experiments by name, e.g.:

    python pilot_one_model.py historical
    python pilot_one_model.py ssp126 ssp245 ssp370 ssp585

With no argument it defaults to `historical` only (the playbook STOP point).
For each experiment it writes ArcticIDX_<model>_<experiment>.{nc,csv} with
the six indices plus tas_arctic / tas_rest, and prints a sanity table.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

import arcticidx_indices as aidx

CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
MODEL = "MPI-ESM1-2-LR"
MEMBER = "r1i1p1f1"
AMON_VARS = ["tas", "psl", "zg", "ua"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
# experiment -> (start_year, end_year) inclusive time slice
PERIODS = {
    "historical": ("1950", "2014"),
    "ssp126": ("2015", "2100"),
    "ssp245": ("2015", "2100"),
    "ssp370": ("2015", "2100"),
    "ssp585": ("2015", "2100"),
}
AAI_BASE = ("1981", "2010")  # baseline climatology, taken from the historical run


# Decode EVERY store's time axis to cftime, uniformly. Some Pangeo CMIP6 stores
# have time encodings whose dates fall outside numpy datetime64[ns] range and
# silently fall back to cftime, while sibling stores of the same model decode to
# datetime64. If a model's historical and one of its SSPs decode differently,
# concatenating them for the AAI baseline splice yields an unusable object-dtype
# time index (both string-slice .sel and the .dt accessor fail on it). Forcing
# cftime removes that inconsistency at the source and silences the out-of-range
# SerializationWarning. On read-back of our output NetCDFs (standard units +
# in-range calendars) xarray still decodes to datetime64, so downstream tools are
# unaffected.
_CFTIME_DECODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def _open_zarr(fs: gcsfs.GCSFileSystem, zstore: str) -> xr.Dataset:
    """Open a Pangeo zarr store with uniform cftime time decoding (see above)."""
    return xr.open_zarr(fs.get_mapper(zstore), consolidated=True,
                        decode_times=_CFTIME_DECODER)


def _year_slice(obj, y0, y1):
    """Inclusive year-range selection via the ``time.dt.year`` accessor.

    Equivalent to ``.sel(time=slice(y0, y1))`` for our monthly fields, but works
    on both datetime64 and cftime axes (string-slice .sel raises on a cftime
    axis). Accepts a DataArray or Dataset; the streamed field stays lazy because
    only the small time coordinate is evaluated to build the mask.
    """
    years = obj["time"].dt.year
    return obj.where((years >= int(y0)) & (years <= int(y1)), drop=True)


def _time_kind(obj) -> str:
    """'datetime64', 'cftime', or 'other' for obj's time-axis element type."""
    t = obj["time"]
    if np.issubdtype(t.dtype, np.datetime64):
        return "datetime64"
    vals = np.asarray(t.values).ravel()
    if vals.size and isinstance(vals[0], cftime.datetime):
        return "cftime"
    return "other"


# standard / gregorian / proleptic_gregorian are the SAME calendar for every date
# after the 1582 Gregorian reform, i.e. for all CMIP6 data (1850+). Models can even
# label their historical and SSP streams differently (e.g. MIROC6: historical
# proleptic_gregorian, SSPs standard); treat the whole family as interchangeable
# and canonicalize to proleptic_gregorian rather than refusing to splice.
_GREGORIAN_FAMILY = {"standard", "gregorian", "proleptic_gregorian"}


def _canonical_calendar(cals: set) -> str | None:
    """One target calendar for a set of cftime calendar names, or None if they are
    genuinely incompatible (different length-of-year families, e.g. noleap vs
    360_day). Gregorian-family names collapse to 'proleptic_gregorian'."""
    if len(cals) == 1:
        return next(iter(cals))
    if cals <= _GREGORIAN_FAMILY:
        return "proleptic_gregorian"
    return None


def _reconcile_time(parts: list, experiment: str) -> list:
    """Force a list of DataArrays onto ONE time representation before they are
    concatenated for the AAI baseline splice.

    Mixed datetime64/cftime parts — or two cftime parts on differently *named* but
    equivalent Gregorian calendars — concatenate into an unusable object-dtype
    index. Convert every part to one cftime calendar (datetime64 -> cftime and
    standard/gregorian -> proleptic_gregorian are both lossless for post-1582
    dates). Raise a clear error naming the model and experiment only when the
    calendars are genuinely incompatible (e.g. noleap vs 360_day).
    """
    kinds = {_time_kind(p) for p in parts}
    if kinds == {"datetime64"}:
        return parts                                   # already uniform
    cals = {p["time"].dt.calendar for p in parts if _time_kind(p) == "cftime"}
    target = _canonical_calendar(cals)
    if target is None:
        raise ValueError(
            f"{MODEL}/{experiment}: cannot splice the AAI baseline — parts carry "
            f"incompatible time calendars {sorted(cals)}. Inspect the stores."
        )
    out = []
    for p in parts:
        kind = _time_kind(p)
        if kind == "other":
            raise ValueError(
                f"{MODEL}/{experiment}: unrecognized (non-datetime) time axis; "
                f"inspect the stores."
            )
        if kind == "cftime" and p["time"].dt.calendar == target:
            out.append(p)
        else:                                          # datetime64, or a Gregorian
            out.append(p.convert_calendar(target, use_cftime=True))  # -family alias
    return out


# which index each raw input field feeds (for the coverage-gap warning)
_FIELD_INDICES = {"zg": ("GBI", "UBI"), "psl": ("BHI", "SHI"),
                  "ua": ("PVI",), "tas": ("AAI",)}


def _warn_time_gaps(fields: dict, experiment: str) -> list:
    """Warn LOUDLY (stderr) when a time-sliced input field does not cover the same
    period as the others, and return the gap records.

    The four Amon fields for one model/experiment should share one monthly time
    axis. When one is short (e.g. CanESM5 ssp126 ``ua`` is archived only for
    2101-2300, so its 2015-2100 slice is EMPTY), assembling the output Dataset
    would SILENTLY NaN-fill that field's index(es) via xarray alignment rather than
    erroring. This surfaces the mismatch at compute time. It is warn-and-continue:
    the fully-covered indices are still computed and written; the short field's
    index is (correctly) NaN where its data is missing. The reference coverage is
    the fullest field, so a model whose fields are all consistently short (e.g.
    CAMS-CSM1-0 ending 2099) is NOT flagged — only inter-field disagreement is.
    """
    sizes = {k: v["time"].size for k, v in fields.items()}
    full = max(sizes.values())
    gaps = []
    for k, v in fields.items():
        if sizes[k] < full:
            if sizes[k]:
                yrs = v["time"].dt.year.values
                span = f"{int(yrs.min())}-{int(yrs.max())}"
            else:
                span = "no overlap with requested period"
            gaps.append({"field": k, "have": int(sizes[k]), "expected": int(full),
                         "span": span, "indices": _FIELD_INDICES[k]})
            print(f"  WARNING {MODEL}/{experiment}: input '{k}' covers only "
                  f"{sizes[k]}/{full} months of the requested period ({span}); "
                  f"index(es) {', '.join(_FIELD_INDICES[k])} will be NaN where "
                  f"'{k}' is missing.", file=sys.stderr)
    return gaps


# ----------------------------------------------------------------- catalog I/O
def load_catalog() -> pd.DataFrame:
    print("Downloading Pangeo CMIP6 catalog (large CSV, be patient)...")
    cat = pd.read_csv(CATALOG_URL, low_memory=False)
    print(f"Catalog rows: {len(cat):,}")
    return cat


def _select_row(rows: pd.DataFrame) -> pd.Series:
    """Pick EXACTLY ONE catalog row from possibly several matches, so we never
    concatenate duplicate/overlapping stores: prefer grid_label 'gn' (native
    grid), then the latest version."""
    pick = rows
    if (pick["grid_label"] == "gn").any():
        pick = pick[pick["grid_label"] == "gn"]
    pick = pick.sort_values("version")   # ascending -> last is newest
    return pick.iloc[-1]


def _clean_time(da: xr.DataArray, experiment: str, variable: str) -> xr.DataArray:
    """Make a store's time axis trustworthy before any slicing.

    Some Pangeo CMIP6 stores ship a scrambled time axis (e.g. MPI ssp126 tas
    runs 2055..2034). Sort by time, drop duplicate timestamps, then REQUIRE a
    monotonic-increasing axis; raise a clear, named error if it still isn't.
    sortby reorders the data together with its coordinate, so a merely-shuffled
    axis is fully repaired.
    """
    da = da.sortby("time").drop_duplicates("time")
    if not pd.Index(da["time"].values).is_monotonic_increasing:
        raise ValueError(
            f"Non-monotonic time axis for {experiment}/{variable} even after "
            f"sortby+drop_duplicates -- store is unusable, inspect it manually."
        )
    return da


def open_var(cat: pd.DataFrame, experiment: str, variable: str,
             fs: gcsfs.GCSFileSystem) -> xr.DataArray:
    """Open the zarr store for (model, member, Amon, experiment, variable) and
    return the lazy DataArray with a sorted, de-duplicated, verified-monotonic
    time axis. Streams nothing until .compute()."""
    rows = cat[
        (cat["source_id"] == MODEL)
        & (cat["member_id"] == MEMBER)
        & (cat["table_id"] == "Amon")
        & (cat["variable_id"] == variable)
        & (cat["experiment_id"] == experiment)
    ]
    if rows.empty:
        raise ValueError(f"No {variable} for {MODEL} {MEMBER} {experiment}")
    row = _select_row(rows)
    ds = _open_zarr(fs, row["zstore"])
    return _clean_time(ds[variable], experiment, variable)


def open_areacella(cat: pd.DataFrame, fs: gcsfs.GCSFileSystem):
    """Return the standardize()d areacella field, or None (cos-lat fallback)."""
    rows = cat[
        (cat["source_id"] == MODEL)
        & (cat["table_id"] == "fx")
        & (cat["variable_id"] == "areacella")
    ]
    if rows.empty:
        print("  areacella: not available -> cosine-latitude weights")
        return None
    pref = rows[rows["member_id"] == MEMBER]
    row = _select_row(pref if not pref.empty else rows)
    ds = _open_zarr(fs, row["zstore"])
    print(f"  areacella: using member {row['member_id']} (area weighting)")
    return aidx.standardize(ds["areacella"])


# --------------------------------------------------------------- index compute
def compute_indices(experiment: str, cat: pd.DataFrame,
                    fs: gcsfs.GCSFileSystem, area) -> xr.Dataset:
    """Compute all six indices (+ tas_arctic/tas_rest) for one experiment."""
    y0, y1 = PERIODS[experiment]

    # zg @ 500 hPa and ua @ 10 hPa: select the level BEFORE loading.
    zg500 = aidx.standardize(
        _year_slice(aidx.get_level(open_var(cat, experiment, "zg", fs), 500), y0, y1)
    )
    ua10 = aidx.standardize(
        _year_slice(aidx.get_level(open_var(cat, experiment, "ua", fs), 10), y0, y1)
    )
    psl = aidx.to_hpa(
        aidx.standardize(_year_slice(open_var(cat, experiment, "psl", fs), y0, y1))
    )
    tas = aidx.standardize(_year_slice(open_var(cat, experiment, "tas", fs), y0, y1))

    # Guard: every input must cover the requested period identically, else its
    # index would be silently NaN-filled by Dataset alignment (see _warn_time_gaps).
    _warn_time_gaps({"zg": zg500, "ua": ua10, "psl": psl, "tas": tas}, experiment)

    gbi = aidx.box_index(zg500, "GBI", cell_area=area)
    ubi = aidx.box_index(zg500, "UBI", cell_area=area)
    bhi = aidx.box_index(psl, "BHI", cell_area=area)
    shi = aidx.box_index(psl, "SHI", cell_area=area)
    pvi = aidx.pvi(ua10)

    # AAI baseline is (1981-2010) from the HISTORICAL run. For historical the
    # slice already contains those years; for an SSP we splice the historical
    # baseline period onto the SSP series so aidx.aai() is reused unchanged,
    # then keep only the SSP period.
    if experiment == "historical":
        aai_ds = aidx.aai(tas, base=AAI_BASE, cell_area=area)
    else:
        hist_base = aidx.standardize(
            _year_slice(open_var(cat, "historical", "tas", fs), AAI_BASE[0], "2014")
        )
        # Splice baseline (historical) BEFORE the SSP series, chronological order.
        # Reconcile the two time representations FIRST: if historical and the SSP
        # decoded differently (datetime64 vs cftime), a raw concat produces an
        # unusable object-dtype index. Then sort + de-dup so the axis is monotonic.
        parts = _reconcile_time([hist_base, tas], experiment)
        combined = (
            xr.concat(parts, dim="time")
            .sortby("time")
            .drop_duplicates("time")
        )
        aai_ds = _year_slice(aidx.aai(combined, base=AAI_BASE, cell_area=area), y0, y1)

    # Physical-plausibility guard on the global-mean tas. Had the areacella grid
    # float-mismatch (silent inner-join) survived, the global mean would be far too
    # cold (~280 K instead of ~287 K); fail loudly rather than ship a skewed field.
    # Uses the 1981-2010 baseline window when present (stable ~285-290 K across
    # models), else a broad physical range for warm/late SSP-only slices.
    tg = aai_ds["tas_global"]
    ty = tg["time"].dt.year
    in_base = (ty >= int(AAI_BASE[0])) & (ty <= int(AAI_BASE[1]))
    if bool(in_base.any()):
        bm = float(tg.where(in_base).mean())
        if not (284.0 <= bm <= 291.0):
            raise ValueError(f"{MODEL}/{experiment}: 1981-2010 global-mean tas "
                             f"{bm:.2f} K outside physical 284-291 K — areacella "
                             f"weighting broken (grid mismatch)?")
    fm = float(tg.mean())
    if not (282.0 <= fm <= 297.0):
        raise ValueError(f"{MODEL}/{experiment}: global-mean tas {fm:.2f} K outside "
                         f"physical 282-297 K — inspect the weighting.")

    out = xr.Dataset(
        {
            "GBI": gbi,
            "UBI": ubi,
            "BHI": bhi,
            "SHI": shi,
            "PVI": pvi,
            "AAI": aai_ds["aai"],
            "tas_arctic": aai_ds["tas_arctic"],
            "tas_rest": aai_ds["tas_rest"],
            "tas_global": aai_ds["tas_global"],
        }
    )
    out.attrs.update(model=MODEL, member=MEMBER, experiment=experiment)
    print("  streaming and computing indices...")
    return out.compute()


# --------------------------------------------------------------- CF metadata
# Per-variable attributes for the archival NetCDF. These REPLACE (not merge
# with) whatever attrs the indices inherited from their CMIP6 source fields, so
# no stale comment/history/cell_methods/cell_measures/standard_name/_ChunkSizes
# leaks through. Keep in sync with arcticidx_indices.py and the paper.
VAR_META = {
    "GBI": {
        "long_name": "Greenland Blocking Index",
        "units": "m",
        "definition": "Area-weighted mean of monthly 500 hPa geopotential height "
                      "over 60-80N, 20-80W",
        "reference": "Hanna et al. (2013)",
    },
    "UBI": {
        "long_name": "Ural Blocking Index",
        "units": "m",
        "definition": "Area-weighted mean of monthly 500 hPa geopotential height "
                      "over 45-80N, 10W-80E, all calendar months",
        "reference": "Peings (2019)",
    },
    "BHI": {
        "long_name": "Beaufort High Index",
        "units": "hPa",
        "definition": "Area-weighted mean of monthly sea-level pressure over "
                      "72.5-82.5N, 180-120W",
        "reference": "Serreze & Barrett (2011); Zhang et al. (2024)",
    },
    "SHI": {
        "long_name": "Siberian High Index",
        "units": "hPa",
        "definition": "Area-weighted mean of monthly sea-level pressure over "
                      "40-65N, 80-120E",
        "reference": "Panagiotopoulos et al. (2005)",
    },
    "PVI": {
        "long_name": "Polar Vortex Index",
        "units": "m s-1",
        "definition": "Zonal-mean zonal wind at 10 hPa at 60N",
        "reference": "Charlton & Polvani (2007); Butler et al. (2015)",
        "note": "Physically meaningful primarily November-March",
    },
    "AAI": {
        "long_name": "Arctic Amplification Index",
        "units": "K",
        "definition": "Difference between the area-weighted mean tas anomaly "
                      "poleward of 70N and the global-mean tas anomaly",
        "baseline": "1981-2010",
        "reference": "Liu et al. (2020)",
        "note": "Arctic minus GLOBAL mean; related to the Arctic-minus-rest-of-globe "
                "form by AAI_global = (1 - f) * AAI_rest, f ~ 0.030. Reconstruct the "
                "rest-of-globe form from tas_arctic and tas_rest if required.",
    },
    "tas_arctic": {
        "long_name": "Arctic-mean near-surface air temperature (poleward of 70N)",
        "units": "K",
    },
    "tas_rest": {
        "long_name": "Rest-of-globe mean near-surface air temperature "
                     "(equatorward of 70N)",
        "units": "K",
    },
    "tas_global": {
        "long_name": "Global-mean near-surface air temperature",
        "units": "K",
    },
}


def _is_datetime(da: xr.DataArray) -> bool:
    """True only if the axis holds real datetime64 / cftime values. Derived
    products (seasonal string labels, integer years) return False, so the
    monthly-specific time_bnds machinery is skipped for them."""
    if np.issubdtype(da.dtype, np.datetime64):
        return True
    if da.dtype == object and da.size:
        try:
            import cftime
            return isinstance(np.asarray(da.values).ravel()[0], cftime.datetime)
        except Exception:
            return False
    return False


def _time_bounds(time: xr.DataArray) -> np.ndarray:
    """CF cell bounds for a monthly time axis.

    For each step returns [first instant of that month, first instant of the
    next month] as an (N, 2) array in the SAME dtype as ``time`` (numpy
    datetime64 for standard calendars, cftime objects otherwise), so xarray
    encodes it with the time axis' own units and calendar.
    """
    years = time.dt.year.values
    months = time.dt.month.values
    nyears = np.where(months == 12, years + 1, years)
    nmonths = np.where(months == 12, 1, months + 1)
    if np.issubdtype(time.dtype, np.datetime64):
        lower = np.array([np.datetime64(f"{int(y):04d}-{int(m):02d}-01", "ns")
                          for y, m in zip(years, months)])
        upper = np.array([np.datetime64(f"{int(y):04d}-{int(m):02d}-01", "ns")
                          for y, m in zip(nyears, nmonths)])
    else:
        import cftime
        cal = time.dt.calendar
        lower = np.array([cftime.datetime(int(y), int(m), 1, calendar=cal)
                          for y, m in zip(years, months)])
        upper = np.array([cftime.datetime(int(y), int(m), 1, calendar=cal)
                          for y, m in zip(nyears, nmonths)])
    return np.stack([lower, upper], axis=1)


def enrich_metadata(ds: xr.Dataset, model: str, member: str,
                    experiment: str, frequency: str = "mon",
                    cell_methods: str | None = None) -> xr.Dataset:
    """Return a copy of ``ds`` with clean, CF-style metadata for archival NetCDF.

    (1) drops the stray 2 m 'height' scalar coordinate;
    (2) REPLACES each index variable's attrs wholesale from ``VAR_META`` and
        clears its encoding, so no inherited CMIP6 comment/history/cell_methods/
        cell_measures/standard_name/_ChunkSizes survives; if ``cell_methods`` is
        given (e.g. "time: mean" for derived aggregates) it is added afterwards;
    (3) for a real (datetime) time axis, wires up CF time_bnds -- building the
        monthly bounds itself if none are attached, or keeping caller-supplied
        bounds (seasonal/annual spans) and just linking + encoding them;
    (4) sets a fresh set of global attrs, tagged with ``frequency``.

    The area-weighting label is read from ``ds.attrs['area_weighting']`` (set by
    the caller before this call) before the globals are wiped; it falls back to
    'cos-lat' if absent.
    """
    weighting = ds.attrs.get("area_weighting", "cos-lat")
    ds = ds.drop_vars("height", errors="ignore")

    for name, attrs in VAR_META.items():
        if name in ds:
            ds[name].attrs = dict(attrs)
            if cell_methods:
                ds[name].attrs["cell_methods"] = cell_methods
            ds[name].encoding = {}

    # Wire up CF time_bnds for any real datetime axis. If the caller already
    # attached time_bnds (derived seasonal/annual spans), keep it; otherwise
    # build the monthly bounds. Either way link via time:bounds and match the
    # time axis' units/calendar so the numbers stay consistent.
    if "time" in ds.coords and _is_datetime(ds["time"]):
        tenc = ds["time"].encoding
        units = tenc.get("units") or "hours since 1850-01-01 00:00:00"
        calendar = tenc.get("calendar") or ds["time"].dt.calendar
        ds["time"].encoding["units"] = units
        ds["time"].encoding["calendar"] = calendar
        ds["time"].encoding.pop("bounds", None)   # avoid attrs/encoding 'bounds' clash
        if "time_bnds" not in ds:
            ds["time_bnds"] = (("time", "bnds"), _time_bounds(ds["time"]))
        ds["time"].attrs["bounds"] = "time_bnds"
        bnds_enc = {"units": units, "calendar": calendar}
        if "dtype" in tenc:
            bnds_enc["dtype"] = tenc["dtype"]
        ds["time_bnds"].encoding = bnds_enc

    ds.attrs = {
        "title": f"ArcticIDX indices: {model} {experiment}",
        "Conventions": "CF-1.8",
        "source_id": model,
        "variant_label": member,
        "experiment_id": experiment,
        "frequency": frequency,
        "area_weighting": weighting,
        "creation_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": "v0.1-pilot",
        "license": "CC BY 4.0",
        "contact": "[to fill]",
    }
    return ds


# ------------------------------------------------------------------- CSV writer
def write_csv(ds: xr.Dataset, csv: str) -> None:
    """Write the per-experiment CSV view of the results.

    Two simplifications versus the raw dataset, applied to the CSV ONLY (the
    NetCDF keeps the full datetime axis and all coords for downstream tools):
      1. drop the stray 2 m 'height' scalar coordinate that rides along on tas;
      2. render the monthly time axis as compact 'YYYY-MM' strings so the period
         of each row is obvious (and doesn't show as '####' in a narrow Excel
         column).
    """
    df = ds.drop_vars(["height", "time_bnds"], errors="ignore").to_dataframe()
    df.index = [f"{int(y):04d}-{int(m):02d}"
                for y, m in zip(ds["time.year"].values, ds["time.month"].values)]
    df.index.name = "time"
    df.to_csv(csv)


# ---------------------------------------------------------------- sanity table
def print_sanity(ds: xr.Dataset, experiment: str) -> None:
    djf = ds.sel(time=ds["time.season"] == "DJF")
    gbi_mean = float(ds["GBI"].mean())
    shi_winter = float(djf["SHI"].mean())
    pvi_djf = float(djf["PVI"].mean())

    index_vars = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI",
                  "tas_arctic", "tas_rest", "tas_global"]
    nan_counts = {v: int(np.isnan(ds[v].values).sum()) for v in index_vars}
    total_nans = sum(nan_counts.values())

    print(f"\n============ Sanity table — {MODEL} / {experiment} ============")
    print(f"  time span            : {str(ds['time'].values[0])[:7]} .. "
          f"{str(ds['time'].values[-1])[:7]}  ({ds['time'].size} months)")
    print(f"  mean GBI             : {gbi_mean:8.2f} m      (expect ~5200-5600)")
    print(f"  mean winter SHI (DJF): {shi_winter:8.2f} hPa    (expect ~1025-1040)")
    print(f"  mean DJF PVI         : {pvi_djf:8.2f} m/s    (expect positive westerlies)")
    print("  ---- full-series means ----")
    for v in ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]:
        print(f"    {v:11s}: {float(ds[v].mean()):10.3f}")
    if total_nans == 0:
        print("  NaN check            : PASS (0 NaNs across all indices)")
    else:
        print(f"  NaN check            : FAIL ({total_nans} NaNs: "
              f"{ {k: c for k, c in nan_counts.items() if c} })")
    print("=" * (len(f"============ Sanity table — {MODEL} / {experiment} ============")))


# ------------------------------------------------------------------------ main
def main() -> None:
    requested = [a for a in sys.argv[1:] if a in PERIODS]
    experiments = requested or ["historical"]

    print(f"ArcticIDX pilot — model {MODEL}, member {MEMBER}")
    print(f"Experiments this run: {experiments}\n")

    cat = load_catalog()
    fs = gcsfs.GCSFileSystem(token="anon")
    print("Weighting source:")
    area = open_areacella(cat, fs)

    for experiment in experiments:
        print(f"\n----- {experiment} -----")
        ds = compute_indices(experiment, cat, fs, area)
        nc = f"ArcticIDX_{MODEL}_{experiment}.nc"
        csv = f"ArcticIDX_{MODEL}_{experiment}.csv"
        ds.attrs["area_weighting"] = "areacella" if area is not None else "cos-lat"
        ds = enrich_metadata(ds, MODEL, MEMBER, experiment)
        ds.to_netcdf(nc)
        write_csv(ds, csv)
        print(f"  wrote {nc} and {csv}")
        print_sanity(ds, experiment)

    remaining = [s for s in SSPS if s not in experiments]
    if "historical" in experiments and remaining:
        print("\n" + "#" * 64)
        print("# STOP — historical done. Review the sanity table above.")
        print(f"# When approved, run the SSPs:  python pilot_one_model.py {' '.join(SSPS)}")
        print("#" * 64)


if __name__ == "__main__":
    main()
