"""
ArcticIDX index computation engine.

Implements the six indices exactly as defined in Methods:
area-weighted spatial means on the native grid (areacella if available,
cosine-latitude otherwise), longitude convention harmonized to [-180, 180).

All box definitions live in BOXES — the single source of truth that must
match the paper text.

Usage sketch (monthly CMIP6 fields already loaded as xarray DataArrays):

    import arcticidx_indices as aidx

    zg500 = aidx.get_level(zg, 500)            # select 500 hPa
    zg500 = aidx.standardize(zg500)            # names + lon convention
    gbi   = aidx.box_index(zg500, "GBI")       # monthly GBI series

    ds_aai = aidx.aai(aidx.standardize(tas), base=("1981", "2010"))
    # -> Dataset with 'aai', 'tas_arctic', 'tas_rest', 'tas_global'
"""
from __future__ import annotations

import numpy as np
import xarray as xr

G = 9.80665  # m s-2, to convert ERA5 geopotential (m2 s-2) to height (m)

# ---------------------------------------------------------------- definitions
# Longitudes in [-180, 180). In this convention every box is a contiguous
# slice (in 0-360 the UBI box would wrap around the prime meridian).
BOXES = {
    "GBI": {"lat": (60.0, 80.0),  "lon": (-80.0, -20.0)},   # Hanna et al. 2013
    "BHI": {"lat": (72.5, 82.5),  "lon": (-180.0, -120.0)}, # Serreze & Barrett 2011
    "SHI": {"lat": (40.0, 65.0),  "lon": (80.0, 120.0)},    # Panagiotopoulos et al. 2005
    "UBI": {"lat": (45.0, 80.0),  "lon": (-10.0, 80.0)},    # Peings 2019
}
ARCTIC_LAT = 70.0   # AAI: Arctic = poleward of 70N (Liu et al. 2020)
PVI_LAT = 60.0      # PVI: zonal-mean ua at 10 hPa along 60N

_COORD_RENAMES = {  # common alternative coordinate names -> standard
    "latitude": "lat", "longitude": "lon",
    "Latitude": "lat", "Longitude": "lon",
}


# ------------------------------------------------------------------- helpers
def standardize(da: xr.DataArray) -> xr.DataArray:
    """Rename coords to lat/lon, move lon to [-180, 180), sort both ascending.

    Call this once on every field before computing indices. Sorting makes
    .sel(slice) safe for reanalyses stored with descending latitude.
    """
    renames = {k: v for k, v in _COORD_RENAMES.items() if k in da.dims or k in da.coords}
    if renames:
        da = da.rename(renames)
    lon = da["lon"]
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    da = da.assign_coords(lon=lon_wrapped).sortby("lon").sortby("lat")
    return da


def get_level(da: xr.DataArray, hpa: float) -> xr.DataArray:
    """Select a pressure level given in hPa, whatever units the file uses.

    CMIP6 stores plev in Pa (500 hPa -> 50000), reanalyses often in hPa.
    """
    for name in ("plev", "level", "lev", "pressure_level"):
        if name in da.dims:
            vals = da[name].values
            target = hpa * 100.0 if float(np.max(vals)) > 2000.0 else hpa
            return da.sel({name: target}, method="nearest").drop_vars(name)
    raise ValueError("No pressure-level dimension found on this DataArray.")


def to_hpa(psl: xr.DataArray) -> xr.DataArray:
    """Convert sea-level pressure to hPa if it arrives in Pa."""
    if float(psl.max()) > 2000.0:  # Pa territory; hPa values are ~1000
        out = psl / 100.0
        out.attrs["units"] = "hPa"
        return out
    return psl


def _weights(da: xr.DataArray, cell_area: xr.DataArray | None,
             label: str = "index") -> xr.DataArray:
    """Weight field on da's own (lat, lon) grid: areacella if supplied, else cos(lat).

    A CMIP6 fx (areacella) store can write lat/lon in a slightly different FLOAT
    representation than the data store even though it is the SAME physical grid.
    Left alone, ``da.weighted(cell_area)`` then inner-joins on the coordinates and
    SILENTLY drops every cell whose latitude does not bit-match (e.g. 56 of 96
    latitudes for MPI-ESM1-2-LR), skewing the area-weighted mean. So we REASSIGN
    da's own lat/lon onto the (equal-sized) cell-area field, after confirming the
    dimension sizes match — and fail loudly, naming the index, if they do not.
    """
    if cell_area is None:
        w = np.cos(np.deg2rad(da["lat"]))
        w.name = "cos_lat_weights"
        return w
    for d in ("lat", "lon"):
        if cell_area.sizes.get(d) != da.sizes.get(d):
            raise ValueError(
                f"{label}: areacella {d} size {cell_area.sizes.get(d)} != data {d} "
                f"size {da.sizes.get(d)} — the grids genuinely differ, cannot align "
                f"cell areas by coordinate reassignment; inspect the fx store.")
    aligned = cell_area.assign_coords(lat=da["lat"], lon=da["lon"])
    # Hard guard: after alignment the weight grid MUST match the data grid exactly,
    # so no silent inner-join can ever recur.
    if aligned.sizes["lat"] != da.sizes["lat"] or aligned.sizes["lon"] != da.sizes["lon"]:
        raise ValueError(f"{label}: weight/data (lat, lon) shape mismatch after "
                         f"alignment — refusing to weight.")
    return aligned


# -------------------------------------------------------------- box indices
def box_index(
    da: xr.DataArray,
    name: str,
    cell_area: xr.DataArray | None = None,
) -> xr.DataArray:
    """Area-weighted mean over a named ArcticIDX box (GBI/BHI/SHI/UBI).

    `da` must already be standardize()d; pass zg at 500 hPa for GBI/UBI
    and psl (any pressure unit; convert with to_hpa first) for BHI/SHI.
    """
    box = BOXES[name]
    sub = da.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
    if sub["lat"].size == 0 or sub["lon"].size == 0:
        raise ValueError(f"{name}: empty selection — was the field standardize()d?")
    ca = None
    if cell_area is not None:
        ca = cell_area.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
    out = sub.weighted(_weights(sub, ca, name)).mean(dim=("lat", "lon"))
    out.name = name
    return out


# ---------------------------------------------------------------------- PVI
def pvi(ua10: xr.DataArray) -> xr.DataArray:
    """Polar Vortex Index: zonal-mean zonal wind at 10 hPa along 60N.

    Pass ua already selected at 10 hPa (use get_level(ua, 10)) and
    standardize()d. Uses the model latitude nearest to 60N; on a single
    latitude circle all cells have equal area, so no weighting is needed.
    """
    out = ua10.mean(dim="lon").sel(lat=PVI_LAT, method="nearest")
    out = out.drop_vars("lat")
    out.name = "PVI"
    return out


# ---------------------------------------------------------------------- AAI
def _region_mean(
    da: xr.DataArray, mask: xr.DataArray, cell_area: xr.DataArray | None,
    label: str = "AAI",
) -> xr.DataArray:
    w = _weights(da, cell_area, label)
    return da.where(mask).weighted(w).mean(dim=("lat", "lon"))


def monthly_anomalies(series: xr.DataArray, base: tuple[str, str]) -> xr.DataArray:
    """Anomalies relative to the per-calendar-month climatology of `base`
    (inclusive year strings, e.g. ("1981", "2010")).

    The baseline years are selected with the ``time.dt.year`` accessor and a
    boolean mask, NOT string slicing (``.sel(time=slice("1981", "2010"))``).
    The two are equivalent for our monthly data, but string-slice ``.sel``
    compares the slice bounds (str) against the time values and raises on any
    non-datetime64 index, whereas the year mask behaves identically whether the
    axis decoded to numpy datetime64 or to cftime objects (360-day, noleap and
    other non-standard calendars included).
    """
    y0, y1 = int(base[0]), int(base[1])
    in_base = (series["time"].dt.year >= y0) & (series["time"].dt.year <= y1)
    clim = series.where(in_base, drop=True).groupby("time.month").mean("time")
    return (series.groupby("time.month") - clim).drop_vars("month")


def aai(
    tas: xr.DataArray,
    base: tuple[str, str] = ("1981", "2010"),
    cell_area: xr.DataArray | None = None,
) -> xr.Dataset:
    """Arctic Amplification Index (Liu et al. 2020): the difference between the
    area-weighted mean tas anomaly poleward of 70N and the GLOBAL-mean tas
    anomaly, both relative to `base`.

    The Arctic cap is INSIDE the global mean, so this is the true Liu et al.
    (2020) definition (Arctic minus global), NOT Arctic minus rest-of-globe.
    The two formulations are related exactly by

        AAI_global = (1 - f) * AAI_rest,

    where f = (area poleward of 70N) / (global area) ~ 0.030 — a ~3 % rescaling
    that leaves every correlation and trend sign unchanged.

    Area-mean first, anomaly second — mathematically identical to
    anomaly-first (both operations are linear), but far cheaper.

    Returns a Dataset with 'aai' plus the auxiliary absolute-temperature series
    'tas_arctic', 'tas_rest' and 'tas_global' (all distributed so users can
    reconstruct either the global or the rest-of-globe formulation).
    """
    arctic = _region_mean(tas, tas["lat"] >= ARCTIC_LAT, cell_area, "AAI/arctic")
    rest = _region_mean(tas, tas["lat"] < ARCTIC_LAT, cell_area, "AAI/rest")
    glob = _region_mean(tas, tas["lat"] >= -90.0, cell_area, "AAI/global")   # whole globe
    index = monthly_anomalies(arctic, base) - monthly_anomalies(glob, base)
    return xr.Dataset(
        {"aai": index, "tas_arctic": arctic, "tas_rest": rest, "tas_global": glob}
    )
