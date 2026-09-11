"""
Verification of arcticidx_indices.py on synthetic data with KNOWN answers.
Run:  python3 test_arcticidx.py
Every test prints PASS or raises — no test framework needed.
"""
import numpy as np
import pandas as pd
import xarray as xr

import arcticidx_indices as aidx


def make_field(values, lon0360=False, name="zg"):
    """1-degree global field. values: scalar, or callable f(lat2d, lon2d)."""
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.0, 360.0, 1.0) if lon0360 else np.arange(-180.0, 180.0, 1.0)
    time = pd.date_range("2000-01-01", periods=3, freq="MS")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    base = np.full_like(la, float(values)) if np.isscalar(values) else values(la, lo)
    data = np.broadcast_to(base, (len(time),) + base.shape).copy()
    return xr.DataArray(
        data, coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"), name=name,
    )


def check(label, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    assert ok, label


print("1) Constant field -> every box index returns exactly the constant")
zg = aidx.standardize(make_field(5500.0))
for box in ("GBI", "UBI"):
    val = aidx.box_index(zg, box).isel(time=0).item()
    check(f"{box} == 5500 (got {val:.6f})", abs(val - 5500.0) < 1e-6)
psl = aidx.standardize(make_field(101300.0, name="psl"))
psl = aidx.to_hpa(psl)
for box in ("BHI", "SHI"):
    val = aidx.box_index(psl, box).isel(time=0).item()
    check(f"{box} == 1013 hPa (got {val:.6f})", abs(val - 1013.0) < 1e-6)

print("2) cos-weighting matches the closed-form integral (field = latitude)")
# weighted mean of phi over [60,80] with cos weights:
# I = int phi*cos(phi) dphi / int cos(phi) dphi  (phi in radians for eval)
p1, p2 = np.deg2rad(60.0), np.deg2rad(80.0)
num = (np.cos(p2) + p2 * np.sin(p2)) - (np.cos(p1) + p1 * np.sin(p1))
den = np.sin(p2) - np.sin(p1)
expected = np.rad2deg(num / den)
lat_field = aidx.standardize(make_field(lambda la, lo: la))
got = aidx.box_index(lat_field, "GBI").isel(time=0).item()
check(f"analytic {expected:.4f} vs computed {got:.4f}", abs(got - expected) < 0.05)
naive = (
    lat_field.sel(lat=slice(60, 80), lon=slice(-80, -20)).mean(("lat", "lon")).isel(time=0).item()
)
check(f"weighted ({got:.3f}) < naive unweighted ({naive:.3f})", got < naive)

print("3) Longitude convention: 0-360 input gives identical results")
f = lambda la, lo: 5000.0 + 3.0 * np.cos(np.deg2rad(((lo + 180) % 360) - 180))
a = aidx.box_index(aidx.standardize(make_field(f)), "UBI").isel(time=0).item()
b = aidx.box_index(aidx.standardize(make_field(f, lon0360=True)), "UBI").isel(time=0).item()
check(f"UBI(-180..180)={a:.6f} == UBI(0..360)={b:.6f}", abs(a - b) < 1e-9)

print("4) PVI: zonal mean along the 60N circle")
lat4 = np.arange(-90.0, 90.5, 1.0)          # contains 60.0 exactly
lon4 = np.arange(-180.0, 180.0, 1.0)
la4, lo4 = np.meshgrid(lat4, lon4, indexing="ij")
vals = np.full_like(la4, 7.0)
vals[la4 == 60.0] = np.where(lo4[la4 == 60.0] >= 0.0, 44.0, 40.0)  # mean = 42
t4 = pd.date_range("2000-01-01", periods=3, freq="MS")
ua = xr.DataArray(np.broadcast_to(vals, (3,) + vals.shape).copy(),
                  coords={"time": t4, "lat": lat4, "lon": lon4},
                  dims=("time", "lat", "lon"), name="ua")
val = aidx.pvi(aidx.standardize(ua)).isel(time=0).item()
check(f"PVI == 42, the zonal mean at 60N (got {val:.6f})", abs(val - 42.0) < 1e-6)

print("5) Pressure-level selection handles Pa and hPa")
lat = np.arange(-89.5, 90, 1.0); lon = np.arange(-180.0, 180, 1.0)
plev_pa = xr.DataArray(
    np.stack([np.full((len(lat), len(lon)), 5000.0),      # 100 hPa level
              np.full((len(lat), len(lon)), 5500.0)]),    # 500 hPa level
    coords={"plev": [10000.0, 50000.0], "lat": lat, "lon": lon},
    dims=("plev", "lat", "lon"),
)
check("Pa axis: 500 hPa -> 5500", aidx.get_level(plev_pa, 500).mean().item() == 5500.0)
plev_hpa = plev_pa.assign_coords(plev=[100.0, 500.0])
check("hPa axis: 500 hPa -> 5500", aidx.get_level(plev_hpa, 500).mean().item() == 5500.0)

print("6) AAI: Arctic +2 K vs rest +0.5 K, Arctic-minus-GLOBAL (Liu et al. 2020)")
lat = np.arange(-89.5, 90, 1.0); lon = np.arange(-180.0, 180, 1.0)
time = pd.date_range("1981-01-01", periods=36, freq="MS")  # 1981-1983
la, _ = np.meshgrid(lat, lon, indexing="ij")
base = np.where(la >= 70.0, 250.0, 285.0)
data = np.broadcast_to(base, (36,) + base.shape).copy()
data[24:] = np.where(la >= 70.0, base + 2.0, base + 0.5)   # year 3 perturbed
tas = xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                   dims=("time", "lat", "lon"), name="tas")
out = aidx.aai(aidx.standardize(tas), base=("1981", "1982"))

# area fraction poleward of 70N with cos-latitude weights on this 1-degree grid
w = np.cos(np.deg2rad(lat))
F70 = float(w[lat >= 70.0].sum() / w.sum())
# Arctic anomaly +2, rest anomaly +0.5 -> Arctic-minus-rest AAI = 1.5; the
# Arctic sits inside the global mean, so Arctic-minus-global AAI = (1 - f) * 1.5.
AAI_PERT = (1.0 - F70) * 1.5

last = out["aai"].isel(time=-1).item()
during_base = out["aai"].isel(time=5).item()
check(f"f (area fraction poleward of 70N) ~ 0.0301 (got {F70:.6f})",
      abs(F70 - 0.0301537) < 5e-4)
check(f"AAI in baseline period == 0 (got {during_base:.6f})", abs(during_base) < 1e-9)
check(f"AAI (Arctic-global) after perturbation == (1-f)*1.5 = {AAI_PERT:.6f} "
      f"(got {last:.6f})", abs(last - AAI_PERT) < 1e-9)
check("auxiliary tas_arctic/tas_rest/tas_global present",
      set(out.data_vars) == {"aai", "tas_arctic", "tas_rest", "tas_global"})

# The exact 3% rescaling identity the paper relies on: AAI_global = (1 - f) * AAI_rest
# at EVERY timestep (so all correlations and trend signs are unchanged), where
# AAI_rest is the old Arctic-minus-rest-of-globe formulation rebuilt from the aux vars.
aai_rest = (aidx.monthly_anomalies(out["tas_arctic"], ("1981", "1982"))
            - aidx.monthly_anomalies(out["tas_rest"], ("1981", "1982")))
check("identity: AAI_global == (1 - f) * AAI_rest at every timestep",
      np.allclose(out["aai"].values, (1.0 - F70) * aai_rest.values, atol=1e-9))
check("tas_global == f*tas_arctic + (1-f)*tas_rest (global mean = area-weighted blend)",
      np.allclose(out["tas_global"].values,
                  F70 * out["tas_arctic"].values + (1.0 - F70) * out["tas_rest"].values,
                  atol=1e-9))

print("\n7) AAI on a cftime 360-day calendar axis -> identical construction result")
# Same construction as test 6 (Arctic +2 K, rest +0.5 K in year 3 -> AAI = 1.5),
# but on a non-standard 360-day calendar that decodes to cftime. Proves the
# baseline selection in monthly_anomalies is calendar-agnostic: string-slice
# .sel can raise on a cftime axis; the year-mask must give the same answer.
import warnings as _warnings


def _cftrange(*a, **k):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", FutureWarning)   # cftime_range -> date_range rename
        return xr.cftime_range(*a, **k)


lat = np.arange(-89.5, 90, 1.0); lon = np.arange(-180.0, 180, 1.0)
time360 = _cftrange("1981-01-01", periods=36, freq="MS", calendar="360_day")
la, _ = np.meshgrid(lat, lon, indexing="ij")
base = np.where(la >= 70.0, 250.0, 285.0)
data = np.broadcast_to(base, (36,) + base.shape).copy()
data[24:] = np.where(la >= 70.0, base + 2.0, base + 0.5)   # year 3 perturbed
tas360 = xr.DataArray(data, coords={"time": time360, "lat": lat, "lon": lon},
                      dims=("time", "lat", "lon"), name="tas")
check("360-day axis decodes to CFTimeIndex",
      type(tas360.indexes["time"]).__name__ == "CFTimeIndex")
out360 = aidx.aai(aidx.standardize(tas360), base=("1981", "1982"))
during360 = out360["aai"].isel(time=5).item()
last360 = out360["aai"].isel(time=-1).item()
check(f"[360-day] AAI in baseline period == 0 (got {during360:.6f})", abs(during360) < 1e-9)
check(f"[360-day] AAI after perturbation == (1-f)*1.5 (got {last360:.6f})", abs(last360 - AAI_PERT) < 1e-9)

print("\n8) Mixed datetime64+cftime splice: pilot._reconcile_time repairs the index")
# Reproduces the production dry-run bug: historical decoded to datetime64, an SSP
# decoded to cftime. Concatenated directly they form an object-dtype time index on
# which BOTH string-slice .sel and the .dt accessor fail, so aai() cannot build its
# baseline. pilot._reconcile_time converts both to one cftime calendar first.
import pilot_one_model as pilot

h_t = pd.date_range("1981-01-01", periods=24, freq="MS")                            # datetime64
s_t = _cftrange("1983-01-01", periods=12, freq="MS", calendar="proleptic_gregorian")  # cftime
bb = np.where(la >= 70.0, 250.0, 285.0)
hist = aidx.standardize(xr.DataArray(
    np.broadcast_to(bb, (24,) + bb.shape).copy(),
    coords={"time": h_t, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="tas"))
sv = np.where(la >= 70.0, bb + 2.0, bb + 0.5)
ssp = aidx.standardize(xr.DataArray(
    np.broadcast_to(sv, (12,) + sv.shape).copy(),
    coords={"time": s_t, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="tas"))

# (a) the hazard: a raw concat is an unusable object index, and aai() fails on it
bad = xr.concat([hist, ssp], dim="time")
check(f"raw mixed concat is neither Datetime nor CFTimeIndex "
      f"(got {type(bad.indexes['time']).__name__})",
      type(bad.indexes["time"]).__name__ not in ("DatetimeIndex", "CFTimeIndex"))
bug = False
try:
    aidx.aai(bad, base=("1981", "1982"))
except (TypeError, AttributeError):
    bug = True
check("aai() on the unreconciled mixed index fails (year-mask alone cannot fix it)", bug)

# (b) the fix: reconcile -> one cftime calendar -> clean index -> correct AAI
parts = pilot._reconcile_time([hist, ssp], "ssp585")
fixed = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")
check(f"reconciled concat is a CFTimeIndex (got {type(fixed.indexes['time']).__name__})",
      type(fixed.indexes["time"]).__name__ == "CFTimeIndex")
outm = aidx.aai(fixed, base=("1981", "1982"))
during_m = outm["aai"].isel(time=5).item()
last_m = outm["aai"].isel(time=-1).item()
check(f"[reconciled] AAI in baseline period == 0 (got {during_m:.6f})", abs(during_m) < 1e-9)
check(f"[reconciled] AAI after perturbation == (1-f)*1.5 (got {last_m:.6f})", abs(last_m - AAI_PERT) < 1e-9)

# (c) genuinely incompatible calendars are refused with a clear, named error
nl = ssp.assign_coords(time=_cftrange("1983-01-01", periods=12, freq="MS", calendar="noleap"))
d3 = ssp.assign_coords(time=_cftrange("1985-01-01", periods=12, freq="MS", calendar="360_day"))
refused = False
try:
    pilot._reconcile_time([nl, d3], "ssp585")
except ValueError as e:
    refused = "ssp585" in str(e)
check("incompatible calendars (noleap vs 360_day) raise a ValueError naming the experiment", refused)

# (d) the MIROC6 case: historical 'proleptic_gregorian' + SSP 'standard' name the
# SAME calendar for post-1582 dates -> must reconcile (not refuse) and give 1.5
hp = aidx.standardize(xr.DataArray(
    np.broadcast_to(bb, (24,) + bb.shape).copy(),
    coords={"time": _cftrange("1981-01-01", periods=24, freq="MS", calendar="proleptic_gregorian"),
            "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="tas"))
sp = aidx.standardize(xr.DataArray(
    np.broadcast_to(sv, (12,) + sv.shape).copy(),
    coords={"time": _cftrange("1983-01-01", periods=12, freq="MS", calendar="standard"),
            "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="tas"))
greg = pilot._reconcile_time([hp, sp], "ssp126")
merged = xr.concat(greg, dim="time").sortby("time").drop_duplicates("time")
check(f"standard+proleptic reconcile to one CFTimeIndex (cal={merged['time'].dt.calendar})",
      type(merged.indexes["time"]).__name__ == "CFTimeIndex")
outg = aidx.aai(merged, base=("1981", "1982"))
check(f"[greg-family] AAI after perturbation == (1-f)*1.5 (got {outg['aai'].isel(time=-1).item():.6f})",
      abs(outg["aai"].isel(time=-1).item() - AAI_PERT) < 1e-9)

print("\nAll index-engine + calendar-decode tests passed.")

print("\n9) Availability-scan selection logic (synthetic mini-catalog)")
import arcticidx_scan as scan

def rows(model, exp, member, variables, table="Amon"):
    return [dict(activity_id="x", institution_id="x", source_id=model,
                 experiment_id=exp, member_id=member, table_id=table,
                 variable_id=v, grid_label="gn", zstore="s", version=1)
            for v in variables]

FOUR = ["tas", "psl", "zg", "ua"]
cat = []
# A: r1i1p1f1 complete everywhere + areacella -> in, 4 SSPs
for e in ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]:
    cat += rows("A", e, "r1i1p1f1", FOUR)
cat += rows("A", "historical", "r1i1p1f1", ["areacella"], table="fx")
# B: only r2i1p1f1 complete in historical; ssp245+ssp585 complete,
#    ssp126 missing ua -> in, member r2i1p1f1, 2 SSPs, no areacella
cat += rows("B", "historical", "r2i1p1f1", FOUR)
cat += rows("B", "ssp245", "r2i1p1f1", FOUR)
cat += rows("B", "ssp585", "r2i1p1f1", FOUR)
cat += rows("B", "ssp126", "r2i1p1f1", ["tas", "psl", "zg"])
# C: historical missing ua -> excluded
cat += rows("C", "historical", "r1i1p1f1", ["tas", "psl", "zg"])
cat += rows("C", "ssp585", "r1i1p1f1", FOUR)
# D: complete historical, no SSP at all -> excluded
cat += rows("D", "historical", "r1i1p1f1", FOUR)
# F: members r10 and r2 complete in historical (no r1i1p1f1);
#    ssp370 complete only for r2 -> must pick r2 (numeric sort), 1 SSP
cat += rows("F", "historical", "r10i1p1f1", FOUR)
cat += rows("F", "historical", "r2i1p1f1", FOUR)
cat += rows("F", "ssp370", "r2i1p1f1", FOUR)

m = scan.build_matrix(pd.DataFrame(cat)).set_index("model")
check(f"included set == {{A, B, F}} (got {sorted(m.index)})",
      sorted(m.index) == ["A", "B", "F"])
check("A: member r1i1p1f1, 4 SSPs, areacella True",
      m.loc["A", "member"] == "r1i1p1f1" and m.loc["A", "n_ssps"] == 4
      and bool(m.loc["A", "areacella"]))
check("B: member r2i1p1f1, ssp245+ssp585 only, areacella False",
      m.loc["B", "member"] == "r2i1p1f1" and m.loc["B", "n_ssps"] == 2
      and bool(m.loc["B", "ssp245"]) and bool(m.loc["B", "ssp585"])
      and not bool(m.loc["B", "ssp126"]) and not bool(m.loc["B", "areacella"]))
check("F: numeric member sort picks r2i1p1f1 over r10i1p1f1",
      m.loc["F", "member"] == "r2i1p1f1" and m.loc["F", "n_ssps"] == 1)

print("\n10) Time-coverage guard flags an inter-field gap (silent-NaN prevention)")
# CanESM5 ssp126 shipped ua only for 2101-2300, so its 2015-2100 slice was empty
# and PVI became all-NaN via Dataset alignment. _warn_time_gaps must catch that.
import io
import contextlib
import pilot_one_model as pilot


def _fld(n):
    t = pd.date_range("2015-01-01", periods=n, freq="MS")
    return xr.DataArray(np.zeros((n, 2, 2)),
                        coords={"time": t, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
                        dims=("time", "lat", "lon"))


aligned = {"zg": _fld(1032), "ua": _fld(1032), "psl": _fld(1032), "tas": _fld(1032)}
with contextlib.redirect_stderr(io.StringIO()) as _e:
    r_aligned = pilot._warn_time_gaps(aligned, "ssp245")
check("no gap + no warning when all four inputs share coverage",
      r_aligned == [] and "WARNING" not in _e.getvalue())

empty_ua = {"zg": _fld(1032), "ua": _fld(0), "psl": _fld(1032), "tas": _fld(1032)}
with contextlib.redirect_stderr(io.StringIO()) as _e:
    r_empty = pilot._warn_time_gaps(empty_ua, "ssp126")
check("empty ua gap detected + warned (the CanESM5 case)",
      len(r_empty) == 1 and r_empty[0]["field"] == "ua" and r_empty[0]["have"] == 0
      and "PVI" in r_empty[0]["indices"]
      and "WARNING" in _e.getvalue() and "PVI" in _e.getvalue())

partial_ua = {"zg": _fld(1032), "ua": _fld(600), "psl": _fld(1032), "tas": _fld(1032)}
with contextlib.redirect_stderr(io.StringIO()):
    r_part = pilot._warn_time_gaps(partial_ua, "ssp126")
check("partial ua gap detected (600/1032)",
      len(r_part) == 1 and r_part[0]["have"] == 600 and r_part[0]["expected"] == 1032)

short_all = {"zg": _fld(1020), "ua": _fld(1020), "psl": _fld(1020), "tas": _fld(1020)}
with contextlib.redirect_stderr(io.StringIO()):
    r_short = pilot._warn_time_gaps(short_all, "ssp126")
check("consistently-short coverage is NOT flagged (the CAMS-CSM1-0 case)", r_short == [])

print("\n11) areacella lat float-mismatch is grid-aligned, not silently inner-joined")
# Reproduces the production bug: a model's areacella (fx) store writes latitudes in a
# slightly different float representation than its data store, though it is the SAME
# grid. Left to xarray, .weighted(areacella) inner-joins on lat and silently drops the
# mismatched cells (56 of 96 latitudes for MPI-ESM1-2-LR -> a ~7 K global-tas skew);
# aidx._weights must reassign the data grid onto the equal-sized cell-area field.
lat = np.arange(-89.5, 90, 1.0); lon = np.arange(-180.0, 180, 1.0)
fld = aidx.standardize(make_field(lambda la, lo: la, name="zg"))     # cell value = latitude
la_c, _ = np.meshgrid(lat, lon, indexing="ij")
area_vals = np.cos(np.deg2rad(la_c))                                 # a plausible areacella
area_off = xr.DataArray(area_vals, dims=("lat", "lon"),             # SAME grid, lat nudged
                        coords={"lat": lat + 1e-9, "lon": lon})     # by a tiny float offset
area_aln = xr.DataArray(area_vals, dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})

# (a) the hazard: a raw .weighted() with the offset coords inner-joins to nothing, so
# the GBI-box weighted mean silently collapses (all box cells dropped -> NaN).
sub = fld.sel(lat=slice(60, 80), lon=slice(-80, -20)).isel(time=0)
ca_off = area_off.sel(lat=slice(60, 80), lon=slice(-80, -20))
try:
    raw = float(sub.weighted(ca_off).mean(("lat", "lon")).item())
except Exception:
    raw = float("nan")
check(f"raw .weighted(offset areacella) silently corrupts the box mean (got {raw})",
      not np.isfinite(raw))

# (b) the fix: box_index aligns the cell area onto the data grid, so the offset
# areacella gives EXACTLY the same, finite result as the perfectly-aligned one.
gbi_off = aidx.box_index(fld, "GBI", cell_area=area_off).isel(time=0).item()
gbi_aln = aidx.box_index(fld, "GBI", cell_area=area_aln).isel(time=0).item()
check(f"box_index aligns offset areacella == aligned areacella "
      f"({gbi_off:.6f} vs {gbi_aln:.6f})",
      np.isfinite(gbi_off) and abs(gbi_off - gbi_aln) < 1e-9)

# (c) a genuinely different grid SIZE is refused with a ValueError naming the index.
area_wrong = area_aln.isel(lat=slice(0, 100))                        # 100 lats, none >= 60N
refused = False
try:
    aidx.box_index(fld, "GBI", cell_area=area_wrong)
except ValueError as e:
    refused = "GBI" in str(e)
check("mismatched areacella grid SIZE raises a ValueError naming the index", refused)

print("\nAll tests passed — index engine AND scan logic verified.")
