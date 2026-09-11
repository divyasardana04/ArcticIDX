"""production_run.py -- scale the ArcticIDX pipeline to the full CMIP6 ensemble.

Reuses the pilot engine (pilot_one_model.py) and the tested index math
(arcticidx_indices.py) WITHOUT rewriting any of it. Per model it just:
  * sets pilot.MODEL / pilot.MEMBER (and derived_products' copies),
  * pre-filters the catalog to a single chosen grid_label (so all four
    variables + areacella come from the same grid), and
  * calls pilot.compute_indices / enrich_metadata / write_csv and the
    derived_products aggregators, writing everything into a per-model folder.

Model selection : models_included.csv rows with n_ssps == 4 (expect 30).
Grid selection  : prefer 'gn', else 'gr', else 'gr1'; recorded per model.
Robustness      : per-(model, experiment) try/except -> failures.log, continue.
Resumable       : skip any model/experiment whose output files already exist.
Outputs         : <model>/ArcticIDX_<model>_<exp>.{nc,csv} (monthly) plus the
                  seasonal/annual derived products, and a top-level
                  run_summary.csv.

Usage:
  python production_run.py --audit           # selection + grid audit, no streaming
  python production_run.py --dry-run         # only the 3 dry-run models
  python production_run.py --models A B C     # specific models
  python production_run.py                    # the full 30
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

import pilot_one_model as pilot
import derived_products as dp

MODELS_CSV = "models_included.csv"
AMON_VARS = ["tas", "psl", "zg", "ua"]
EXPERIMENTS = ["historical"] + pilot.SSPS          # historical first (SSP baseline/splice)
GRID_PREFERENCE = ["gn", "gr", "gr1"]
INDEX_VARS = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]
FAILURES_LOG = "failures.log"
SUMMARY_CSV = "run_summary.csv"
# 3 models for the dry run: the finished pilot (tests resume/skip), a clean gn
# model, and a gr1 model (tests the grid fallback).
DRY_RUN_MODELS = ["MPI-ESM1-2-LR", "ACCESS-CM2", "GFDL-ESM4"]


# --------------------------------------------------------------- model selection
def select_models() -> pd.DataFrame:
    matrix = pd.read_csv(MODELS_CSV)
    keep = matrix[matrix["n_ssps"] == 4].reset_index(drop=True)
    print(f"Models with all four SSPs (n_ssps == 4): {len(keep)}")
    for i, r in keep.iterrows():
        print(f"  {i + 1:>2}. {r['model']:<20} member={r['member']}  areacella={r['areacella']}")
    return keep[["model", "member", "areacella"]]


# ------------------------------------------------------------------- grid audit
def _var_grids(cat: pd.DataFrame, model: str, member: str,
               experiment: str, variable: str) -> set:
    rows = cat[
        (cat["source_id"] == model)
        & (cat["member_id"] == member)
        & (cat["table_id"] == "Amon")
        & (cat["variable_id"] == variable)
        & (cat["experiment_id"] == experiment)
    ]
    return set(rows["grid_label"].unique())


def choose_grid(cat: pd.DataFrame, model: str, member: str):
    """Return (chosen_grid, grids_per_var, common_grids, mixed_flag) using the
    historical grids of the four Amon variables."""
    gpv = {v: _var_grids(cat, model, member, "historical", v) for v in AMON_VARS}
    common = set.intersection(*gpv.values()) if all(gpv.values()) else set()
    chosen = next((g for g in GRID_PREFERENCE if g in common), None)
    if chosen is None and common:                 # a common grid, but not gn/gr/gr1
        chosen = sorted(common)[0]
    mixed = len(common) == 0                       # no single grid holds all four
    return chosen, gpv, common, mixed


def grid_audit(cat: pd.DataFrame, models_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("GRID-LABEL AUDIT (historical; tas/psl/zg/ua)")
    print("=" * 78)
    audit = {}
    flagged = []
    for _, r in models_df.iterrows():
        model, member = r["model"], r["member"]
        chosen, gpv, common, mixed = choose_grid(cat, model, member)
        audit[model] = {"member": member, "chosen": chosen, "gpv": gpv,
                        "common": common, "mixed": mixed}
        detail = " ".join(f"{v}={sorted(gpv[v]) or '[MISSING]'}" for v in AMON_VARS)
        tag = "OK"
        if mixed:
            tag = "*** MIXED: no single grid has all four ***"
        elif chosen not in GRID_PREFERENCE:
            tag = f"*** UNUSUAL GRID: {chosen} ***"
        if tag != "OK":
            flagged.append(model)
        print(f"{model:<20} ({member}) -> chosen={str(chosen):<4} [{tag}]")
        print(f"     {detail}")
    print("-" * 78)
    if flagged:
        print(f"FLAGGED ({len(flagged)}): {', '.join(flagged)}")
    else:
        print("No mixed/unusual grids: every model has all four variables on one grid.")
    print("=" * 78)
    return audit


# ------------------------------------------------------------- helpers / logging
def log_failure(model: str, experiment: str, err: BaseException) -> None:
    with open(FAILURES_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {model} / {experiment} =====\n")
        f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))


def monthly_paths(outdir: str, model: str, exp: str) -> tuple[str, str]:
    return (os.path.join(outdir, f"ArcticIDX_{model}_{exp}.nc"),
            os.path.join(outdir, f"ArcticIDX_{model}_{exp}.csv"))


def _nan_count(ds: xr.Dataset) -> int:
    return int(sum(int(np.isnan(ds[v].values).sum()) for v in INDEX_VARS if v in ds))


def sub_catalog(cat: pd.DataFrame, model: str, grid: str) -> pd.DataFrame:
    """All catalog rows for this model on the chosen grid (Amon + fx areacella),
    so every variable and the cell-area field share one grid."""
    return cat[(cat["source_id"] == model) & (cat["grid_label"] == grid)]


# ------------------------------------------------------------- per-model driver
def process_model(idx: int, total: int, model: str, member: str,
                  cat: pd.DataFrame, fs: gcsfs.GCSFileSystem) -> dict:
    outdir = model
    os.makedirs(outdir, exist_ok=True)
    chosen, _, _, mixed = choose_grid(cat, model, member)
    if chosen is None:
        raise RuntimeError(f"no usable grid_label (mixed={mixed}) for {model}")

    subcat = sub_catalog(cat, model, chosen)
    pilot.MODEL = dp.MODEL = model
    pilot.MEMBER = dp.MEMBER = member

    # weighting source (areacella on the chosen grid, else cos-lat)
    try:
        area = pilot.open_areacella(subcat, fs)
    except Exception:
        area = None
    weighting = "areacella" if area is not None else "cos-lat"

    summary = {"model": model, "member": member, "grid_label": chosen,
               "weighting": weighting, "succeeded": [], "rows": {}, "nans": {}}

    # ---- monthly, per experiment (streamed; resumable) ----
    for exp in EXPERIMENTS:
        t0 = time.time()
        nc, csv = monthly_paths(outdir, model, exp)
        try:
            if os.path.exists(nc) and os.path.exists(csv):
                ds = xr.open_dataset(nc)
                status = "skipped (exists)"
            else:
                ds = pilot.compute_indices(exp, subcat, fs, area)
                ds.attrs["area_weighting"] = weighting
                ds = pilot.enrich_metadata(ds, model, member, exp)
                ds.to_netcdf(nc)
                pilot.write_csv(ds, csv)
                status = "done"
            summary["succeeded"].append(exp)
            summary["rows"][exp] = int(ds["time"].size)
            summary["nans"][exp] = _nan_count(ds)
            ds.close()
        except Exception as err:                       # one experiment can't kill the run
            log_failure(model, exp, err)
            status = f"failed ({type(err).__name__})"
        print(f"model {idx} of {total}: {model} - {exp} - {status} ({time.time() - t0:.1f}s)")

    # ---- derived products (local; individual + spliced) ----
    try:
        run_derived(model, member, outdir, summary["succeeded"], weighting)
    except Exception as err:
        log_failure(model, "derived", err)
        print(f"model {idx} of {total}: {model} - derived - failed ({type(err).__name__})")

    return summary


def run_derived(model: str, member: str, outdir: str,
                succeeded: list, weighting: str) -> None:
    """Seasonal + annual products for each succeeded experiment, plus the spliced
    historical+SSP series. Reuses derived_products aggregators; resumable."""
    dp.MODEL, dp.MEMBER = model, member
    exps = [e for e in EXPERIMENTS if e in succeeded]
    cache: dict[str, xr.Dataset] = {}

    def monthly(e: str) -> xr.Dataset:
        if e not in cache:
            cache[e] = dp.load_monthly(e, outdir=outdir)
        return cache[e]

    def make(name: str, source: xr.Dataset) -> None:
        for suffix, fn, freq in [("seasonal", dp.seasonal_means, "seasonal"),
                                 ("annual", dp.annual_means, "annual")]:
            out_nc = os.path.join(outdir, f"ArcticIDX_{model}_{name}_{suffix}.nc")
            if os.path.exists(out_nc):
                continue                                # resumable
            prod, _ = fn(source)
            dp.save_product(prod, name, freq, suffix, weighting, outdir=outdir)

    for e in exps:                                      # individual experiments
        make(e, monthly(e))
    if "historical" in exps:                            # spliced 1950-2100 series
        for ssp in [s for s in pilot.SSPS if s in exps]:
            spliced = (
                xr.concat([monthly("historical"), monthly(ssp)], dim="time")
                .sortby("time").drop_duplicates("time")
            )
            make(f"historical-{ssp}", spliced)


# ------------------------------------------------------------------- run summary
def write_summary(rows: list) -> None:
    def compact(d: dict) -> str:
        return ";".join(f"{k}={d[k]}" for k in EXPERIMENTS if k in d)

    records = []
    for s in rows:
        n_ok = len(s["succeeded"])
        records.append({
            "model": s["model"],
            "member": s["member"],
            "grid_label": s["grid_label"],
            "weighting": s["weighting"],
            "n_experiments_succeeded": n_ok,
            "experiments_succeeded": ";".join(s["succeeded"]),
            "monthly_rows": compact(s["rows"]),
            "nan_counts": compact(s["nans"]),
            "status": "ok" if n_ok == len(EXPERIMENTS) else ("partial" if n_ok else "failed"),
        })
    pd.DataFrame(records).to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV} ({len(records)} models).")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="print selection + grid audit, then exit")
    ap.add_argument("--dry-run", action="store_true", help="process only the 3 dry-run models")
    ap.add_argument("--models", nargs="+", help="explicit list of models to process")
    args = ap.parse_args()

    models_df = select_models()
    cat = pilot.load_catalog()
    audit = grid_audit(cat, models_df)

    if args.audit:
        return

    if args.models:
        chosen = args.models
    elif args.dry_run:
        chosen = DRY_RUN_MODELS
        print(f"\nDRY RUN -- processing only: {', '.join(chosen)}")
    else:
        chosen = list(models_df["model"])

    todo = models_df[models_df["model"].isin(chosen)].reset_index(drop=True)
    missing = [m for m in chosen if m not in set(models_df["model"])]
    if missing:
        print(f"WARNING: requested models not in the n_ssps==4 set, skipped: {missing}")

    fs = gcsfs.GCSFileSystem(token="anon")
    total = len(todo)
    print(f"\nProcessing {total} model(s). Mixed/unusual grids flagged above.\n")

    summaries = []
    for i, r in todo.iterrows():
        model, member = r["model"], r["member"]
        t0 = time.time()
        try:
            summaries.append(process_model(i + 1, total, model, member, cat, fs))
            print(f"model {i + 1} of {total}: {model} - MODEL COMPLETE ({time.time() - t0:.1f}s)\n")
        except Exception as err:
            log_failure(model, "SETUP", err)
            summaries.append({"model": model, "member": member,
                              "grid_label": audit.get(model, {}).get("chosen"),
                              "weighting": "n/a", "succeeded": [], "rows": {}, "nans": {}})
            print(f"model {i + 1} of {total}: {model} - SETUP FAILED - {type(err).__name__} "
                  f"({time.time() - t0:.1f}s)\n")

    write_summary(summaries)


if __name__ == "__main__":
    main()
