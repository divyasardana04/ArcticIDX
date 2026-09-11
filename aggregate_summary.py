"""Rebuild ONE combined run summary from the files on disk, with a per-index NaN
scan.

Batched runs (production_run.py --models ...) each overwrite run_summary.csv with
only that batch's models, so after batching there is no single full summary. This
scans every n_ssps==4 model's output folder (the actual .nc on disk are the source
of truth), writes the complete run_summary.csv INCLUDING a machine-readable
``nan_gaps`` column (``experiment:index=nan/total`` entries), and prints a table of
every index series that is entirely or partially NaN — so a data gap is recorded in
the summary itself, not only in a run log.

    python aggregate_summary.py
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

MODELS_CSV = "models_included.csv"
EXPERIMENTS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
INDEX_VARS = ["GBI", "UBI", "BHI", "SHI", "PVI", "AAI"]
OUT = "run_summary.csv"

models = pd.read_csv(MODELS_CSV).query("n_ssps == 4")["model"].tolist()

records = []
gap_rows = []          # (model, experiment, index, nan, n_time, kind) for NaN>0
for m in models:
    got, weighting, rows, model_gaps = [], "", {}, []
    for e in EXPERIMENTS:
        nc = os.path.join(m, f"ArcticIDX_{m}_{e}.nc")
        csv = os.path.join(m, f"ArcticIDX_{m}_{e}.csv")
        if not (os.path.exists(nc) and os.path.exists(csv)):
            continue
        got.append(e)
        try:
            ds = xr.open_dataset(nc)
        except Exception as err:                       # corrupt / half-written
            gap_rows.append((m, e, "<read-error>", -1, -1, type(err).__name__))
            model_gaps.append(f"{e}:READ_ERROR")
            continue
        nt = int(ds["time"].size)
        rows[e] = nt
        weighting = ds.attrs.get("area_weighting", weighting) or weighting
        for v in INDEX_VARS:
            if v in ds:
                nan = int(np.isnan(ds[v].values).sum())
                if nan:
                    kind = "FULL" if nan == nt else "partial"
                    gap_rows.append((m, e, v, nan, nt, kind))
                    model_gaps.append(f"{e}:{v}={nan}/{nt}")
        ds.close()
    n = len(got)
    records.append({
        "model": m,
        "weighting": weighting,
        "n_experiments": n,
        "experiments": ";".join(got),
        "monthly_rows": ";".join(f"{k}={rows[k]}" for k in EXPERIMENTS if k in rows),
        "nan_gaps": ";".join(model_gaps),              # machine-readable per-model note
        "status": "ok" if n == 5 else ("partial" if n else "missing"),
    })

df = pd.DataFrame(records)
df.to_csv(OUT, index=False)

# --------------------------------------------------------------------- report
full_models = df[df.n_experiments == 5]
print(f"Models with ALL FIVE experiments: {len(full_models)} / {len(df)}")
print(f"Wrote {OUT} (machine-readable per-model 'nan_gaps' column)\n")

print("NaN re-scan — every index series that is entirely or partially NaN:")
if gap_rows:
    print(f"  {'model':<16}{'experiment':<12}{'index':<9}{'nan/total':>12}   kind")
    print("  " + "-" * 54)
    for m, e, v, nan, nt, kind in gap_rows:
        print(f"  {m:<16}{e:<12}{v:<9}{f'{nan}/{nt}':>12}   {kind}")
    affected = sorted({r[0] for r in gap_rows})
    print(f"\n  {len(gap_rows)} index series affected across {len(affected)} model(s): "
          f"{', '.join(affected)}")
else:
    print("  NONE — every index in every experiment of every model is fully populated.")
