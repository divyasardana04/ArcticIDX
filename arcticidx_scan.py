"""
ArcticIDX — CMIP6 availability scan.

Answers: which models can be in the dataset, with which ensemble member,
under which SSPs? Implements the Methods inclusion rule exactly:

  * required monthly (Amon) variables: tas, psl, zg, ua
  * one member per model: r1i1p1f1 preferred, else lowest-numbered member
    that has all four variables in `historical`
  * a model is included if the chosen member also has all four variables
    in at least one Tier-1 SSP
  * areacella availability is recorded (cos-lat fallback otherwise)


"""
from __future__ import annotations

import re

import pandas as pd

CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
REQUIRED_AMON = {"tas", "psl", "zg", "ua"}
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PREFERRED_MEMBER = "r1i1p1f1"


def member_sort_key(member: str) -> tuple:
    """'r10i1p2f1' -> (10, 1, 2, 1) so r2 sorts before r10 (numeric, not
    lexicographic)."""
    nums = re.findall(r"\d+", member)
    return tuple(int(n) for n in nums) if nums else (10**9,)


def build_matrix(cat: pd.DataFrame) -> pd.DataFrame:
    """Apply the inclusion rule to a catalog table.

    Expects the Pangeo CMIP6 catalog schema (source_id, experiment_id,
    member_id, table_id, variable_id, ...). Returns one row per included
    model.
    """
    amon = cat[
        (cat["table_id"] == "Amon")
        & (cat["variable_id"].isin(REQUIRED_AMON))
        & (cat["experiment_id"].isin(["historical"] + SSPS))
    ]
    has_areacella = set(
        cat.loc[cat["variable_id"] == "areacella", "source_id"].unique()
    )

    # variables present per (model, experiment, member)
    grouped = (
        amon.groupby(["source_id", "experiment_id", "member_id"])["variable_id"]
        .agg(lambda v: frozenset(v))
    )

    rows = []
    for model in sorted(amon["source_id"].unique()):
        g = grouped.loc[model] if model in grouped.index.get_level_values(0) else None
        if g is None or "historical" not in g.index.get_level_values(0):
            continue
        hist = g.loc["historical"]
        candidates = [m for m, vs in hist.items() if REQUIRED_AMON <= vs]
        if not candidates:
            continue
        member = (
            PREFERRED_MEMBER
            if PREFERRED_MEMBER in candidates
            else sorted(candidates, key=member_sort_key)[0]
        )
        ssp_ok = {}
        for ssp in SSPS:
            ok = False
            if ssp in g.index.get_level_values(0):
                exp = g.loc[ssp]
                ok = member in exp.index and REQUIRED_AMON <= exp.loc[member]
            ssp_ok[ssp] = ok
        if not any(ssp_ok.values()):
            continue
        rows.append(
            {
                "model": model,
                "member": member,
                **ssp_ok,
                "n_ssps": sum(ssp_ok.values()),
                "areacella": model in has_areacella,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("Downloading Pangeo CMIP6 catalog (large CSV, be patient)...")
    cat = pd.read_csv(CATALOG_URL, low_memory=False)
    print(f"Catalog rows: {len(cat):,}")

    matrix = build_matrix(cat)
    matrix.to_csv("models_included.csv", index=False)

    n = len(matrix)
    counts = {ssp: int(matrix[ssp].sum()) for ssp in SSPS}
    all_four = int((matrix["n_ssps"] == 4).sum())
    with_area = int(matrix["areacella"].sum())

    print("\n================ ArcticIDX availability summary ================")
    print(f"Models included (historical + >=1 Tier-1 SSP): {n}")
    for ssp, c in counts.items():
        print(f"  {ssp}: {c} models")
    print(f"Models covering all four SSPs: {all_four}")
    print(f"Models providing areacella:    {with_area} (others: cos-lat weights)")
    print(f"Non-r1i1p1f1 members chosen:   {int((matrix['member'] != PREFERRED_MEMBER).sum())}")
    print("\nMethods-ready sentence:")
    print(
        f"  The resulting ensemble comprises {n} CMIP6 models, of which "
        f"{counts['ssp126']}, {counts['ssp245']}, {counts['ssp370']}, and "
        f"{counts['ssp585']} provide SSP1-2.6, SSP2-4.5, SSP3-7.0, and "
        f"SSP5-8.5, respectively; {all_four} models cover all four scenarios."
    )
    print("\nWrote models_included.csv — send this file back to fill Table 1.")


if __name__ == "__main__":
    main()
