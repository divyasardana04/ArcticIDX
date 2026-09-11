# ArcticIDX — Arctic atmospheric index dataset (CMIP6 + ERA5)

ArcticIDX is a dataset of six standard Arctic atmospheric circulation indices,
computed consistently for **30 CMIP6 models** (one ensemble member each) and for
**ERA5** reanalysis. Each index is provided as a **monthly** series plus derived
**seasonal** (DJF/MAM/JJA/SON) and **annual** products, over the **historical**
experiment (1950–2014) and the four Tier‑1 scenarios **SSP1‑2.6, SSP2‑4.5,
SSP3‑7.0, SSP5‑8.5** (2015–2100). ERA5 covers 1950–2024.

Indices are area‑weighted spatial means on each source's native grid
(`areacella` cell areas where available, cosine‑latitude otherwise), with the
longitude convention harmonised to `[-180, 180)`.

## The six indices

| Index | Long name | Definition (domain / field) | Units | Reference |
|-------|-----------|-----------------------------|-------|-----------|
| **GBI** | Greenland Blocking Index | mean 500 hPa geopotential height, 60–80°N, 20–80°W | m | Hanna et al. (2013) |
| **UBI** | Ural Blocking Index | mean 500 hPa geopotential height, 45–80°N, 10°W–80°E | m | Peings (2019) |
| **BHI** | Beaufort High Index | mean sea‑level pressure, 72.5–82.5°N, 180–120°W | hPa | Serreze & Barrett (2011) |
| **SHI** | Siberian High Index | mean sea‑level pressure, 40–65°N, 80–120°E | hPa | Panagiotopoulos et al. (2005) |
| **PVI** | Polar Vortex Index | zonal‑mean zonal wind at 10 hPa, 60°N (meaningful Nov–Mar) | m s⁻¹ | Charlton & Polvani (2007) |
| **AAI** | Arctic Amplification Index | Arctic (poleward of 70°N) minus **global**‑mean 2 m temperature anomaly, relative to the 1981–2010 monthly climatology | K | Liu et al. (2020) |

The AAI is the true Arctic‑minus‑**global** form. Because its 1981–2010 mean is
zero by construction, each output also ships the absolute auxiliary series
`tas_arctic`, `tas_rest`, and `tas_global`, so users can reconstruct either the
Arctic‑minus‑global or the Arctic‑minus‑rest‑of‑globe formulation
(`AAI_global = (1 − f) · AAI_rest`, with `f` the area fraction poleward of 70°N).

**Note**: the BOXES dictionary in arcticidx_indices.py defines the spatial domains for GBI, UBI, BHI and SHI, and PVI_LAT / ARCTIC_LAT define the PVI latitude and the Arctic cap. These constants are the definitive source for the index domains and match the definitions given in the paper.

## Dataset layout

```
<model>/ArcticIDX_<model>_<experiment>.{nc,csv}            # monthly
<model>/ArcticIDX_<model>_<experiment>_{annual,seasonal}.{nc,csv}
<model>/ArcticIDX_<model>_historical-<ssp>_{annual,seasonal}.{nc,csv}  # spliced 1950–2100
ERA5/ArcticIDX_ERA5_reanalysis{,_annual,_seasonal}.{nc,csv}
models_included.csv                                        # the 30 models, member, SSPs, areacella
run_summary.csv                                            # rows/NaN summary per model
```
Every NetCDF carries `GBI, UBI, BHI, SHI, PVI, AAI` plus `tas_arctic, tas_rest,
tas_global`. CSVs are the same series with a readable time index.



## Quick start (xarray)

```python
import xarray as xr

# monthly indices for one model + experiment
ds = xr.open_dataset("MPI-ESM1-2-LR/ArcticIDX_MPI-ESM1-2-LR_historical.nc")
print(list(ds.data_vars))          # GBI, UBI, BHI, SHI, PVI, AAI, tas_arctic, tas_rest, tas_global

gbi = ds["GBI"]                    # monthly Greenland Blocking Index (m)
djf_pvi = ds["PVI"].sel(time=ds["time"].dt.season == "DJF")   # winter PVI

# reconstruct the AAI physical basis (Arctic minus global 2 m temperature)
contrast = ds["tas_arctic"] - ds["tas_global"]

# derived products (annual means, spliced historical+scenario 1950–2100)
ann = xr.open_dataset("MPI-ESM1-2-LR/ArcticIDX_MPI-ESM1-2-LR_historical-ssp585_annual.nc")
```
markdown
## Requirements

Python 3.13. Create an environment and install the dependencies:

**Windows**

python -m venv .venv
..venv\Scripts\Activate.ps1
pip install -r requirements.txt


**Linux / macOS**

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


`cartopy` is needed only for the Figure 1 domain map; the core pipeline runs without it.

