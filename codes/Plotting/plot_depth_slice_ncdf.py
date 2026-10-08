"""
extract_netcdf_depthslice.py

Extract horizontal depth slices from NetCDF tomography models.

Designed for NetCDF3 Classic tomography models such as:
    - SPiRaL 1.4
    - other EMC visualization products

Outputs a numpy dictionary compatible with existing tomography plotting codes.

By Nico Valencia
August 2026
"""

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import interp1d


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to NetCDF tomography model
MODEL_FILE = (
    "/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SPiRaL_slice_data/SPiRaL-1.4.Interpolated.Flattened-viz-only.r0.0.nc"
)

# Output naming
MODEL_NAME = "SPiRaL"

# Parameter to extract
#
# SPiRaL options:
#   vsv
#   vsh
#   vpv
#   vph
#   eta
#
PARAMETER = "vsv"

# Desired depth (km)
DEPTH_KM = 2800.0

# Interpolate between depth samples?
#
# True:
#     interpolate to exact depth
#
# False:
#     use closest model layer
#
INTERPOLATE = True


OUTPUT_FILE = (
    f"/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SPiRaL_slice_data/{MODEL_NAME}_{PARAMETER}_depthslice_{int(DEPTH_KM)}km.npy"
)

# =============================================================================
# LOAD NETCDF FILE
# =============================================================================

print("=" * 60)
print("Opening tomography model")
print("=" * 60)

nc = netcdf_file(
    MODEL_FILE,
    mode="r"
)


# =============================================================================
# READ COORDINATES
# =============================================================================

depth = nc.variables["depth"].data.copy()
latitude = nc.variables["latitude"].data.copy()
longitude = nc.variables["longitude"].data.copy()


print(f"Depth samples : {len(depth)}")
print(f"Latitude grid : {len(latitude)}")
print(f"Longitude grid: {len(longitude)}")


# =============================================================================
# CHECK PARAMETER
# =============================================================================

if PARAMETER not in nc.variables:

    print("\nAvailable variables:")

    for var in nc.variables:
        print("   ", var)

    raise ValueError(
        f"\nParameter '{PARAMETER}' not found."
    )


# =============================================================================
# LOAD TOMOGRAPHY PARAMETER
# =============================================================================

print()
print(f"Loading parameter: {PARAMETER}")

model_var = nc.variables[PARAMETER]

values3d = model_var.data.copy()


# units
try:
    units = model_var.units.decode()

except:
    try:
        units = model_var.units
    except:
        units = "unknown"


# fill value handling
try:
    fill_value = model_var._FillValue

    values3d[values3d == fill_value] = np.nan

except:
    pass


nc.close()


# =============================================================================
# EXTRACT DEPTH SLICE
# =============================================================================

print()
print(f"Requested depth: {DEPTH_KM:.1f} km")


if INTERPOLATE:

    print("Interpolating between depth layers")

    interpolator = interp1d(
        depth,
        values3d,
        axis=0,
        bounds_error=False,
        fill_value=np.nan
    )

    values = interpolator(
        DEPTH_KM
    )


else:

    print("Using nearest model depth")

    depth_index = np.argmin(
        np.abs(depth - DEPTH_KM)
    )

    actual_depth = depth[depth_index]

    print(
        f"Nearest depth layer: {actual_depth:.1f} km"
    )

    values = values3d[depth_index]


# =============================================================================
# BUILD MAP GRID
# =============================================================================

lon2, lat2 = np.meshgrid(
    longitude,
    latitude
)


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 60)
print("Depth Slice Summary")
print("=" * 60)

print(f"Model      : {MODEL_NAME}")
print(f"Parameter  : {PARAMETER}")
print(f"Depth      : {DEPTH_KM:.1f} km")
print(f"Units      : {units}")
print(f"Shape      : {values.shape}")

print(
    f"Minimum    : {np.nanmin(values):.5f}"
)

print(
    f"Maximum    : {np.nanmax(values):.5f}"
)

print(
    f"Mean       : {np.nanmean(values):.5f}"
)


# =============================================================================
# SAVE OUTPUT
# =============================================================================

output = {

    "lon": lon2,
    "lat": lat2,

    "values": values,

    "depth": DEPTH_KM,

    "parameter": PARAMETER,

    "units": units,

    "model": MODEL_NAME

}


np.save(
    OUTPUT_FILE,
    output
)


print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Finished.")