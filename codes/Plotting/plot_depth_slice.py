'''
Using codes from plot_xsec_ortho.py
Use geographical extent of region of interest and specific depth
Find tomography model values at that 2d slice and save it for plotting
Ideally numpy style workflow only, we will see how feasible this will be 

By Nico Valencia and help from ChatGPT & Claude
May 19 2026

'''

import json
import os
import pickle
from obspy.taup import TauPyModel
import copy 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

#########
import sys
path2ucbpy = '/work/range/BR/utpalkumar/FWI/000_bsl_group/FWI_github/ucbpy/python3'
sys.path.insert(0, path2ucbpy)
############

import pyspl
from ModelA3d import ModelA3d
from Model1D import Model1D
from Sphere import delaz, gc_minor, shoot
from UCBColorMaps import cmapSved, cmapXved

from FigTools import plot_plates, plot_hotspots, plot_gc_clean, get_tectonics_path
from circle import circle_dist


########################################################################
# const

r_earth = 6371.0


########################################################################
# config

# parameter to plot (A3d descriptor name)
param = 'S'

# model configuration
model_config = {
    # simply for naming of output files
    'name': 'SEMUCB',
    # A3d file
    'modelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/A3d.dat',
    # 1D reference model (needed for plotting xi sections)
    'refmodelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/model1D.dat',
    # spherical spline grid files
    'gridfiles':  {
        'S': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots.dat',
        'X': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots2.dat'}}

########################################################################

# load the A3d model
model = ModelA3d(model_config['modelfile'])
model.load_from_file()
coefs = model.get_parameter_by_name(param).get_values()

# model = ModelA3d(model_config['modelfile'])
# model.load_from_file()

# coefs = model.get_parameter_by_name(param).get_values()

# ============================================================
# LOAD SPHERICAL SPLINE GRID
# ============================================================

grid = np.loadtxt(
    model_config['gridfiles'][param],
    skiprows=1
)

# spherical spline interpolator
sspl = pyspl.SphericalSplines(
    grid[:,0],
    grid[:,1],
    grid[:,2]
)

# radial spline interpolator
bspl = pyspl.CubicBSplines(
    model.get_bspl_knots()
)

# ============================================================
# SELECT DEPTH
# ============================================================

depth_km = 100.0

print(f"Building depth slice at {depth_km} km")

# convert depth -> radius
r = r_earth - depth_km

# ============================================================
# BUILD LON/LAT GRID
# ============================================================

# start modest for speed
nlon = 720
nlat = 360

lons = np.linspace(-180, 180, nlon)
lats = np.linspace(-80, 80, nlat)

lon2, lat2 = np.meshgrid(lons, lats)

print("Grid shape:", lon2.shape)

# ============================================================
# EVALUATE LATERAL BASIS
# ============================================================

print("Evaluating spherical spline basis...")

# returns sparse matrix:
# shape = (npoints, nbasis)
H = sspl.evaluate(
    lon2.ravel(),
    lat2.ravel()
)

print("H shape:", H.shape)

# ============================================================
# EVALUATE RADIAL BASIS
# ============================================================

print("Evaluating radial basis...")

V = bspl.evaluate(np.array([r]))

print("V shape:", V.shape)

# ============================================================
# BUILD DEPTH SLICE
# ============================================================

print("Combining model coefficients...")

# This is the SAME tensor contraction used in your cross-section code
x = (H * (V * coefs).T).T

print("Raw output shape:", x.shape)

# flatten to 1D field
x = np.asarray(x).ravel()

print("Flattened shape:", x.shape)

# reshape back to map
x = x.reshape(lon2.shape)

# convert to percent
x *= 100.0



# ============================================================
# OPTIONAL: SAVE SLICE
# ============================================================

# np.save(
#     f"/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SEMUCB_slice_data/SEMUCB_depthslice_{int(depth_km)}km.npy",
#     x
# )

np.save(
    f"/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SEMUCB_slice_data/SEMUCB_depthslice_{int(depth_km)}km.npy",
    {
        "lon": lon2,
        "lat": lat2,
        "values": x,
        "depth": depth_km
    }
)
print("Saved numpy slice")

# ============================================================
# PLOT
# ============================================================

# print("Plotting...")

# fig = plt.figure(figsize=(14,7))

# ax = plt.axes(
#     projection=ccrs.PlateCarree()
# )

# img = ax.pcolormesh(
#     lon2,
#     lat2,
#     x,
#     transform=ccrs.PlateCarree(),
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-2.5,
#     vmax=2.5
# )

# ax.coastlines(linewidth=0.8)

# ax.add_feature(
#     cfeature.BORDERS,
#     linewidth=0.3
# )

# gl = ax.gridlines(
#     draw_labels=True,
#     linewidth=0.3,
#     alpha=0.5
# )

# plt.colorbar(
#     img,
#     ax=ax,
#     shrink=0.7,
#     label='dlnVs (%)'
# )

# plt.title(
#     f"SEMUCB dlnVs Depth Slice ({depth_km:.0f} km)"
# )

# plt.tight_layout()

# plt.savefig(
#     f"SEMUCB_depthslice_{int(depth_km)}km.png",
#     dpi=300,
#     bbox_inches='tight'
# )

# plt.show()