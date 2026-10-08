import pandas as pd
import os, sys, glob
import numpy as np
import enum # Enum for size units
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime as dt
# import pygmt
from math import * #added for error fix
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import sys
from helper_functions.plotting import read_gmt_file, read_and_filter_gmt, plot_station_splitting, build_cmap
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import matplotlib.colors as colors
import re
from support import dms_to_dd
from station_keys import STATION_PHASE_MAP
from scipy.interpolate import griddata

import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

import sys

# Add path to UCB python library
path2ucbpy = '/work/range/BR/utpalkumar/FWI/000_bsl_group/FWI_github/ucbpy/python3'
sys.path.insert(0, path2ucbpy)

# Import UCB colormaps
from UCBColorMaps import cmapSved, cmapXved



'''
WILL BE SWITCHING FROM -180 TO 180 LON --> 0 TO 360 LON
MODULO 360 EVERY LONGITUDE DATA

'''

# #PacificSplittingMapCoordinates
# minlon, maxlon = 90.00, 180.00
# minlat, maxlat = -55.00, 5.00

#arma bounce points extent
minlon, maxlon = 145.00, 180.00
minlat, maxlat = -55.00, -20.00


# #blob region extent
# minlon, maxlon = 155.00, 166.00
# minlat, maxlat = -51.00, -36.00



slice_depth = 2800 #km

##### Data Paths #####

# measurements that I made
new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_2.txt'

# potential CMB D" anisotropy contamination data
cmb_contamination_path      = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/blob_plots/blob_coordinates_ARMA_only.txt'

# Bernhard Steinberger lithospheric thickness model
lithosphere_thickness_path  = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/lithosphere_thickness/lith_ave_no_slabs.d'
                        
# Thorsten Becker SKS station average database (as of May 7 2026)
becker_sks_data_path        = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/sks_stat_avg/jsg_geodynamics.sks.avg.table.txt'

# Plate Boundary Data Bird (2003)
ridge_path      = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/ridge.gmt'       
transform_path  = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/transform.gmt'
trench_path     = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/trench.gmt'

# Bathymetry data from GEBCO
tif_files = glob.glob("/work/gcl3/BR/nicolasv03/GEBCO/*.tif")
src_files = [rasterio.open(f) for f in tif_files]

# SEMUCB slice at certain depth
# tomo_data_path = f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SEMUCB_slice_data/SEMUCB_depthslice_{slice_depth}km.npy'
tomo_data_path = f"/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SPiRaL_slice_data/SPiRaL_vsv_depthslice_{slice_depth}km.npy"

# WOlf 2026 TSR lower mantle diffSI
diffSI_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/wolf_lower_mantle/diff_splitting_grid_resampledforplotting.grd'

# bathymetry_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry/gebco_2026_n6.807_s-50.49_w87.582_e193.256_geotiff.tif'


##### DataFrames #####
                                
new_data_df = pd.read_csv(new_measurements_path, sep='\t', encoding='cp1252') 
new_data_df[['Latitude','Longitude','phi','dt']] = new_data_df[['Latitude','Longitude','phi','dt']].apply(pd.to_numeric, errors='coerce')
new_data_df['Longitude'] = new_data_df['Longitude'] % 360

blob_df = pd.read_csv(cmb_contamination_path, delimiter=r'\s+') #, names=['latitude','longitude','evlat','evlon','statlat','statlon','evdp','evphase','evepic','SI'],header=None)
blob_df['lon'] = pd.to_numeric(blob_df['lon'], errors='coerce') % 360

lithos_df = pd.read_csv(lithosphere_thickness_path, header=None, delimiter=r'\s+', names=['longitude', 'latitude', 'thickness'], low_memory=False)
lithos_df['longitude'] = lithos_df['longitude'] % 360

# Thorsten Becker SKS station average database (as of May 7 2026)
stat_avg_df = pd.read_csv(becker_sks_data_path, header=None, delimiter=r'\s+', names=["station","lat","lon","azi","d_azi","dt","d_dt",
                                                                                      "type","elat","elon","edep","mlat","mlon","bazi"])
stat_avg_df['lon'] = stat_avg_df['lon'] % 360
stat_avg_df['elon'] = stat_avg_df['elon'] % 360

stat_avg_df['station_clean'] = stat_avg_df['station'].str.split('-', n=1).str[0]
stat_avg_df['num_of_splits'] = stat_avg_df['station'].str.extract(r'avg(\d+)').astype(int)
stat_avg_df = stat_avg_df[(minlon < stat_avg_df['lon']) & (stat_avg_df['lon'] < maxlon)]
stat_avg_df = stat_avg_df[(minlat < stat_avg_df['lat']) & (stat_avg_df['lat'] < maxlat)]

# print(stat_avg_df.head(5))
# base_station = (stat_avg_df.loc[5481, 'station'].split('-')[0])
# stat_avg_df = stat_avg_df[(stat_avg_df['station'].str.split('-').str[0] != base_station) | (stat_avg_df.index == 5481)]
stat_avg_df = (stat_avg_df.sort_values('num_of_splits', ascending=False).drop_duplicates(subset='station_clean').reset_index(drop=True))
stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(new_data_df['Station'])]
# stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not

# just for this one, plot only ARMA
stations_to_plot = ["ARMA", "CAN", "CNB", "TOO","YNG"]

new_data_df = new_data_df[new_data_df['Station'].isin(stations_to_plot)]

ridge_gmt       = read_and_filter_gmt(ridge_path,minlon,maxlon,minlat,maxlat, max_gap=20)
transform_gmt   = read_and_filter_gmt(transform_path,minlon,maxlon,minlat,maxlat, max_gap=20)
trench_gmt      = read_and_filter_gmt(trench_path,minlon,maxlon,minlat,maxlat, max_gap=20)

plate_df = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/pm10242.csv',sep=',',header=None,names=['station','lat', 'lon', 'rate','angle','a','b','c'])
# plate_df['lon'] = pd.to_numeric(plate_df['lon'], errors='coerce') % 360
print(plate_df.head(10))
plate_df['lat_dd'] = plate_df['lat'].apply(dms_to_dd)
plate_df['lon_dd'] = plate_df['lon'].apply(dms_to_dd)

# Compute u (east) and v (north) components
plate_df['angle_rad'] = np.deg2rad(plate_df['angle'])

plate_df = plate_df[(plate_df['lat_dd'] >= minlat) & (plate_df['lat_dd'] <= maxlat) &
                  (plate_df['lon_dd'] >= minlon) & (plate_df['lon_dd'] <= maxlon)] #filter within map bounds

plate_df['u'] = plate_df['rate'] * np.sin(plate_df['angle_rad'])  # east component
plate_df['v'] = plate_df['rate'] * np.cos(plate_df['angle_rad'])  # north component

tomo_data = np.load(tomo_data_path, allow_pickle=True).item()
lon2 = tomo_data["lon"]
lat2 = tomo_data["lat"]
slice_vals = tomo_data["values"]
depth_km = tomo_data["depth"]


# diffSI data from Wolf 2026 TSR
diffSI_src = rasterio.open(diffSI_path)
diffSI_data = diffSI_src.read(1).astype(float)
height, width = diffSI_data.shape
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(diffSI_src.transform, rows, cols)
lons = np.array(xs)[0]   # flatten row structure
lats = np.array(ys)[:, 0]

pad = 5.0

window = from_bounds(
    minlon - pad,
    minlat - pad,
    maxlon + pad,
    maxlat + pad,
    transform=diffSI_src.transform
)

diffSI_data = diffSI_src.read(1, window=window).astype(float)

window_transform = diffSI_src.window_transform(window)

h, w = diffSI_data.shape
cols, rows = np.meshgrid(np.arange(w), np.arange(h))
xs, ys = rasterio.transform.xy(window_transform, rows, cols)
lon2d = np.array(xs)
lat2d = np.array(ys)

print("SPiRaL values:", np.nanmin(slice_vals), np.nanmax(slice_vals))

new_data_df = new_data_df[(minlon < new_data_df['Longitude']) & (new_data_df['Longitude'] < maxlon)]
new_data_df = new_data_df[(minlat < new_data_df['Latitude']) & (new_data_df['Latitude'] < maxlat)]

legend_df = pd.DataFrame([{
    'Station': r'$\mathbf{\delta t}$ = 1.0',
    'Latitude': -40,
    'Longitude': 105,
    'phi': 90,
    'dt': 1
}])

# Create a figure and axis for the map
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([minlon,maxlon,minlat,maxlat], crs=ccrs.PlateCarree())
# ax.gridlines(draw_labels=True,linewidth=0)

import matplotlib.ticker as mticker

gl = ax.gridlines(draw_labels=True, linewidth=0.25)

# remove duplicate dateline label
gl.xlocator = mticker.FixedLocator([80,100, 120, 140,145,150,155, 160,165,170,175, 180])

# optional latitude ticks
gl.ylocator = mticker.FixedLocator([-60,-50,-45,-40,-35,-30,-25,-20,-10, 0, 10])

# turn off top/right labels
gl.top_labels = False
gl.right_labels = False

gl.xlabels_top = False
gl.ylabels_right = False

gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}

from matplotlib.ticker import FuncFormatter

def deg_formatter(x, pos):
    return f"{x:.0f}°"

gl.xformatter = FuncFormatter(deg_formatter)
gl.yformatter = FuncFormatter(deg_formatter)

ax.coastlines()

#only use if doing spiral model
reference_vsv = np.nanmean(slice_vals)
dln_vsv = 100 * np.log(slice_vals / reference_vsv)

dln_vsv = np.clip(dln_vsv, -2.5, 2.5)

slice_vals = dln_vsv 


print(f"SPiRaL reference Vsv: {reference_vsv:.4f} km/s")
print(
    f"SPiRaL dlnVsv range: "
    f"{np.nanmin(dln_vsv):.2f} to {np.nanmax(dln_vsv):.2f} %"
)

# Create the UCB colormap for S-wave perturbations
cmap_ucb = cmapSved(41)

im = ax.contourf(
    lon2,
    lat2,
    slice_vals,
    levels = np.linspace(-2.5, 2.5, 101),
    cmap=cmap_ucb,  # ← USE UCB COLORMAP
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

# # cbar = plt.colorbar(im, ax=ax, extend='neither')
# im = ax.contourf(
#     lon2,
#     lat2,
#     slice_vals,
#     levels = np.linspace(-2.5, 2.5, 101),
#     cmap="seismic_r",
#     # extend='both',

#     # shading="auto",
#     # vmin=-2.5,
#     # vmax=2.5,
#     alpha=0.7,
#     transform=ccrs.PlateCarree()
# )


# colorbar
cbar = plt.colorbar(im, ax=ax,orientation='vertical', 
                        pad=0.05, shrink=0.8)
cbar.set_ticks([-2,-1,0,1,2])

# plt.colorbar(contour, ax=ax, orientation='vertical', 
#                               pad=0.05, shrink=0.8)

cbar.set_label("dlnVs (%)",fontsize=20)
cbar.ax.tick_params(labelsize=14)



# scatter plot
sc = ax.scatter(
    blob_df['lon'],
    blob_df['lat'],
    # c='black',          # <-- use c instead of color
    # cmap='PRGn',
    # norm=norm,
    edgecolor='black',
    marker='o',
    facecolors='none',
    s=500,
    alpha=0.7,
    transform=ccrs.PlateCarree(),
    zorder=10
)


#####
spacing = 10  # degrees

# Create bins
plate_df['lat_bin'] = np.floor(plate_df['lat_dd'] / spacing) * spacing
plate_df['lon_bin'] = np.floor(plate_df['lon_dd'] / spacing) * spacing

# Average vectors within bins
plate_df_binned = (
    plate_df
    .groupby(['lat_bin', 'lon_bin'], as_index=False)
    .agg({
        'u': 'mean',
        'v': 'mean',
        'lat_dd': 'mean',
        'lon_dd': 'mean'
    })
)

    
### plot station names only
for i in range(len(new_data_df)):
    plot_station_splitting(new_data_df['Station'].iloc[i],
                           new_data_df['Latitude'].iloc[i],
                           new_data_df['Longitude'].iloc[i],
                           new_data_df['phi'].iloc[i],
                           4*new_data_df['dt'].iloc[i],
                           ax=ax,
                           version='CMB_single_station',
                           annotate=True
                           )




# # Add a colorbar
# cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.3, pad=0.05)
# cbar.set_label('Delay Time (s)', fontsize=14)

# plt.title(f"CMB Pierce Points w/ SEMUCB-WM1 : {slice_depth} km",fontsize=25)
# plt.title(f"CMB Pierce Points w/ Diff SI",fontsize=25)

fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/cmb_tomo_spiral_map_sep_23.pdf', dpi=600,bbox_inches="tight", pad_inches=0.2)
 #original dpi=300

