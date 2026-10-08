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
from rasterio.enums import Resampling

import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})


'''
WILL BE SWITCHING FROM -180 TO 180 LON --> 0 TO 360 LON
MODULO 360 EVERY LONGITUDE DATA

August 12 2026 
Including bathymetry along with my splits and the literature average splits
'''

#PacificSplittingMapCoordinates
minlon, maxlon = 90.00, 180.00
minlat, maxlat = -55.00, 5.00

# #blob region extent
# minlon, maxlon = 155.00, 166.00
# minlat, maxlat = -51.00, -36.00



slice_depth = 2800 #km

##### Data Paths #####

# measurements that I made
# new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_2.txt'
new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_corrected_jul292026.txt'


# potential CMB D" anisotropy contamination data
cmb_contamination_path      = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/blob_plots/blob_coordinates_obspy_only.txt'

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

blob_df = pd.read_csv(cmb_contamination_path, delimiter=r'\s+', names=['latitude','longitude','evlat','evlon','statlat','statlon','evdp','evphase','evepic','SI'],header=None)
blob_df['longitude'] = pd.to_numeric(blob_df['longitude'], errors='coerce') % 360

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
# stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(new_data_df['Station'])]
# stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not

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

# ============================================================
# GEBCO BATHYMETRY
# ============================================================

# Find GEBCO tiles
tif_files = glob.glob(
    "/work/gcl3/BR/nicolasv03/GEBCO/*.tif"
)

src_files = [rasterio.open(f) for f in tif_files]

print(f"Found {len(src_files)} GEBCO tiles")


# ------------------------------------------------------------
# Downsampling
# ------------------------------------------------------------

scale_factor = 32


# ------------------------------------------------------------
# Determine whether map crosses the dateline
# ------------------------------------------------------------

# Your map uses 0–360° longitude.
# Convert only the upper bound for the special case where
# the requested region actually extends past 180°.

if maxlon <= 180:

    crosses_dateline = False

else:

    crosses_dateline = True


# ------------------------------------------------------------
# Read GEBCO
# ------------------------------------------------------------

if not crosses_dateline:

    print(
        f"Reading GEBCO region: "
        f"{minlon}–{maxlon}°E, "
        f"{minlat}–{maxlat}°N"
    )

    # GEBCO raster uses -180–180 longitude.
    # For a normal region entirely between 0 and 180,
    # these coordinates are already valid.
    bounds = (
        minlon,
        minlat,
        maxlon,
        maxlat
    )

    topo, topo_transform = merge(
        src_files,
        bounds=bounds
    )

    # Remove raster band dimension
    topo = topo[0].astype(float)

    # Downsample
    topo = topo[::scale_factor, ::scale_factor]

    # Replace nodata
    topo[topo <= -32768] = np.nan

    extent = [
        minlon,
        maxlon,
        minlat,
        maxlat
    ]


else:

    print("GEBCO region crosses the dateline")

    # --------------------------------------------------------
    # 180–360° portion
    # --------------------------------------------------------

    minlon_wrap = minlon - 360
    maxlon_wrap = maxlon - 360

    topo_left, transform_left = merge(
        src_files,
        bounds=(
            minlon,
            minlat,
            180,
            maxlat
        )
    )

    topo_left = topo_left[0].astype(float)
    topo_left = topo_left[::scale_factor, ::scale_factor]
    topo_left[topo_left <= -32768] = np.nan


    # --------------------------------------------------------
    # -180–(maxlon-360) portion
    # --------------------------------------------------------

    topo_right, transform_right = merge(
        src_files,
        bounds=(
            -180,
            minlat,
            maxlon_wrap,
            maxlat
        )
    )

    topo_right = topo_right[0].astype(float)
    topo_right = topo_right[::scale_factor, ::scale_factor]
    topo_right[topo_right <= -32768] = np.nan


    extent_left = [
        minlon,
        180,
        minlat,
        maxlat
    ]

    extent_right = [
        -180,
        maxlon_wrap,
        minlat,
        maxlat
    ]


# ------------------------------------------------------------
# Close GEBCO files
# ------------------------------------------------------------

for src in src_files:
    src.close()

print("GEBCO data loaded successfully")



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
gl.xlocator = mticker.FixedLocator([80,100, 120, 140, 160, 180])

# optional latitude ticks
gl.ylocator = mticker.FixedLocator([-60,-50,-40,-30,-20,-10, 0, 10])

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

# ##### choose basemap: lithospheric thickness or bathymetry data or diffSI data



### bathymetry
vmin = -4000   # ocean
vmax = 2000    # land (australia peak elevation is ~2200m)

if not crosses_dateline:
    img = ax.imshow(topo,extent=extent,
                    origin="upper",
                    transform=ccrs.PlateCarree(),
                    cmap="terrain",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.6)
    cbar_bathymetry = plt.colorbar(img, ax=ax, pad=0.05, shrink=0.8)
    cbar_bathymetry.set_label("Bathymetry [m]", fontsize=20)
    cbar_bathymetry.ax.tick_params(labelsize=14)
else:
    img1 = ax.imshow(
        topo_left,
        extent=[minlon_wrap, 180, minlat, maxlat],
        origin="upper",
        transform=ccrs.PlateCarree(),
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        alpha=0.6
    )

    img2 = ax.imshow(
        topo_right,
        extent=[-180, maxlon_wrap, minlat, maxlat],
        origin="upper",
        transform=ccrs.PlateCarree(),
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        alpha=0.6
    )

    cbar_bathymetry = plt.colorbar(img1, ax=ax, pad=0.05, shrink=0.8)
    cbar_bathymetry.set_label("Bathymetry [m]", fontsize=20)
    cbar_bathymetry.ax.tick_params(labelsize=14)    
    
# ## plot the tectonic boundaries 
# for segment in ridge_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# for segment in trench_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# for segment in transform_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# print(blob_df.head(20))


# define colormap + normalization
# cmap = plt.cm.vanimo   # or 'coolwarm', 'plasma', etc.

#####

# # df = new_data_df
for i in range(len(new_data_df)):
    plot_station_splitting(new_data_df['Station'].iloc[i],
                           new_data_df['Latitude'].iloc[i],
                           new_data_df['Longitude'].iloc[i],
                           new_data_df['phi'].iloc[i],
                           4*new_data_df['dt'].iloc[i],
                           ax=ax,
                           version='New',
                           annotate=False
                           )

# for i in range(len(legend_df)):
#     plot_station_splitting(legend_df['Station'].iloc[i],
#                            legend_df['Latitude'].iloc[i],
#                            legend_df['Longitude'].iloc[i],
#                            legend_df['phi'].iloc[i],
#                            4*legend_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Legend')

# # print(stat_avg_df.head(5))
for i in range(len(stat_avg_df)):
    plot_station_splitting(stat_avg_df['station'].iloc[i],
                           stat_avg_df['lat'].iloc[i],
                           stat_avg_df['lon'].iloc[i],
                           stat_avg_df['azi'].iloc[i],
                           4*stat_avg_df['dt'].iloc[i],
                           ax=ax,
                           version='Old',
                           annotate=False)
    
# ### plot station names only
# for i in range(len(new_data_df)):
#     plot_station_splitting(new_data_df['Station'].iloc[i],
#                            new_data_df['Latitude'].iloc[i],
#                            new_data_df['Longitude'].iloc[i],
#                            new_data_df['phi'].iloc[i],
#                            4*new_data_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Dummy',
#                            annotate=True
#                            )



# # Combine both datasets if you want to show all measurements
# combined_df = pd.concat([sks_meas_all_newdata])

# # Set colormap and normalization
# cmap = plt.cm.viridis # Or 'plasma', 'coolwarm', etc.
# norm = Normalize(vmin=0, vmax=1.5)

# # Plot scatter points colored by dt
# sc = ax.scatter(combined_df['Longitude'], combined_df['Latitude'],
#                 c=combined_df['dt'], cmap=cmap, norm=norm,
#                 s=40, edgecolor='k', linewidth=0.5,
#                 transform=ccrs.PlateCarree(), zorder=5)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cax = inset_axes(ax,
#                  width="50%",  # width of the colorbar
#                  height="3%",  # height
#                  loc='lower center',
#                  bbox_to_anchor=(0.5, -0.1, 0.5, 1),  # (x0, y0, width, height)
#                  bbox_transform=ax.transAxes,
#                  borderpad=0)

# # Add a colorbar
# cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.3, pad=0.05)
# cbar.set_label('Delay Time (s)', fontsize=14)

# plt.title(f"CMB Pierce Points w/ SEMUCB-WM1 : {slice_depth} km",fontsize=25)
# plt.title(f"CMB Pierce Points w/ Diff SI",fontsize=25)

fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry_and_splits_map_sep_5.pdf',dpi=600,bbox_inches="tight", pad_inches=0.2) #original dpi=300
