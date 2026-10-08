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

COPY OF OTHER CUZ I AM LAZY
THIS IS FOR PRAGUE SPLITTING MAPS I MADE

AND ONCE AGAIN ANOTHER COPY TO MAKE FIGURE 6 FOR THE PAPER

will include: 
- my splitting measurements for the stations
- the predicted splitting measurements from surface wave model code that Jonathan provided me

more note from july 6 2026
- used to compare birkey et al 2023 measurements with mine 
- can use birkey's circular or arithmetic mean

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

# events used to make these measurements (including cluster data)
events_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/swp_event_data/eventData_jul29.txt'

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

# # Bathymetry data from GEBCO
# # tif_files = glob.glob("/work/gcl3/BR/nicolasv03/GEBCO/*.tif")
# tif_files = glob.glob("/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry/*.tif")

# src_files = [rasterio.open(f) for f in tif_files]

# SEMUCB slice at certain depth
tomo_data_path = f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SEMUCB_slice_data/SEMUCB_depthslice_{slice_depth}km.npy'

# WOlf 2026 TSR lower mantle diffSI
diffSI_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/wolf_lower_mantle/diff_splitting_grid_resampledforplotting.grd'

# bathymetry_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry/gebco_2026_n6.807_s-50.49_w87.582_e193.256_geotiff.tif'

birkey_data_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/birkey_2023_data/birkey_station_splitting_data.txt'


##### DataFrames #####
                                
new_data_df = pd.read_csv(new_measurements_path, sep='\t', encoding='cp1252') 
new_data_df[['Latitude','Longitude','phi','dt']] = new_data_df[['Latitude','Longitude','phi','dt']].apply(pd.to_numeric, errors='coerce')
new_data_df['Longitude'] = new_data_df['Longitude'] % 360

birkey_data_df = pd.read_csv(birkey_data_path,sep=',')
birkey_data_df = birkey_data_df.merge(new_data_df[['Station', 'Latitude', 'Longitude']].drop_duplicates('Station'), left_on='STA', right_on='Station', how='left')
birkey_data_df = birkey_data_df.dropna(subset=['Station'])
# print(birkey_data_df.head(10))

events_df = pd.read_csv(events_path, sep='\t')
# print(len(events_df))
events_df = events_df[events_df['evstation'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not
# print(len(events_df))

events_df["phase_map"] = events_df["evstation"].map(STATION_PHASE_MAP)
# print(events_df.head(10))

stations_to_plot = (
    events_df[['evstation', 'evnetwork', 'evstationlat', 'evstationlon', 'phase_map']]
    .drop_duplicates()
    .reset_index(drop=True)
)
stations_to_plot = stations_to_plot.drop_duplicates(subset=['evstation'])
stations_to_plot_complex = stations_to_plot[stations_to_plot['phase_map'] == "complex"]


print(stations_to_plot_complex.to_string())
print(len(stations_to_plot_complex))

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


####### using regional map code for this for now ###############
src = rasterio.open('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry/gebco_2026_n6.807_s-50.49_w87.582_e193.256_geotiff.tif')
window = from_bounds(
    minlon, minlat,
    maxlon, maxlat,
    transform=src.transform
)
scale_factor = 8

# h = int(window.height)
# w = int(window.width)

# topo = src.read(
#     1,
#     window=window,
#     out_shape=(h, w),
#     resampling=Resampling.bilinear
# )
topo = src.read(1, window=window,
                out_shape=(
        int(window.height // scale_factor),
        int(window.width // scale_factor)
    ))
topo = np.where(topo == src.nodata, np.nan, topo)
# topo = np.where(topo < 0, 0, topo)
# transform = src.window_transform(window)

extent = [
    minlon,
    maxlon,
    minlat,
    maxlat
]
####################################################



new_data_df = new_data_df[(minlon < new_data_df['Longitude']) & (new_data_df['Longitude'] < maxlon)]
new_data_df = new_data_df[(minlat < new_data_df['Latitude']) & (new_data_df['Latitude'] < maxlat)]

legend_df = pd.DataFrame([{
    'Station': r'$\mathbf{\delta t}$ = 1.0',
    'Latitude': -45,
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

### lithospheric thickness
lon_grid = np.linspace(lithos_df['longitude'].min(), lithos_df['longitude'].max(), 200)
lat_grid = np.linspace(lithos_df['latitude'].min(), lithos_df['latitude'].max(), 200)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

thickness_grid = griddata((lithos_df['longitude'], lithos_df['latitude']),lithos_df['thickness'],
                            (lon_mesh, lat_mesh),method='cubic')

# Create clean, evenly-spaced levels
vmin = lithos_df['thickness'].min()
vmax = lithos_df['thickness'].max()

# Round to nice numbers
vmin = int(np.floor(vmin / 10) * 10)  # Round down to nearest 10
vmax = int(np.ceil(vmax / 10) * 10)   # Round up to nearest 10

levels = np.linspace(vmin, 250, 50)

contour = ax.contourf(lon_mesh, lat_mesh, thickness_grid,
                      levels=levels,
                      cmap='RdYlBu',
                      alpha=0.6,
                      transform=ccrs.PlateCarree(),
                      zorder=1)

cbar_thickness = plt.colorbar(contour, ax=ax, orientation='vertical', 
                              pad=0.05, shrink=0.8)
cbar_thickness.set_label('Lithosphere Thickness [km]', fontsize=20)
cbar_thickness.ax.tick_params(labelsize=14)

# Set fewer, evenly spaced ticks
# cbar_thickness.set_ticks(np.linspace(vmin, vmax, 8))
cbar_thickness.set_ticks([10, 50, 100, 150, 200, 250, 270])

### plot the tectonic boundaries 
# for segment in ridge_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
for segment in trench_gmt:
    if len(segment) > 1:
        lons, lats = zip(*segment)
        ax.plot(lons, lats,color='darkblue', linewidth=1.5,
                transform=ccrs.PlateCarree(), zorder=1)
        
# for segment in transform_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# print(blob_df.head(20))




# def plot_plate_motion(statlat,statlon,phi,dt,version=False):
#     # ax.plot(statlon,statlat, 'y', marker = 'v',markersize=6, transform=ccrs.PlateCarree())
#     phi = np.radians(90 - phi)
#     half_dt = dt / 2 

#     #halfway points of each line
#     dx = half_dt * np.cos(phi)
#     dy = half_dt * np.sin(phi)

#     x0 = statlon - dx
#     x1 = statlon + dx
#     y0 = statlat - dy
#     y1 = statlat + dy

#     if version:
#         ax.plot([x0, x1], [y0, y1], color='blue', linewidth=2, transform=ccrs.PlateCarree())
#     else:
#         ax.plot([x0, x1], [y0, y1], color='black', linewidth=1, transform=ccrs.PlateCarree())
# df_sorted = df.sort_values(by='column_name')

# platedf['lat_bin'] = (platedf['lat_dd'] / 2).round()
# platedf['lon_bin'] = (platedf['lon_dd'] / 2).round()

# platedf_sparse = platedf.drop_duplicates(
#     subset=['lat_bin', 'lon_bin']
# )

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

# ax.quiver(
#     plate_df_binned['lon_dd'].to_numpy(),
#     plate_df_binned['lat_dd'].to_numpy(),
#     plate_df_binned['u'].to_numpy(),
#     plate_df_binned['v'].to_numpy(),
#     scale=20,
#     width=0.003,
#     color='grey',
#     alpha=0.5,
#     headwidth=4,
#     headlength=2.5,
#     headaxislength=2.0,
#     zorder=20
# )

# Define geographic bounds for each plate
# Adjust these coordinates to match your three plates
plate1_mask = (plate_df_binned['lat_dd'] >= -35) & (plate_df_binned['lat_dd'] <= -25) & \
              (plate_df_binned['lon_dd'] >= 105) & (plate_df_binned['lon_dd'] <= 110)

plate2_mask = (plate_df_binned['lat_dd'] >= -5) & (plate_df_binned['lat_dd'] <= 5) & \
              (plate_df_binned['lon_dd'] >= 100) & (plate_df_binned['lon_dd'] <= 120)

plate3_mask = (plate_df_binned['lat_dd'] >= -10) & (plate_df_binned['lat_dd'] <= 5) & \
              (plate_df_binned['lon_dd'] >= 160) & (plate_df_binned['lon_dd'] <= 180)

# Get one point from each region (e.g., the centroid or first point)
plate1_point = plate_df_binned[plate1_mask].iloc[[0]]  # or .mean() for centroid
plate2_point = plate_df_binned[plate2_mask].iloc[[0]]
plate3_point = plate_df_binned[plate3_mask].iloc[[0]]

# Combine into one dataframe
plate_df_three = pd.concat([plate1_point, plate2_point, plate3_point])


print('here is the data for plate motion of australia')
print(plate1_point)


plt.quiver(
    plate_df_three['lon_dd'].to_numpy(),
    plate_df_three['lat_dd'].to_numpy(),
    plate_df_three['u'].to_numpy(),
    plate_df_three['v'].to_numpy(),
    transform=ccrs.PlateCarree(),
    angles='xy',
    scale_units='xy',
    scale=10,
    width=0.009,
    color='limegreen',
    alpha=0.9,
    headwidth=2,
    headlength=2,
    headaxislength=2.0,
    zorder=20
)

# plate_df = plate_df.sort_values(by=['lon_dd','lat_dd']).reset_index(drop=True)
# print(plate_df.head(20))

# plate_df['lat_bin'] = np.floor(plate_df['lat_dd'] / 3)#.round()
# plate_df['lon_bin'] = np.floor(plate_df['lon_dd'] / 3)#.round()
# plate_df_sparse = plate_df.drop_duplicates(
#     subset=['lat_bin', 'lon_bin']
# )
# # plate_df = plate_df.sort_values(by='lat_dd').reset_index(drop=True)

# print(plate_df_sparse.head(20))
# # plate_df.iloc[1::2]

# plate_df = plate_df_sparse
# plt.quiver(plate_df['lon_dd'], plate_df['lat_dd'], plate_df['u'], plate_df['v'],
#            angles='xy', scale_units='xy', scale=50, color='grey',alpha=0.9,
#            headwidth=3,headlength=3,headaxislength=3.5)

print(new_data_df.head(5))

# # df = new_data_df
plotted_points=[]

for i in range(len(new_data_df)):
    plot_station_splitting(new_data_df['Station'].iloc[i],
                           new_data_df['Latitude'].iloc[i],
                           new_data_df['Longitude'].iloc[i],
                           new_data_df['phi'].iloc[i],
                           4*new_data_df['dt'].iloc[i],
                           ax=ax,
                           version='New'
                           )


for i in range(len(legend_df)):
    plot_station_splitting(legend_df['Station'].iloc[i],
                           legend_df['Latitude'].iloc[i],
                           legend_df['Longitude'].iloc[i],
                           legend_df['phi'].iloc[i],
                           4*legend_df['dt'].iloc[i],
                           ax=ax,
                           version='Legend')

# plot stations that are complex and have no resolve 2 phi splitting parameters
for i in range(0,len(stations_to_plot_complex)):
    station = stations_to_plot_complex.iloc[i]

    ax.plot(station['evstationlon'],station['evstationlat'],
                    marker='o',
                    markersize=10,
                    markerfacecolor='none',
                    markeredgecolor='dimgrey',
                    markeredgewidth=5.0,
                    linestyle='None',
                    alpha=0.8,transform=ccrs.PlateCarree())  

# plt.title(f"CMB Pierce Points w/ SEMUCB-WM1 : {slice_depth} km",fontsize=25)
# plt.title(f"CMB Pierce Points w/ Diff SI",fontsize=25)
# plt.title(f"Lithospheric Thickness and APM",fontsize=25)


# fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/lithos_and_APM_map_aug12.png', dpi=720,bbox_inches="tight", pad_inches=0.2) #original dpi=300
fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/lithos_and_APM_map_sep_5.pdf', bbox_inches="tight", pad_inches=0.2) #original dpi=300

# print(np.mean(blob_df['longitude']))
# print(np.mean(blob_df['latitude']))


# #PacificSplittingMapCoordinates
# minlon, maxlon = 90.00, 190.00
# minlat, maxlat = -50.00, 10.00

# #australia coordinates
# minlon, maxlon = 110.00, 160.00
# minlat, maxlat = -45.00, -10.00

# # indonesia coordinates
# minlon, maxlon = 90.00, 135.00
# minlat, maxlat = -15.00, 10.00

# #WestPacificSplittingMapCoordinates:
# minlon, maxlon = 70.00, 170.00
# minlat, maxlat = -60.00, 70.00

# #SEAsiaSplittingMapCoordinates:
# minlon, maxlon = 60.00, 160.00
# minlat, maxlat = -20.00, 20.00

# #USASplittingMapCoordinates:
# minlon, maxlon = -132.89, -66.09
# minlat, maxlat = 22.62, 51.64