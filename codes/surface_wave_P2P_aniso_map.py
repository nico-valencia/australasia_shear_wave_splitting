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
from helper_functions.plotting import read_gmt_file, read_and_filter_gmt, plot_station_splitting, build_cmap, subsample_anisotropy_grid
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import matplotlib.colors as colors
import re
from support import dms_to_dd
from station_keys import STATION_PHASE_MAP
from scipy.interpolate import griddata
from rasterio.enums import Resampling

from scipy.io import netcdf





'''
WILL BE SWITCHING FROM -180 TO 180 LON --> 0 TO 360 LON
MODULO 360 EVERY LONGITUDE DATA

COPY OF OTHER CUZ I AM LAZY
THIS IS FOR PRAGUE SPLITTING MAPS I MADE

AND NOW IT IS FOR MAKING SURFACE WAVE SPLITTING MAPS AT VARIOUS DEPTHS AS DISCUSSED WITH JONATHAN
this will be for sticks of peak 2 peak anisotropy, not just Vs

'''

#PacificSplittingMapCoordinates
minlon, maxlon = 90.00, 180.00
minlat, maxlat = -55.00, 5.00

# #blob region extent
# minlon, maxlon = 155.00, 166.00
# minlat, maxlat = -51.00, -36.00



##### Data Paths #####

# measurements that I made
new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_2.txt'

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

surface_wave_data_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/3DLGL-TPESv/3DLGL-TPESv-depth.v2022-11.r0.0.nc'

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



surface_wave_data = netcdf.NetCDFFile(surface_wave_data_path,'r')
surf_wave_latitudes = np.array(surface_wave_data.variables['latitude'][:].copy(), dtype=np.float64)
surf_wave_longitudes = np.array(surface_wave_data.variables['longitude'][:].copy() , dtype=np.float64)
surf_wave_depths = np.array(surface_wave_data.variables['depth'][:].copy(), dtype=np.float64)
surf_wave_dvs = np.array(surface_wave_data.variables['dvs'][:].copy()  , dtype=np.float64)           # dVs anomaly
surf_wave_az = np.array(surface_wave_data.variables['az'][:].copy()   , dtype=np.float64)              # azimuth of splitting
surf_wave_as = np.array(surface_wave_data.variables['as'][:].copy()   , dtype=np.float64)              # peak 2 peak anisotropy
surface_wave_data.close()

surf_wave_dvs = np.ma.masked_where(surf_wave_dvs == 99999.0, surf_wave_dvs) # fill in broken values with 0
surf_wave_as = np.ma.masked_where(surf_wave_as == 99999.0, surf_wave_as) # fill in broken values with 0





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



####### u

new_data_df = new_data_df[(minlon < new_data_df['Longitude']) & (new_data_df['Longitude'] < maxlon)]
new_data_df = new_data_df[(minlat < new_data_df['Latitude']) & (new_data_df['Latitude'] < maxlat)]

legend_df = pd.DataFrame([{
    'Station': r'$\mathbf{\delta t}$ = 1.0',
    'Latitude': -45,
    'Longitude': 100,
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


gl.xlabel_style = {'size': 20}  # Change 15 to whatever size you want
gl.ylabel_style = {'size': 20}

# remove duplicate dateline label
gl.xlocator = mticker.FixedLocator([80,100, 120, 140, 160, 180])

# optional latitude ticks
gl.ylocator = mticker.FixedLocator([-60,-50,-40,-30,-20,-10, 0, 10])

# turn off top/right labels
gl.top_labels = False
gl.right_labels = False

gl.xlabels_top = False
gl.ylabels_right = False
ax.coastlines()

# ##### choose basemap: lithospheric thickness or bathymetry data or diffSI data or surface wave data

### surface wave data

target_depth = 400 # km
depth_idx = np.argmin(np.abs(surf_wave_depths - target_depth))
actual_depth = surf_wave_depths[depth_idx]

print(f"Requested depth: {target_depth} km")
print(f"Actual depth: {actual_depth} km")
print(f"Depth index: {depth_idx}")

# EXTRACT 2D HORIZONTAL SLICE
slice_dvs_vals = surf_wave_dvs[depth_idx, :, :]  # Shape: (latitude, longitude)
slice_az_vals = surf_wave_az[depth_idx, :, :]  # Shape: (latitude, longitude)
slice_as_vals = surf_wave_as[depth_idx, :, :]  # Shape: (latitude, longitude)


print(f"Slice shape: {slice_dvs_vals.shape}")
print(f"Data range: {slice_dvs_vals.min():.2f} to {slice_dvs_vals.max():.2f}")

# CREATE MESHGRID
lon2, lat2 = np.meshgrid(surf_wave_longitudes, surf_wave_latitudes)

slice_vmin_cbar_min = round(slice_dvs_vals.min() * 0.9,1)
slice_vmin_cbar_max = round(slice_dvs_vals.max() * 0.9,1)

print(slice_vmin_cbar_min,slice_vmin_cbar_max)

# Flatten all arrays
lat_flat = lat2.flatten()
lon_flat = lon2.flatten()
as_flat = slice_as_vals.flatten()
az_flat = slice_az_vals.flatten()

# Create initial DataFrame
surface_df = pd.DataFrame({
    'lat': lat_flat,
    'lon': lon_flat,
    'phi': az_flat,  # azimuth in degrees (your phi)
    'dt': as_flat,   # anisotropy magnitude (your dt equivalent)
    'depth': actual_depth
})

print(f"Initial data points: {len(surface_df)}")

# Remove missing/masked values
# # Check for NaN or masked values
# valid_mask = ~(np.isnan(df['phi']) | np.isnan(df['dt']) | 
#                 (df['phi'] == 99999.0) | (df['dt'] == 99999.0))

# df_clean = df[valid_mask].copy()
# print(f"After removing missing values: {len(df_clean)}")


surface_df = surface_df[(minlon < surface_df['lon']) & (surface_df['lon'] < maxlon)]
surface_df = surface_df[(minlat < surface_df['lat']) & (surface_df['lat'] < maxlat)]
surface_df = surface_df.reset_index(drop=True)


print(surface_df.head(10))

print(surface_df['dt'].min(),surface_df['dt'].max())

surface_df = subsample_anisotropy_grid(surface_df, lat_stride=4, lon_stride=4)

#### remove some measurements so the legend plots clearly

# Set your legend position and exclusion distance
legend_lat = -45
legend_lon = 103  
exclusion_degrees = 5

# Calculate distances and create exclusion mask
lat_diff = np.abs(surface_df['lat'] - legend_lat)
lon_diff = np.abs(surface_df['lon'] - legend_lon)
exclusion_mask = (lat_diff <= exclusion_degrees) & (lon_diff <= exclusion_degrees)

# Keep points outside the exclusion zone
surface_df = surface_df[~exclusion_mask].copy()

############

surface_legend_df = pd.DataFrame([{
    'Station': r'P2P = 5%',  # or whatever label you want
    'lat': -49,              # Position it near your other legend
    'lon': 100, 
    'phi': 90,               # Direction for the legend stick
    'dt': 5,                 # Anisotropy value for the legend
    'depth': actual_depth
}])


# im = ax.pcolormesh(
#     lon2,
#     lat2,
#     slice_vals,
#     cmap="RdBu",
#     shading="auto",
#     vmin=slice_vmin_cbar_min,
#     vmax=slice_vmin_cbar_max,
#     transform=ccrs.PlateCarree()
# )

# cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
#                        pad=0.05, shrink=0.5, aspect=25)
# cbar.set_label("dVs (%)", fontsize=15)



    
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
        

plotted_points=[]

for i in range(len(new_data_df)):
    plot_station_splitting(new_data_df['Station'].iloc[i],
                           new_data_df['Latitude'].iloc[i],
                           new_data_df['Longitude'].iloc[i],
                           new_data_df['phi'].iloc[i],
                           4*new_data_df['dt'].iloc[i],
                           ax=ax,
                           version='BoldNew'
                           )

for i in range(len(legend_df)):
    plot_station_splitting(legend_df['Station'].iloc[i],
                           legend_df['Latitude'].iloc[i],
                           legend_df['Longitude'].iloc[i],
                           legend_df['phi'].iloc[i],
                           4*legend_df['dt'].iloc[i],
                           ax=ax,
                           version='Legend')
    
for i in range(len(surface_df)):
    plot_station_splitting(legend_df['Station'].iloc[0],
                           surface_df['lat'].iloc[i],
                           surface_df['lon'].iloc[i],
                           surface_df['phi'].iloc[i],
                           1*surface_df['dt'].iloc[i],
                           ax=ax,
                           version='Surface',
                           annotate=False)
    
for i in range(len(surface_legend_df)):
    plot_station_splitting(surface_legend_df['Station'].iloc[i],
                           surface_legend_df['lat'].iloc[i],
                           surface_legend_df['lon'].iloc[i],
                           surface_legend_df['phi'].iloc[i],
                           1*surface_legend_df['dt'].iloc[i],  # Same scaling as your surface data
                           ax=ax,
                           version='SurfaceLegend',  # You might need to add this version to your function
                           annotate=True)

# # # print(stat_avg_df.head(5))
# for i in range(len(stat_avg_df)):
#     plot_station_splitting(stat_avg_df['station'].iloc[i],
#                            stat_avg_df['lat'].iloc[i],
#                            stat_avg_df['lon'].iloc[i],
#                            stat_avg_df['azi'].iloc[i],
#                            4*stat_avg_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Old',
#                            annotate=False)
    
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


# plt.title(f"CMB Pierce Points w/ SEMUCB-WM1 : {slice_depth} km",fontsize=25)
# plt.title(f"CMB Pierce Points w/ Diff SI",fontsize=25)
# plt.title(f"Surface Wave Vs Data: {target_depth} km",fontsize=25)
plt.title(f"Peak-to-Peak Anisotropy: {target_depth} km",fontsize=25)


fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/surface_wave_p2p_map_{target_depth}km.png', dpi=720,bbox_inches="tight", pad_inches=0.2) #original dpi=300

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