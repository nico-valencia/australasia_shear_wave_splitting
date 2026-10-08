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

AND ONCE AGAIN ANOTHER COPY TO MAKE FIGURE 1 FOR THE PAPER

will include: 
- all stations used in the study (those with enough data at least)
- all events used (even the cluster measurements I later remove)
'''

#PacificSplittingMapCoordinates
minlon, maxlon = 45.00, 270.00
minlat, maxlat = -55.00, 55.00

# #blob region extent
# minlon, maxlon = 155.00, 166.00
# minlat, maxlat = -51.00, -36.00



slice_depth = 2800 #km

##### Data Paths #####

# measurements that I made
new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_2.txt'

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


##### DataFrames #####
                                
new_data_df = pd.read_csv(new_measurements_path, sep='\t', encoding='cp1252') 
new_data_df[['Latitude','Longitude','phi','dt']] = new_data_df[['Latitude','Longitude','phi','dt']].apply(pd.to_numeric, errors='coerce')
new_data_df['Longitude'] = new_data_df['Longitude'] % 360

events_df = pd.read_csv(events_path, sep='\t')
print(len(events_df))
events_df = events_df[events_df['evstation'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not
print(len(events_df))

events_df["phase_map"] = events_df["evstation"].map(STATION_PHASE_MAP)
print(events_df.head(10))

stations_to_plot = (
    events_df[['evstation', 'evnetwork', 'evstationlat', 'evstationlon', 'phase_map']]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"length of events df {len(events_df)}")
events_df = events_df.drop_duplicates(subset=['evtime'])
print(f"length after culling df {len(events_df)}")

print(stations_to_plot.head(10))
print(len(stations_to_plot))


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
stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not

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

# ###########


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
ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=135,central_latitude=-25.0))
ax.set_global()
# ax.set_extent([minlon,maxlon,minlat,maxlat], crs=ccrs.AzimuthalEquidistant())
# ax.gridlines(draw_labels=True,linewidth=0)

import matplotlib.ticker as mticker

# gl = ax.gridlines(draw_labels=True, linewidth=0.25)

# remove duplicate dateline label
# gl.xlocator = mticker.FixedLocator([80,100, 120, 140, 160, 180])

# optional latitude ticks
# gl.ylocator = mticker.FixedLocator([-60,-50,-40,-30,-20,-10, 0, 10])

# turn off top/right labels
# gl.top_labels = False
# gl.right_labels = False

# gl.xlabels_top = False
# gl.ylabels_right = False
ax.coastlines()
# ax.stock_img()

good_station = bad_station = False

for i in range(0,len(stations_to_plot)):
    station = stations_to_plot.iloc[i]

    if station['phase_map'] == 'complex':
        label = 'Complex' if not bad_station else None
        bad_station = True
        ax.plot(station['evstationlon'],station['evstationlat'],
                marker='o',
                markersize=5,
                markerfacecolor='None',
                markeredgecolor='dimgrey',
                markeredgewidth=1.5,
                linestyle='None',
                label=label,
                alpha=0.8,transform=ccrs.PlateCarree())   

    else:
        label = 'Single Layer' if not good_station else None
        good_station = True
        ax.plot(station['evstationlon'],station['evstationlat'],
                        marker='v',
                        markersize=5,
                        markerfacecolor='gold',
                        markeredgecolor='black',
                        markeredgewidth=1.0,
                        linestyle='None',
                        label=label,
                        alpha=0.8,transform=ccrs.PlateCarree())  


# for i in range(0,len(events_df)):
#     event = events_df.iloc[i] 
#     if event['evphase'] == 'SKS':
#             ax.plot(event['evlon'],event['evlat'],
#                     marker='*',
#                 markersize=5,
#                 markerfacecolor='black',
#                 markeredgecolor='black',
#                 markeredgewidth=0.25,
#                 linestyle='None',
#                 alpha=0.7,transform=ccrs.PlateCarree())    
#     if event['evphase'] == 'S':
#         if event['evdp'] >= 400:
#             ax.plot(event['evlon'],event['evlat'],
#                     marker='*',
#                 markersize=5,
#                 markerfacecolor='blue',
#                 markeredgecolor='black',
#                 markeredgewidth=0.25,
#                 linestyle='None',
#                 alpha=0.7,transform=ccrs.PlateCarree())
#         else:
#             ax.plot(event['evlon'],event['evlat'],
#                     marker='*',
#                 markersize=5,
#                 markerfacecolor='orange',
#                 markeredgecolor='black',
#                 markeredgewidth=0.25,
#                 linestyle='None',
#                 alpha=0.7,transform=ccrs.PlateCarree())

sks_plotted = s_deep_plotted = s_shallow_plotted = False


for i in range(0,len(events_df)):
    event = events_df.iloc[i] 
    if event['evphase'] == 'SKS':
            label = 'SKS' if not sks_plotted else None
            sks_plotted = True
            ax.scatter(event['evlon'],event['evlat'],s=8*event['evmg'],
                    marker='*',
                    c='black',
                    label=label,
                # markersize=5,
                # markerfacecolor='black',
                # markeredgecolor='black',
                # markeredgewidth=0.25,
                # linestyle='None',
                alpha=0.7,transform=ccrs.PlateCarree())    
    if event['evphase'] == 'S':
        if event['evdp'] >= 400:
            label = 'Deep S' if not s_deep_plotted else None
            s_deep_plotted = True
            ax.scatter(event['evlon'],event['evlat'],s=8*event['evmg'],
                    marker='*',
                    c='blue',
                    label=label,
                # markersize=5,
                # markerfacecolor='blue',
                # markeredgecolor='black',
                # markeredgewidth=0.25,
                # linestyle='None',
                alpha=0.7,transform=ccrs.PlateCarree())
        else:
            label = 'Shallow S' if not s_shallow_plotted else None
            s_shallow_plotted = True
            ax.scatter(event['evlon'],event['evlat'],s=5*event['evmg'],
                    marker='*',
                    c='red',
                    label=label,
                # markersize=5,
                # markerfacecolor='orange',
                # markeredgecolor='black',
                # markeredgewidth=0.25,
                # linestyle='None',
                alpha=0.7,transform=ccrs.PlateCarree())


ax.legend(fontsize=15, loc='upper left', framealpha=0.9)



# label = 'SKS phase' if not sks_plotted else None
#             sks_plotted = True
####################################

    
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

# for i in range(len(new_data_df)):
#     plot_station_splitting(new_data_df['Station'].iloc[i],
#                            new_data_df['Latitude'].iloc[i],
#                            new_data_df['Longitude'].iloc[i],
#                            new_data_df['phi'].iloc[i],
#                            4*new_data_df['dt'].iloc[i],
#                            ax=ax,
#                            version='New'
#                            )

# for i in range(len(legend_df)):
#     plot_station_splitting(legend_df['Station'].iloc[i],
#                            legend_df['Latitude'].iloc[i],
#                            legend_df['Longitude'].iloc[i],
#                            legend_df['phi'].iloc[i],
#                            4*legend_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Legend')

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
# plt.title(f"Stations and Events",fontsize=25)


fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/stations_and_events_map_sep_5.pdf', bbox_inches="tight", pad_inches=0.2) #original dpi=300
# fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/stations_and_events_map_sep_4.png', dpi=600,bbox_inches="tight", pad_inches=0.2) #original dpi=300
