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
from helper_functions.plotting import read_gmt_file, read_and_filter_gmt

STATION_PHASE_MAP = {
    "ARMA": "sks_deep", 
    "AS31": "sks", 
    "BBOO": "sks_deep",
    "BLDU": "sks",
    "COEN": "sks",
    "CTAO": "sks",
    "EIDS": "sks_deep",
    "MNAI": "all",
    "MTN": "sks_deep",
    "MULG": "all",
    "NOUC": "sks_deep",
    "QIS": "sks",
    "SNZO": "all",
    "STKA": "sks_deep",
    "TOO": "sks_deep",
    "XMI": "all",
    "XMIS": "all",}

# nameoffilenewdata = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/splittingDB3.txt'
# nameoffilenewdata = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/swp_KA_saved_SI_measurments_may15.txt'
nameoffilenewdata = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements.txt'
nameoffile = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/splittingDB.txt'

df = pd.read_csv(nameoffile, sep='|', encoding='cp1252')
dfnewdata = pd.read_csv(nameoffilenewdata, sep='\t', encoding='cp1252') #NEW DATA
# dfnewdata = dfnewdata.loc[:, ~dfnewdata.columns.str.contains('^Unnamed')]

# Bernhard Steinberger lithospheric thickness model
lithos_df = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/lithosphere_thickness/lith_ave_no_slabs.d',
                        header=None,
                        delimiter=r'\s+',
                        names=['longitude', 'latitude', 'thickness'],
                        low_memory=False)

# Thorsten Becker SKS station average database (as of May 7 2026)
stat_avg_df = pd.read_csv("/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/sks_stat_avg/jsg_geodynamics.sks.avg.table.txt",
                          header=None,
                          delimiter=r'\s+',
                          names=["station","lat","lon","azi","d_azi","dt","d_dt","type","elat","elon","edep","mlat","mlon","bazi"]
                          )
stat_avg_df['station_clean'] = stat_avg_df['station'].str.split('-', n=1).str[0]
stat_avg_df['num_of_splits'] = stat_avg_df['station'].str.extract(r'avg(\d+)').astype(int)
# stat_avg_df = stat_avg_df[stat_avg_df['num_of_splits']>1]

keep_idx = 5481

base_station = (
    stat_avg_df.loc[keep_idx, 'station']
    .split('-')[0]
)

stat_avg_df = stat_avg_df[
    (stat_avg_df['station'].str.split('-').str[0] != base_station)
    | (stat_avg_df.index == keep_idx)
]

stat_avg_df = (
    stat_avg_df
    .sort_values('num_of_splits', ascending=False)
    .drop_duplicates(subset='station_clean')
    .reset_index(drop=True)
)

stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(STATION_PHASE_MAP.keys())]


# print(stat_avg_df.head(20))
# sys.exit()


# ridge_gmt = read_gmt_file('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/ridge.gmt')
# transform_gmt = read_gmt_file('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/transform.gmt')
# trench_gmt = read_gmt_file('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/trench.gmt')

ridge_gmt = read_and_filter_gmt('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/ridge.gmt', 
                                90, 190, -50, 50, max_gap=20)

transform_gmt = read_and_filter_gmt('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/transform.gmt', 
                                90, 190, -50, 50, max_gap=20)

trench_gmt = read_and_filter_gmt('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/trench.gmt', 
                                90, 190, -50, 50, max_gap=20)


# Add your own column names
# lithos_df.columns = ['longitude', 'latitude', 'thickness']
# print(lithos_df.head())

# print(stat_avg_df.head())

# print(len(stat_avg_df))

# sys.exit()
# print(df)
# print(df['Latitude'])

#old data
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['phi'] = pd.to_numeric(df['phi'], errors='coerce')
df['dt'] = pd.to_numeric(df['dt'], errors='coerce')

#New Data
dfnewdata['Latitude'] = pd.to_numeric(dfnewdata['Latitude'], errors='coerce')
dfnewdata['Longitude'] = pd.to_numeric(dfnewdata['Longitude'], errors='coerce')
dfnewdata['phi'] = pd.to_numeric(dfnewdata['phi'], errors='coerce')
dfnewdata['dt'] = pd.to_numeric(dfnewdata['dt'], errors='coerce')

#PacificSplittingMapCoordinates
minlon, maxlon = 90.00, 190.00
minlat, maxlat = -50.00, 10.00

# #australia coordinates
# minlon, maxlon = 110.00, 160.00
# minlat, maxlat = -45.00, -10.00

# # indonesia coordinates
# minlon, maxlon = 90.00, 135.00
# minlat, maxlat = -15.00, 10.00


import re

# plate motion data 
platedf = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/pm10242.csv',sep=',',header=None,names=['station','lat', 'lon', 'rate','angle','a','b','c'])
# platedf = platedf[platedf['c']=='AU(NNR)']# platedf = platedf.dropna()
# print(platedf.head())

# def dms_to_dd(dms_str):
#     """
#     Convert a DMS string like '88 1\' 3.57" S' to decimal degrees.
#     """
#     match = re.match(r"(\d+)[°\s]+(\d+)[\'\s]+([\d.]+)\"?\s*([NSEW])", dms_str.strip())
#     if not match:
#         return None  # Could not parse
#     deg, minute, second, direction = match.groups()
#     dd = float(deg) + float(minute) / 60 + float(second) / 3600
#     if direction in ['S', 'W']:
#         dd *= -1
#     return dd

# def dms_to_dd(dms_str):
#     dms_str = dms_str.strip().replace('′', "'").replace('″', '"')  # normalize quotes
#     # Try to match DMS
#     match = re.match(r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)[^\d]*([NSEW])", dms_str)
#     if match:
#         deg, minute, second, direction = match.groups()
#         dd = float(deg) + float(minute) / 60 + float(second) / 3600
#     else:
#         # Try to match just degree and direction (e.g., "90 S")
#         match_simple = re.match(r"(\d+)\s*([NSEW])", dms_str)
#         if match_simple:
#             dd = float(match_simple.group(1))
#             direction = match_simple.group(2)
#         else:
#             return None  # Failed to parse

#     if direction in ['S', 'W']:
#         dd *= -1
#     return dd

def dms_to_dd(dms_str):
    # Remove extraneous quotes and whitespace
    dms_str = dms_str.strip().replace('"', '').replace('′', "'").replace('″', '"')

    # Match DMS pattern like: 57 51' 26.19" S
    match = re.match(r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\s*([NSEW])", dms_str)
    if match:
        deg, minute, second, direction = match.groups()
        dd = float(deg) + float(minute) / 60 + float(second) / 3600
    else:
        # Try to match simpler patterns like "57 S"
        match_simple = re.match(r"(\d+)\s*([NSEW])", dms_str)
        if match_simple:
            dd = float(match_simple.group(1))
            direction = match_simple.group(2)
        else:
            return None  # Unable to parse

    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Convert to decimal degrees
platedf['lat_dd'] = platedf['lat'].apply(dms_to_dd)
platedf['lon_dd'] = platedf['lon'].apply(dms_to_dd)

# print(platedf.head())

# sys.exit()

# Drop any row that didn't parse correctly
# platedf = platedf.dropna(subset=['lat_dd', 'lon_dd'])

# Compute u (east) and v (north) components
import numpy as np
platedf['angle_rad'] = np.deg2rad(platedf['angle'])

platedf = platedf[(platedf['lat_dd'] >= minlat) & (platedf['lat_dd'] <= maxlat) &
                  (platedf['lon_dd'] >= minlon) & (platedf['lon_dd'] <= maxlon)]

# platedf = platedf[(platedf['lat_dd'] >= minlat) & (platedf['lat_dd'] <= maxlat)]

platedf['u'] = platedf['rate'] * np.sin(platedf['angle_rad'])  # east component
platedf['v'] = platedf['rate'] * np.cos(platedf['angle_rad'])  # north component


# print(platedf.head())


# sys.exit()
# #WestPacificSplittingMapCoordinates:
# minlon, maxlon = 70.00, 170.00
# minlat, maxlat = -60.00, 70.00

# #SEAsiaSplittingMapCoordinates:
# minlon, maxlon = 60.00, 160.00
# minlat, maxlat = -20.00, 20.00

# #USASplittingMapCoordinates:
# minlon, maxlon = -132.89, -66.09
# minlat, maxlat = 22.62, 51.64



dftmp1 = df[(minlon < df['Longitude']) & (df['Longitude'] < maxlon)]
sks_meas_all = dftmp1[(minlat < df['Latitude']) & (df['Latitude'] < maxlat)]
sks_meas_all.reset_index(inplace=True)

# #New Data
dftmp1newdata = dfnewdata[(minlon < dfnewdata['Longitude']) & (dfnewdata['Longitude'] < maxlon)]
sks_meas_all_newdata = dftmp1newdata[(minlat < dfnewdata['Latitude']) & (dfnewdata['Latitude'] < maxlat)]
sks_meas_all_newdata.reset_index(inplace=True)

stat_avg_df = stat_avg_df[(minlon < stat_avg_df['lon']) & (stat_avg_df['lon'] < maxlon)]
stat_avg_df = stat_avg_df[(minlat < stat_avg_df['lat']) & (stat_avg_df['lat'] < maxlat)]

print(len(stat_avg_df))

print(stat_avg_df)


# sys.exit()


# sys.exit()
# print(sks_meas_all.head())
# print(sks_meas_all_newdata.head())

# print(len(df))
# print(len(sks_meas_all_newdata))

# legend_df = pd.DataFrame(columns)
# print(sks_meas_all_newdata.head(4))

# ### add dummy measurement for legend
# sks_meas_all_newdata.loc[len(sks_meas_all_newdata)] = {
#     'index':len(sks_meas_all_newdata),
#     # 'index':len(sks_meas_all_newdata),
#     'Station': 'TEST',
#     'Latitude': -25,
#     'Longitude': 135,
#     'phi': 90,
#     'dt': 1
# }

legend_df = pd.DataFrame([{
    'Station': r'$\delta t$ = 1.0',
    'Latitude': -40,
    'Longitude': 105,
    'phi': 90,
    'dt': 1
}])

# fr'$\phi$ = {round(phi_binned)}°, $\delta t$ = {round(dt_binned,1)}s'

# Create a figure and axis for the map
fig = plt.figure(figsize=(20, 10))
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=140))
# ax.set_extent([70, 160, -50, 10], crs=ccrs.Mollweide())

ax = plt.axes(projection=ccrs.PlateCarree()) #central_longitude=140))
# ax.set_extent([90,180, minlat, maxlat], crs=ccrs.PlateCarree())

# asutralia 
# ax.set_extent([110,160,-45,-10], crs=ccrs.PlateCarree())

# swp
ax.set_extent([95,180,-50,5], crs=ccrs.PlateCarree())

# #indonesia
# ax.set_extent([90,135,-15,10], crs=ccrs.PlateCarree())



# 90.00, 190.00
# fig = plt.figure(figsize=(20, 10))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())


# minlon, maxlon = 90.00, 190.00
# minlat, maxlat
# ax.set_global()
ax.stock_img()
ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black')

# ax.add_feature(cfeature.NaturalEarthFeature(
#     'raster',
#     'shaded_relief',
#     '50m',  # or '10m' for higher resolution
#     edgecolor='none'
# ), zorder=0)

# # Add coastlines on top
# ax.coastlines(linewidth=0.8, color='black', zorder=2)

# # Plot each segment
# for segment in ridge_gmt:
#     ax.plot(segment['lon'], segment['lat'],
#             color='red',
#             linewidth=2,
#             transform=ccrs.PlateCarree(),
#             zorder=4)  # Above everything else
    
for segment in ridge_gmt:
    if len(segment) > 1:
        lons, lats = zip(*segment)
        ax.plot(lons, lats,
                color='darkred', linewidth=1.5,
                transform=ccrs.PlateCarree(), zorder=1)
        
for segment in trench_gmt:
    if len(segment) > 1:
        lons, lats = zip(*segment)
        ax.plot(lons, lats,
                color='darkred', linewidth=1.5,
                transform=ccrs.PlateCarree(), zorder=1)
        
for segment in transform_gmt:
    if len(segment) > 1:
        lons, lats = zip(*segment)
        ax.plot(lons, lats,
                color='darkred', linewidth=1.5,
                transform=ccrs.PlateCarree(), zorder=1)

# # Prepare data for contour plot
# # Convert to 2D grid if needed
# from scipy.interpolate import griddata

# # Create grid
# lon_grid = np.linspace(lithos_df['longitude'].min(), lithos_df['longitude'].max(), 200)
# lat_grid = np.linspace(lithos_df['latitude'].min(), lithos_df['latitude'].max(), 200)
# lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# # Interpolate thickness onto grid
# thickness_grid = griddata(
#     (lithos_df['longitude'], lithos_df['latitude']),
#     lithos_df['thickness'],
#     (lon_mesh, lat_mesh),
#     method='cubic'
# )

# # Plot filled contours
# levels = np.linspace(lithos_df['thickness'].min(), lithos_df['thickness'].max(), 20)
# contour = ax.contourf(lon_mesh, lat_mesh, thickness_grid,
#                       levels=levels,
#                       cmap='RdYlBu_r',  # Or 'viridis', 'plasma', 'jet'
#                       alpha=0.6,
#                       transform=ccrs.PlateCarree(),
#                       zorder=1)

# # Add colorbar for thickness
# cbar_thickness = plt.colorbar(contour, ax=ax, orientation='vertical', 
#                               pad=0.05, shrink=0.8)
# cbar_thickness.set_label('Lithosphere Thickness [km]', fontsize=14)



def plot_station_splitting(statname,statlat,statlon,phi,dt,version=None,annotate=True):
    # ax.plot(statlon,statlat, 'y', marker = 'v',markersize=6, transform=ccrs.PlateCarree())
    phi = np.radians(90 - phi)
    half_dt = dt / 2 

    #halfway points of each line
    dx = half_dt * np.cos(phi)
    dy = half_dt * np.sin(phi)

    x0 = statlon - dx
    x1 = statlon + dx
    y0 = statlat - dy
    y1 = statlat + dy

    if version is 'New':
        ax.plot([x0, x1], [y0, y1], color='blue', linewidth=3, transform=ccrs.PlateCarree(),zorder=10)
    if version is 'Legend':
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=3, transform=ccrs.PlateCarree())

    else:
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=0.5, transform=ccrs.PlateCarree(),alpha=0.7)

    if annotate:
        ax.annotate(
        statname,
        xy=(statlon,statlat),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=16,
        color='blue',
        zorder=8
        )

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
platedf['lat_bin'] = np.floor(platedf['lat_dd'] / spacing) * spacing
platedf['lon_bin'] = np.floor(platedf['lon_dd'] / spacing) * spacing

# Average vectors within bins
platedf_binned = (
    platedf
    .groupby(['lat_bin', 'lon_bin'], as_index=False)
    .agg({
        'u': 'mean',
        'v': 'mean',
        'lat_dd': 'mean',
        'lon_dd': 'mean'
    })
)

plt.quiver(
    platedf_binned['lon_dd'],
    platedf_binned['lat_dd'],
    platedf_binned['u'],
    platedf_binned['v'],
    angles='xy',
    scale_units='xy',
    scale=20,
    width=0.003,
    color='grey',
    alpha=0.5,
    headwidth=4,
    headlength=2.5,
    headaxislength=2.0
)

# platedf = platedf.sort_values(by=['lon_dd','lat_dd']).reset_index(drop=True)
# print(platedf.head(20))

# platedf['lat_bin'] = np.floor(platedf['lat_dd'] / 3)#.round()
# platedf['lon_bin'] = np.floor(platedf['lon_dd'] / 3)#.round()
# platedf_sparse = platedf.drop_duplicates(
#     subset=['lat_bin', 'lon_bin']
# )
# # platedf = platedf.sort_values(by='lat_dd').reset_index(drop=True)

# print(platedf_sparse.head(20))
# # platedf.iloc[1::2]

# platedf = platedf_sparse
# plt.quiver(platedf['lon_dd'], platedf['lat_dd'], platedf['u'], platedf['v'],
#            angles='xy', scale_units='xy', scale=50, color='grey',alpha=0.9,
#            headwidth=3,headlength=3,headaxislength=3.5)

df_averaged = sks_meas_all.groupby('Station')[['phi', 'dt','Latitude',"Longitude"]].mean().reset_index()


# df = df_averaged
# for i in range(len(df)):
#     plot_station_splitting(df['Latitude'].iloc[i],df['Longitude'].iloc[i],df['phi'].iloc[i],4*df['dt'].iloc[i])

df = sks_meas_all_newdata
for i in range(len(sks_meas_all_newdata)):
    plot_station_splitting(sks_meas_all_newdata['Station'].iloc[i],
                           sks_meas_all_newdata['Latitude'].iloc[i],
                           sks_meas_all_newdata['Longitude'].iloc[i],
                           sks_meas_all_newdata['phi'].iloc[i],
                           4*sks_meas_all_newdata['dt'].iloc[i],
                           version='New'
                           )

for i in range(len(legend_df)):
    plot_station_splitting(legend_df['Station'].iloc[i],
                           legend_df['Latitude'].iloc[i],
                           legend_df['Longitude'].iloc[i],
                           legend_df['phi'].iloc[i],
                           4*legend_df['dt'].iloc[i],
                           version='Legend')

# print(stat_avg_df.head(5))
for i in range(len(stat_avg_df)):
    plot_station_splitting(stat_avg_df['station'].iloc[i],
                           stat_avg_df['lat'].iloc[i],
                           stat_avg_df['lon'].iloc[i],
                           stat_avg_df['azi'].iloc[i],
                           4*stat_avg_df['dt'].iloc[i],
                           version='Legend',
                           annotate=False)
    


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Combine both datasets if you want to show all measurements
combined_df = pd.concat([sks_meas_all_newdata])

# Set colormap and normalization
cmap = plt.cm.viridis # Or 'plasma', 'coolwarm', etc.
norm = Normalize(vmin=0, vmax=1.5)

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

# plt.title("Indonesia Splitting Results",fontsize=25)
fig.savefig('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/test_map_boundaries.png', dpi=720) #original dpi=300