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

# nameoffilenewdata = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/splittingDB3.txt'
nameoffilenewdata = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/swp_KA_saved_SI_measurments_may15.txt'
nameoffile = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/splittingDB.txt'

df = pd.read_csv(nameoffile, sep='|', encoding='cp1252')
dfnewdata = pd.read_csv(nameoffilenewdata, sep='|', encoding='cp1252') #NEW DATA

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
minlat, maxlat = -50.00, 50.00

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
print(platedf.head())

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

print(platedf.head())

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


print(platedf.head())


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

print(sks_meas_all.head())
print(sks_meas_all_newdata.head())

print(len(df))
print(len(sks_meas_all_newdata))


# Create a figure and axis for the map
fig = plt.figure(figsize=(20, 10))
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=140))
# ax.set_extent([70, 160, -50, 10], crs=ccrs.Mollweide())

ax = plt.axes(projection=ccrs.PlateCarree()) #central_longitude=140))
# ax.set_extent([90,180, minlat, maxlat], crs=ccrs.PlateCarree())

# asutralia 
# ax.set_extent([110,160,-45,-10], crs=ccrs.PlateCarree())

#swp
# ax.set_extent([90,180,-50,20], crs=ccrs.PlateCarree())

#indonesia
ax.set_extent([90,135,-15,10], crs=ccrs.PlateCarree())



# 90.00, 190.00
# fig = plt.figure(figsize=(20, 10))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())


# minlon, maxlon = 90.00, 190.00
# minlat, maxlat
# ax.set_global()
ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.gridlines(draw_labels=True,linewidth=0)


def plot_station_splitting(statlat,statlon,phi,dt,version=False):
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

    if version:
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=2, transform=ccrs.PlateCarree())
    else:
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=1, transform=ccrs.PlateCarree())

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


plt.quiver(platedf['lon_dd'], platedf['lat_dd'], platedf['u'], platedf['v'],
           angles='xy', scale_units='xy', scale=50, color='grey',alpha=0.5)

df = sks_meas_all
for i in range(len(df)):
    plot_station_splitting(df['Latitude'].iloc[i],df['Longitude'].iloc[i],df['phi'].iloc[i],2*df['dt'].iloc[i])

df = sks_meas_all_newdata
for i in range(len(df)):
    plot_station_splitting(df['Latitude'].iloc[i],df['Longitude'].iloc[i],df['phi'].iloc[i],4*df['dt'].iloc[i],version=True)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Combine both datasets if you want to show all measurements
combined_df = pd.concat([sks_meas_all_newdata])

# Set colormap and normalization
cmap = plt.cm.gist_rainbow  # Or 'plasma', 'coolwarm', etc.
norm = Normalize(vmin=combined_df['dt'].min(), vmax=combined_df['dt'].max())

# Plot scatter points colored by dt
sc = ax.scatter(combined_df['Longitude'], combined_df['Latitude'],
                c=2*combined_df['dt'], cmap=cmap, norm=norm,
                s=20, edgecolor='k', linewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=5)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cax = inset_axes(ax,
#                  width="50%",  # width of the colorbar
#                  height="3%",  # height
#                  loc='lower center',
#                  bbox_to_anchor=(0.5, -0.1, 0.5, 1),  # (x0, y0, width, height)
#                  bbox_transform=ax.transAxes,
#                  borderpad=0)

# Add a colorbar
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.3, pad=0.05)
cbar.set_label('Delay Time (s)', fontsize=14)

# plt.title("Indonesia Splitting Results",fontsize=25)
fig.savefig('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/test_map.png', dpi=720) #original dpi=300